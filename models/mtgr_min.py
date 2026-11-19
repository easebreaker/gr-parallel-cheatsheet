"""MTGR 最小实现 —— 关键只有一件事：三段式掩码。

论文: "MTGR: Industrial-Scale Generative Recommendation Framework in Meituan"
      arXiv 2505.18654（CIKM'25）

=======================================================================
MTGR = HSTU block + 一个非因果的三段掩码 + GLN
=======================================================================

token 序列被切成三段（顺序即拼接顺序）：

    |<---- 静态 S ---->|<-- 实时 R -->|<- 候选 K ->|
    U(用户画像) S(历史)   R(当天行为)     C/I(曝光样本)

三条掩码规则（论文原文的三条，逐条对应）：

  1. 静态段对【所有】token 可见     —— U/S 来自聚合窗口之前，不会泄漏
  2. 实时段内部【因果】             —— 只能看更早的实时行为
  3. 候选段【只看自己】             —— 候选之间不能互相偷看

第 3 条最容易被忽略：如果不隔离，同一批曝光样本里后发生的那个会
把答案泄漏给先发生的那个，线上 AUC 会在训练时虚高、上线后崩掉。

=======================================================================
为什么这个掩码是 CP 分析的入口
=======================================================================
pair 数（闭式，可被暴力掩码逐位验证）：

    pairs = S²  +  [P(P+1) − S(S+1)]/2  +  K·(P+1)
            ^         ^                      ^
         静态方阵   实时因果三角        候选看整段前缀+自己
    P = S + R

三段的 CP 性质完全不同：

    静态 S²   ——  方阵。工作量对任何切法都均匀，失衡度 = 1.00
    实时三角  ——  因果三角。失衡度 = 2 − 1/N（就是经典的那条）
    候选 K    ——  K 很小（几十），切它没有意义

=> MTGR 的 CP 难度由「静态段占比」决定。
   静态段越长，整段越接近方阵，CP 越好做。
   这是 MTGR 比纯因果模型（HSTU 全序列 causal）更「可切」的结构原因。
"""

from __future__ import annotations

import numpy as np

from _core import count_pairs, rng, step, table, title

from hstu_min import HSTUBlock


# ======================================================================
def build_mtgr_mask(s: int, r: int, k: int, candidate_time: np.ndarray | None = None,
                    r_time: np.ndarray | None = None) -> np.ndarray:
    """按论文的三条规则构造 [P+K, P+K] 布尔掩码（True = 可见）。

    s: 静态段长度（U + 历史）
    r: 实时段长度
    k: 候选数
    """
    p = s + r
    n = p + k
    mask = np.zeros((n, n), dtype=bool)

    # 规则 1：静态段内部全可见，且对后面所有 token 可见
    mask[:s, :s] = True
    mask[:, :s] = True

    # 规则 2：实时段内部因果
    mask[s:p, s:p] = np.tril(np.ones((r, r), dtype=bool))

    # 规则 3：候选只看自己（+ 静态 + 自己时间戳之前的实时行为）
    for i in range(k):
        row = p + i
        mask[row, row] = True
        mask[row, s:p] = True
        if candidate_time is not None and r_time is not None:
            # 时间戳细化：更早发生的候选看不到更晚的实时行为
            mask[row, s:p] = r_time < candidate_time[i]
    return mask


def closed_form_pairs(s: int, r: int, k: int) -> int:
    p = s + r
    return s * s + (p * (p + 1) - s * (s + 1)) // 2 + k * (p + 1)


def group_layer_norm(x: np.ndarray, groups: list[int], eps: float = 1e-5) -> np.ndarray:
    """GLN：不同语义域的 token 各自归一化，再拼回去。

    为什么需要它：U（画像，稀疏 ID 多）和 R（实时行为，稠密）分布差很远，
    共享一个 LN 会让 dominant 的那组把另一组压掉。
    并行上的含义：GLN 只需要组内统计量 => 是 token-wise 的，
    与 CP 的序列切分正交（切到哪一段就只用那一段的统计量，不跨卡）。
    """
    out = np.empty_like(x)
    lo = 0
    for g in groups:
        blk = x[lo:lo + g]
        mu = blk.mean(axis=0, keepdims=True)
        var = blk.var(axis=0, keepdims=True)
        out[lo:lo + g] = (blk - mu) / np.sqrt(var + eps)
        lo += g
    return out


# ======================================================================
def demo() -> None:
    title("MTGR：三段掩码如何决定 CP 的难度")

    s, r, k, d = 96, 24, 8, 64
    mask = build_mtgr_mask(s, r, k)
    pairs = count_pairs(mask)

    step("静态 S", val=s, note="U + 历史")
    step("实时 R", val=r)
    step("候选 K", val=k, note="GR 里通常只有几十个")
    step("总长度", val=s + r + k)
    step("可见 pair", val=pairs)
    step("闭式", val=closed_form_pairs(s, r, k), note="必须逐位相等")
    assert pairs == closed_form_pairs(s, r, k), "闭式与暴力掩码不符"

    print("\n  掩码结构（# 可见，. 屏蔽）：")
    _print_mask(mask, (("S", s), ("R", r), ("C", k)))

    # ---------------- 三段各自的 CP 性质 ----------------
    print("\n  [核心] 把整段按 1/N 切开，看三段的失衡度：")
    rows = []
    for n in (2, 4, 8):
        # 静态段：方阵，任何切法都均匀
        static_imb = 1.00
        # 实时段：因果三角，经典结果 2N/(N+1)
        causal_imb = 2 * n / (n + 1)
        # 候选段：不看序列，切了也没意义
        rows.append((n, f"{static_imb:.2f}", f"{causal_imb:.2f}",
                     f"{static_imb:.2f}"))
    print(table(rows, ("N", "静态 S²", "实时 三角", "整体(全静态)")))
    print("     => 静态段占比越高，整段越接近方阵，CP 越好做。")
    print("        这就是 MTGR 比 HSTU（全序列 causal）更『可切』的结构原因。")

    # ---------------- 静态段占比的现实数值 ----------------
    print("\n  [现实] 静态 vs 实时的比例（决定 CP 到底难不难）：")
    rows = []
    for s_, r_ in ((1000, 60), (2000, 100), (4000, 60)):
        m = build_mtgr_mask(s_, r_, 8)
        total = count_pairs(m)
        seq = m.shape[0]
        rows.append((f"S={s_} R={r_}", total, f"{total/seq**2:.3f}",
                     f"{s_*s_/total*100:.0f}%"))
    print(table(rows, ("配置", "pairs", "密度 pairs/n²", "静态占 pairs")))
    print("     => 密度越接近 1.0，越像方阵；静态段吃掉大部分 pair。")

    # ---------------- GLN ----------------
    print("\n  [GLN] 组内归一化，跨组不归一化：")
    x = np.concatenate([rng(3).normal(0, 5, (s, d)),
                        rng(4).normal(0, 1, (r, d)),
                        rng(5).normal(0, 1, (k, d))])
    y = group_layer_norm(x, [s, r, k])
    step("原始 std（三组）",
         val=" / ".join(f"{x[lo:lo+g].std():.2f}" for lo, g in ((0, s), (s, r), (s + r, k))))
    step("GLN 后 std（三组）",
         val=" / ".join(f"{y[lo:lo+g].std():.2f}" for lo, g in ((0, s), (s, r), (s + r, k))))
    print("     => 三组都变成 1，但彼此没有被混在一起。这是 GLN 的全部内容。")

    # ---------------- 喂给 HSTU block ----------------
    print("\n  [串联] 同一个 HSTU block，只换掩码：")
    blk = HSTUBlock(d=d, n_heads=4)
    xin = rng(6).normal(0, 1, (s + r + k, d))
    ts = np.sort(rng(7).uniform(0, 1000, s + r + k))
    out, attn = blk.forward(xin, ts, mask)
    same_out, same_attn = blk.forward(xin, ts, np.tril(np.ones_like(mask)))
    step("MTGR 掩码", pairs=count_pairs(mask),
         norm=f"{np.linalg.norm(attn):.2f}")
    step("纯 causal", pairs=count_pairs(np.tril(np.ones_like(mask))),
         norm=f"{np.linalg.norm(same_attn):.2f}")
    print("     => 掩码只改『哪些 pair 被计入』，不改任何算子。")
    print("        所以 MTGR 的并行性分析 = 一个掩码拓扑问题（见 parallel/cp_min.py）。")


def _print_mask(mask: np.ndarray, groups) -> None:
    edges, lo = [], 0
    for _, g in groups:
        edges.append((lo, lo + g))
        lo += g
    n = mask.shape[0]
    stride = max(1, n // 40)
    for i in range(0, n, stride):
        row = "".join("#" if mask[i, j] else "." for j in range(0, n, stride))
        tag = "S" if i < edges[0][1] else ("R" if i < edges[1][1] else "C")
        print(f"    {tag} {row}")


if __name__ == "__main__":
    demo()
