"""混合并行 —— CP × EP × TP × beam 能不能叠加，以及叠加后还剩多少。

=======================================================================
唯一的组织原则：**每一项成本只有唯一一个轴能切**
=======================================================================
                    ┌───────────────┐
   成本项            │ 谁能让它变小   │
   ─────────────────┼───────────────┤
   prompt KV 读取    │ CP            │  ∝ P，切 P
   prompt attn 计算   │ CP            │  ∝ B·P
   tree 自注意力      │ beam 并行      │  ∝ B²·t，CP 碰不到
   attention 权重     │ TP            │  ∝ d² 的权重
   FFN / 专家权重     │ EP（MoE）或 TP │  ∝ d·d_ff
   全词表打分         │ vocab 轴 / 算法 │ ∝ B·V
   launch / norms     │ 无            │  融掉或忍着

  ⇒ 混合并行不是「把几个加速比乘起来」，
    而是「把每一项分别除以它那个轴的度数，再相加」。

     t_mixed = Σ  f_i / n_i
     speedup = 1 / Σ (f_i / n_i)          f_i 是各项的占比

=======================================================================
两个必须记住的反直觉结论
=======================================================================

(A) **乘积公式是错的，而且会「低估」。**
    单个轴各自的加速比相乘，会重复计算「不受影响的那部分」。
    例：两个各占 50% 的项，各自 2× ——
        真实加速 = 1/(0.25+0.25) = 2.00×
        乘积公式 = (1/0.75)² = 1.78×   ← 低估

(B) **加卡上限由「切不动的残余」决定，不由最强的那一项决定。**
    把 CP/EP/TP 都推到 8，剩下的固定项占比多少，上限就是 1/它。
"""

from __future__ import annotations

import numpy as np

from _core import step, table, title

# ======================================================================
# 锚点：OpenOneRec-1.7B（dense），B=512，L=1024，910B4 单卡
# 按华为 arXiv 2611.23339 的 per-step 算子分解，归并成 6 项。
# 单位：ms/step，总和 ≈ 24.7（与 TPOT 24.0 吻合）
# ======================================================================
ANCHOR = {
    # name          : (ms,     owner)
    "prompt_attn": (3.72, "cp"),      # KV 读取 + prompt 部分的 attention 计算
    "tree_attn": (3.72, "beam"),      # beam 之间树内自注意力（B²·t 项）
    "attn_weights": (6.11, "tp"),     # QKV + o 投影
    "ffn": (9.26, "ep"),              # FFN（MoE 版就是专家）
    "scoring": (0.50, "vocab"),       # B×V 的 logits + top-k
    "fixed": (1.38, "none"),          # launch、norms、beam-path 更新
}

TOTAL = sum(v[0] for v in ANCHOR.values())


def fracs() -> dict[str, float]:
    return {k: v[0] / TOTAL for k, v in ANCHOR.items()}


# ======================================================================
def speedup(degrees: dict[str, int], note: str = "") -> tuple[float, dict]:
    """按「每一项除以它的轴」算混合加速，并给出残余占比。"""
    f = fracs()
    owned = {k: v[1] for k, v in ANCHOR.items()}
    acc = 0.0
    detail = {}
    for k, fk in f.items():
        owner = owned[k]
        n = degrees.get(owner, 1)
        acc += fk / n
        detail[k] = (fk, owner, n, fk / n)
    return 1.0 / acc, detail


def comm_ms(degrees: dict[str, int], B: int, d: int, n_layers: int,
            top_k: int, P: int) -> dict[str, float]:
    """每一步各轴的通信时间（ms）。全部按单机 HCCS 200 GB/s。"""
    BW = 200e9
    out = {}
    c = degrees.get("cp", 1)
    if c > 1:
        # 每层一次 LSE 合并，搬 B·d；边界两次
        out["CP"] = 2 * n_layers * B * d * 4 / BW * 1e3
    e = degrees.get("ep", 1)
    if e > 1:
        # dispatch + combine，各搬 B·k·d（bf16）
        out["EP"] = 2 * n_layers * B * top_k * d * 2 / BW * 1e3
    t = degrees.get("tp", 1)
    if t > 1:
        # 每层两次 all-reduce（attn 输出 + FFN 输出）
        out["TP"] = 2 * 2 * n_layers * B * d * 4 / BW * 1e3
    g = degrees.get("beam", 1)
    if g > 1:
        # 全局 top-B：B·K·4 字节，3 步
        out["beam"] = 3 * B * 64 * 4 / BW * 1e3
    return out


# ======================================================================
def demo() -> None:
    title("混合并行：每一项只有一个轴能动")

    f = fracs()
    rows = [(k, v[1], f"{v[0]:.2f}", f"{f[k]*100:.1f}%") for k, v in ANCHOR.items()]
    print("\n  [锚点] OpenOneRec-1.7B(dense), B=512, L=1024, 单卡 910B4")
    print(table(rows, ("成本项", "能切它的轴", "ms/step", "占比")))
    step("合计", val=f"{TOTAL:.2f} ms", note="与 TPOT 24.0 吻合")

    # ---------------- 单轴 ----------------
    print("\n  [单轴] 各轴单独开到 4：")
    rows = []
    for ax in ("cp", "tp", "ep", "beam", "vocab"):
        sp, _ = speedup({ax: 4})
        share = f[[k for k, v in ANCHOR.items() if v[1] == ax][0]]
        rows.append((ax.upper(), f"{share*100:.1f}%", f"{sp:.2f}x",
                     f"{1/(1-share):.2f}x"))
    print(table(rows, ("轴", "负责的占比", "开到 4 时", "无限卡上限")))
    print("     => 每个轴的加速比上限是 1/(1-占比)，与度数无关。")
    print("        占比 15% 的轴推到 4 只有 1.13×；推到 ∞ 也只有 1.18×。")

    # ---------------- 混合：不是乘积 ----------------
    print("\n  [关键] 混合不是乘积：")
    deg_all = {"cp": 4, "tp": 4, "ep": 4, "beam": 4}
    sp_all, detail = speedup(deg_all)
    single = {}
    for ax in ("cp", "tp", "ep", "beam"):
        single[ax], _ = speedup({ax: 4})
    prod = float(np.prod(list(single.values())))
    rows = [(ax.upper(), f"{single[ax]:.3f}x") for ax in ("cp", "tp", "ep", "beam")]
    print(table(rows, ("轴", "单独开到 4")))
    step("乘积公式（错）", val=f"{prod:.3f}x")
    step("逐项相除（对）", val=f"{sp_all:.3f}x")
    print(f"     => 乘积低估了 {(sp_all/prod - 1)*100:.0f}%。原因：")
    print("        单轴加速比里已经含了「不受影响的那部分」，乘起来就把它们算了多次。")
    print("        正确的算法是 f_i / n_i 逐项相加。")

    # ---------------- 通信侵蚀 ----------------
    print("\n  [代价] 加上通信后，理想值会被侵蚀多少：")
    B, d, n_layers, top_k, P = 512, 2048, 28, 8, 1024
    rows = []
    for name, deg in (
        ("CP=4", {"cp": 4}),
        ("TP=4", {"tp": 4}),
        ("EP=4", {"ep": 4}),
        ("CP+EP", {"cp": 4, "ep": 4}),
        ("CP+EP+TP", {"cp": 4, "ep": 4, "tp": 4}),
        ("全 8", {"cp": 8, "ep": 8, "tp": 8, "beam": 8}),
    ):
        sp, _ = speedup(deg)
        ideal = TOTAL / sp
        comm = comm_ms(deg, B, d, n_layers, top_k, P)
        c_total = sum(comm.values())
        real = ideal + c_total
        rows.append((name, f"{sp:.2f}x", f"{ideal:.1f} ms",
                     f"{c_total:.2f} ms",
                     f"{comm.get('CP',0):.2f}/{comm.get('EP',0):.2f}/"
                     f"{comm.get('TP',0):.2f}/{comm.get('beam',0):.3f}",
                     f"{TOTAL/real:.2f}x"))
    print(table(rows, ("配置", "理想加速", "理想时间", "通信", "CP/EP/TP/beam",
                       "实际加速")))
    print("     => 通信里 **EP 和 TP 最大**（都 ∝ B·d），CP 很小（只搬 B·d）。")
    print("        这正是「EP 贵 k 倍」那条结论在混合场景下的体现。")

    # ---------------- 耦合 1：CP × beam ----------------
    print("\n  [耦合 1] CP 只能切 prompt，切不到 beam 树：")
    print("            CP 的有效份额 = P / (P + B·t)      （t = 已 decode 步数）")
    rows = []
    for P_ in (1024, 4096, 16384, 65536):
        for Bv in (128, 512):
            t_ = 2
            share = P_ / (P_ + Bv * t_)
            rows.append((P_, Bv, f"{Bv*t_}", f"{share*100:.0f}%",
                         f"{share*15.1/4:.1f}%"))
    print(table(rows, ("P", "B", "B·t", "可切份额", "CP=4 后占全步长")))
    print("     => **beam 越大，CP 越没用。** B=512, t=2, P=1024 时只剩一半可切。")
    print("        P 到 32K 以上才恢复（>94%）。这与「长 prompt + 宽 beam」的现实吻合。")

    # ---------------- 耦合 2：EP × beam ----------------
    print("\n  [耦合 2] beam 的假设互相相似 => 路由聚拢 => EP 打折：")
    print("            EP 有效度数 = 1 / (ρ + (1-ρ)/G)     ρ = 跨 shard 路由相关性")
    rows = []
    for rho in (0.0, 0.2, 0.5, 0.7, 1.0):
        r = [f"{rho:.1f}"]
        for G in (2, 4, 8):
            eff = 1 / (rho + (1 - rho) / G)
            r.append(f"{eff:.2f}x")
        rows.append(tuple(r))
    print(table(rows, ("ρ", "G=2", "G=4", "G=8")))
    print("     => ρ=1（所有 beam 完全相关）时，EP 切多少都是 1.0×。")
    print("        LLM 的 DP/EP 对应 ρ≈0，所以能到 G；GR 的 beam 是 ρ 的高危区。")

    # ---------------- 耦合 3：串行 vs 重叠 ----------------
    print("\n  [耦合 3] 三个轴的 collective 能不能重叠：")
    rows = [
        ("同一个 rank 组", "CP / EP / TP 的 collective 串行", "Σ 三者", "最简单，最慢"),
        ("分阶段（Helix）", "attention 用 CP、FFN 用 EP，同一批卡换角色", "max(CP, EP)", "要重配置，3 步摊销不掉"),
        ("分 rank 组（MegaScale）", "attention 节点与 expert 节点分离", "max + 一次搬运", "要 P2P 传激活"),
    ]
    print(table(rows, ("做法", "重叠方式", "通信代价", "代价")))
    print("     => GR 只有 3 步 decode，Moebius 那种「运行时切换」要 215~434 ms，")
    print("        比整个 TPOT 还大一个量级 ⇒ **只能选固定布局**。")

    # ---------------- 残余 ----------------
    print("\n  [残余] 把所有轴都推到 8，还剩什么：")
    sp, detail = speedup({"cp": 8, "ep": 8, "tp": 8, "beam": 8, "vocab": 8})
    rows = []
    for k, (fk, owner, n, after) in sorted(detail.items(), key=lambda x: -x[1][3]):
        rows.append((k, owner, f"{fk*100:.1f}%", n, f"{after*100:.1f}%"))
    print(table(rows, ("成本项", "轴", "原占比", "度数", "切完后占比")))
    step("上限", val=f"{sp:.2f}x", note="不含通信")
    biggest = max((x for x in detail.items() if x[1][1] != "none"),
                  key=lambda x: x[1][3])
    print(f"     => 残余最大的是 **{biggest[0]}**（{biggest[1][3]*100:.1f}%），"
          f"它的轴是 {biggest[1][1].upper()}。")
    print("        fixed（launch/norms）只有 5.6%，但**它没有轴** —— 只有 G→∞ 时才成为瓶颈。")
    print("        换句话说：G=8 时的天花板由「大占比项的 1/G 残余」决定，")
    print("        而不是由固定项决定。想再往上，只能去砍 fixed 与 scoring，")
    print("        那是「图编译 + 融合」和「SID 受限解码」的战场，不是并行的战场。")

    # ---------------- 三条可直接用的判据 ----------------
    print("\n  [判据] 三条可以照着算的：")
    rows = [
        ("值不值得开 CP", "P/(P+B·t) × f_prompt > f_fixed", "可切份额要大于残余"),
        ("EP 值不值得", "ρ < 1 - 1/G + 1/G·(需要的加速)", "先测 ρ"),
        ("混合能不能过 2×", "Σ_{无主} f_i < 0.5", "固定项 + 打分必须 < 50%"),
    ]
    print(table(rows, ("问题", "条件", "含义")))


if __name__ == "__main__":
    demo()
