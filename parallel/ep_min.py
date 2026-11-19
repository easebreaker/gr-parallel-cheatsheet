"""EP 最小实现 —— 把一个 MoE 层摊到 N 张卡上。

对应代码: vLLM `--enable-expert-parallel`、Megatron `moe/`、
          MegaScale-Infer（attention / FFN 分离 + EP）、
          RankMixer 的 Sparse-MoE、OneRec 的 MoE

=======================================================================
一个 MoE + EP 层就四步
=======================================================================

  1. router     每个 token 对 E 个专家打分，取 top-k
                => 产生 b_eff = (token 数) × k 个 (token, expert) 对

  2. dispatch   all-to-all：把 token 发到「持有目标专家的那张卡」
                体积 = b_eff × d × bytes

  3. grouped GEMM  每个专家算自己收到的那批行

  4. combine    all-to-all：把结果还回去
                体积 = b_eff × d × bytes

=======================================================================
三个必须记住的量（EP 全部分析的起点）
=======================================================================

  (A) 通信量        V_EP = 2 · B · k · d · bytes
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    与专家数 E **无关**。加专家只改分片粒度，不改通信量。

  (B) 与 TP 的比值  V_EP / V_TP = k     （TP 一次 all-reduce = 2·B·d）
                    k=8 时 EP 的通信量是 TP 的 8 倍。这是 EP 的固定成本。

  (C) 每专家行数     rows_per_expert = B·k / E
                    LLM 低并发 decode: 1×8/256 = 0.03 行 -> GEMM 效率极差
                    GR    beam  decode: 512×8/64 = 64 行  -> 效率正常
                    **这就是「beam 是 EP 的天然饲料」。**

=======================================================================
唯一的结构性风险：路由相关性
=======================================================================
beam 的 B 条假设共享前缀，输入几乎相同 => 它们**路由到同一批专家**。
负载失衡因此从 1.0x 放大到几倍，而 all-to-all 的完成时间取决于
最忙的那个目的地（max column sum），不是平均值。

这是 GR 的 EP 与 LLM 的 EP 最关键的区别：
LLM 的 batch 是「不同请求」，天然分散；
GR 的 batch 是「同一请求的不同假设」，天然聚拢。
"""

from __future__ import annotations

import numpy as np

from _core import rng, step, table, title


# ======================================================================
class MoELayer:
    def __init__(self, d: int = 64, n_experts: int = 8, top_k: int = 2, n_ranks: int = 4):
        self.d, self.E, self.k, self.N = d, n_experts, top_k, n_ranks
        assert n_experts % n_ranks == 0
        self.experts_per_rank = n_experts // n_ranks
        g = rng(0)
        s = 1.0 / np.sqrt(d)
        self.Wg = g.normal(0, s, (d, n_experts))
        self.W1 = [g.normal(0, s, (d, 4 * d)) for _ in range(n_experts)]
        self.W2 = [g.normal(0, s, (4 * d, d)) for _ in range(n_experts)]

    # ------------------------------------------------------------------
    def route(self, x: np.ndarray):
        """返回 (top-k 专家下标, 门控权重)，形状都是 [n_tokens, k]。"""
        logits = x @ self.Wg
        idx = np.argsort(-logits, axis=1)[:, : self.k]
        w = np.take_along_axis(logits, idx, axis=1)
        return idx, np.maximum(w, 0.0)              # ReLU 门

    def owner(self, expert: int) -> int:
        return expert // self.experts_per_rank

    # ------------------------------------------------------------------
    def forward_ep(self, x: np.ndarray):
        """完整四步。x 是【本 rank】那批 token（例如一个请求的全部 beam）。"""
        n = x.shape[0]
        idx, w = self.route(x)                       # [n, k]

        # ---- 2. dispatch：按目的地 rank 分桶 ----
        inbox = {r: [] for r in range(self.N)}
        for t in range(n):
            for j in range(self.k):
                e = int(idx[t, j])
                inbox[self.owner(e)].append((t, e, x[t] * w[t, j]))
        dispatched = sum(len(v) for v in inbox.values())

        # ---- 3+4. 每个 rank 只算自己持有的专家，结果汇总回 token ----
        acc = np.zeros_like(x)
        for r, items in inbox.items():
            for t, e, val in items:
                assert self.owner(e) == r, "dispatch 把 token 发错卡了"
                acc[t] += np.maximum(val @ self.W1[e], 0.0) @ self.W2[e]

        return acc, dispatched, inbox

    # ------------------------------------------------------------------
    def stats(self, x: np.ndarray):
        idx, _ = self.route(x)
        load = np.zeros(self.E, dtype=int)
        for e in idx.reshape(-1):
            load[e] += 1
        per_rank = np.array([
            load[r * self.experts_per_rank:(r + 1) * self.experts_per_rank].sum()
            for r in range(self.N)
        ])
        return load, per_rank


# ======================================================================
def demo() -> None:
    title("EP：四步、三个量、一个风险")

    d, E, k, N = 64, 16, 2, 4
    layer = MoELayer(d=d, n_experts=E, top_k=k, n_ranks=N)

    # ---------------- 四步 ----------------
    B = 512
    x = rng(1).normal(0, 1, (B, d))
    step("输入 token", val=B, note="正好等于一个请求的 beam 宽度")
    idx, w = layer.route(x)
    step("router 输出", shape=idx.shape, note=f"top-{k} of {E}")
    step("b_eff = B·k", val=B * k, note="dispatch 的对象数")
    acc, dispatched, inbox = layer.forward_ep(x)
    step("dispatch 的对数", val=dispatched)
    step("每个 rank 收到的行数", val=[len(v) for v in inbox.values()])
    step("输出", shape=acc.shape)

    # ---------------- 量 A ----------------
    print("\n  [量 A] 通信量与专家数无关：")
    rows = []
    for EE in (8, 16, 64, 256):
        v = 2 * B * k * d * 4
        rows.append((EE, f"{v / 1024:.0f} KiB", f"{EE // N} 专家/卡",
                     f"{B * k / EE:.1f}"))
    print(table(rows, ("专家数 E", "all-to-all 字节", "分片粒度", "每专家行数")))
    print("     => 第 2 列完全不变。加专家只改第 3、4 列。")

    # ---------------- 量 B ----------------
    print("\n  [量 B] EP 的通信量是 TP 的 k 倍：")
    v_ep = 2 * B * k * d * 4
    v_tp = 2 * B * d * 4
    step("V_EP", val=f"{v_ep / 1024:.0f} KiB")
    step("V_TP", val=f"{v_tp / 1024:.0f} KiB")
    step("比值", val=f"{v_ep / v_tp:.0f}x", note="等于 k，与 B 无关")

    # ---------------- 量 C ----------------
    print("\n  [量 C] 每专家行数 = B·k/E（GEMM 效率的真正开关）：")
    rows = []
    for label, BB, EE in (("LLM decode B=1", 1, 256),
                          ("LLM batch 32", 32, 256),
                          ("GR beam 128", 128, 64),
                          ("GR beam 512", 512, 64)):
        rows.append((label, BB, EE, f"{BB * k / EE:.2f}"))
    print(table(rows, ("场景", "B", "E", "行/专家")))
    print("     => 低并发 LLM 的 MoE 是「每个专家算不到 0.1 行」，纯浪费。")
    print("        GR 即使并发=1，beam 也提供了 512 的 batch。")
    print("        这是 beam 对 EP 唯一真正的正面作用。")

    # ---------------- 风险 ----------------
    print("\n  [风险] beam 的假设互相相似 => 路由聚拢 => 失衡放大：")
    base = rng(2).normal(0, 1, d)
    rows = []
    for label, noise in (("beam（同前缀）", 0.02), ("beam（已分叉）", 0.3),
                         ("iid 独立请求", 1.0)):
        xx = base[None] + noise * rng(3).normal(0, 1, (B, d))
        load, per_rank = layer.stats(xx)
        active = int((load > 0).sum())
        rows.append((label, f"{noise}", f"{active}/{E}",
                     f"{load.max() / (load.mean() + 1e-9):.1f}x",
                     f"{per_rank.max() / (per_rank.mean() + 1e-9):.2f}x"))
    print(table(rows, ("输入", "噪声", "激活专家", "专家失衡", "rank 失衡")))
    print("     => 同样的 B、同样的模型，只因为输入相似度不同，")
    print("        专家负载可以差一个量级。而 all-to-all 等的是最忙的那个。")
    print("        这是 GR 的 EP 与 LLM 的 EP 最关键的区别。")

    # ---------------- 硬约束 ----------------
    print("\n  [硬约束] all-to-all 的绝对耗时（12 层 MoE，bf16，d=2048）：")
    rows = []
    for Bv in (128, 512, 900):
        dd = 2048
        v = 2 * Bv * k * dd * 2 * 12 / 1e6      # MB
        rows.append((Bv, f"{v:.0f} MB", f"{v / 200:.1f} ms", f"{v / 25:.1f} ms"))
    print(table(rows, ("B", "每步总量", "卡内 HCCS 200GB/s", "跨机 200Gbps")))
    print("     => 跨机是 4~7 ms 的硬增量，直接压在 20~40 ms 的 TPOT 上（+10~30%）。")
    print("        卡内 HCCS 只有 0.5 ms。差一个量级。")
    print("        => GR 的 EP 应留在单机域内（Helix / MegaScale 同结论）。")

    # ---------------- 定位 ----------------
    print("\n  [定位] 三个轴各切什么：")
    rows = [
        ("TP", "权重（每层都切）", "2·B·d 每层 all-reduce", "∝ B，B 大时便宜"),
        ("EP", "专家（只切 MoE 层）", "2·B·k·d 两次 all-to-all", "∝ B，是 TP 的 k 倍"),
        ("CP", "序列 / KV", "2·(N−1)·L·d (ring)", "与 B 无关"),
    ]
    print(table(rows, ("轴", "切什么", "通信量", "随 B 的走势")))


if __name__ == "__main__":
    demo()
