"""LONGER 最小实现 —— 「扁矩阵 + softmax + token merge」。

论文: "LONGER: Scaling Up Long Sequence Modeling in Industrial Recommenders"
      arXiv 2505.04421（RecSys'25，字节）

=======================================================================
它和 HSTU 的两个致命区别
=======================================================================

(1) **它用 softmax**（原文公式：Softmax(QKᵀ/√d + M)V）。
    ⇒ w 依赖键集合 ⇒ 分块不能相加 ⇒ 需要 online softmax / LSE 合并。
    「三类 GR 都是纯加归约」这句话在这里直接不成立。

(2) **它是扁矩阵**：query 只有 100~250 个，KV 是全长。
    ⇒ q : L ≈ 1 : 40 ~ 1 : 100
    ⇒ 能省的绝对工作量是 q·L，本来就比方矩阵的 L² 小两个量级。

=======================================================================
一个 LONGER 前向的三段注意力，形状完全不同
=======================================================================

    用户历史 L 个 item
         │
    ┌────┴─────────────────────────────────────────┐
    │ token merge：相邻 K 个合并成 1 个            │
    │   L -> R = L/K                               │
    └────┬─────────────────────────────────────────┘
         │
    ① inner（组内）：每 K 个原始 token 之间做一层轻量 transformer
         代价 ∝ R · K² · d = L · K · d      ← 线性于 L
         只在组内，天然不跨卡 ⇒ 对 CP 完全中立

    ② merged self（合并后）：R 个 token 之间做因果自注意力
         代价 ∝ R² · d = (L/K)² · d         ← 二次，但 R 已经很小

    ③ hybrid cross：q = 全局 token + 采样 token（100~250） 读合并后的 R
         代价 ∝ q · R · d                   ← 扁矩阵，带宽主导

=======================================================================
Amdahl 上限为什么只有 1.2~1.3×
=======================================================================
    可切的部分 = q · R · d（cross）+ R² · d（merged self）
    而 R = L/K 在 K=4、L=10000 时只有 2500，
    KV 字节 = 2500 · 32 · 2 · 2 = 320 KB/层 —— 单卡绰绰有余。
    ⇒ 「容量判据」又一次说：不需要切。
"""

from __future__ import annotations

import numpy as np

from _core import count_pairs, rng, step, table, title


# ======================================================================
class TokenMerge:
    """相邻 K 个 token → 1 个。论文里是线性投影 + 池化。"""

    def __init__(self, d: int = 32, K: int = 4):
        self.d, self.K = d, K
        self.W = rng(0).normal(0, 1 / np.sqrt(d), (d, d))

    def forward(self, x: np.ndarray) -> np.ndarray:
        n, d = x.shape
        assert n % self.K == 0
        g = x.reshape(n // self.K, self.K, d).mean(axis=1)   # 池化
        return g @ self.W


class InnerTrans:
    """组内一层 transformer：只让同一组内的 K 个 token 互相看。

    这是 LONGER「把 L² 降到 L·K」的关键：每个 token 只看组内 K-1 个邻居。
    """

    def __init__(self, d: int = 32, K: int = 4):
        self.d, self.K = d, K
        s = 1.0 / np.sqrt(d)
        g = rng(1)
        self.Wq, self.Wk, self.Wv = (g.normal(0, s, (d, d)) for _ in range(3))

    def forward(self, x: np.ndarray) -> np.ndarray:
        n, d = x.shape
        R, K = n // self.K, self.K
        xg = x.reshape(R, K, d)
        q = xg @ self.Wq
        k = xg @ self.Wk
        v = xg @ self.Wv
        s = (q @ k.transpose(0, 2, 1)) / np.sqrt(d)          # [R, K, K]
        s = np.where(np.tril(np.ones((K, K), dtype=bool))[None], s, -1e9)
        a = _softmax(s)
        return (a @ v).reshape(n, d)

    def pairs(self, n: int) -> int:
        R, K = n // self.K, self.K
        return R * K * (K + 1) // 2


class MergedSelfAttn:
    """合并后序列上的因果自注意力。R² 的那一项。"""

    def __init__(self, d: int = 32):
        self.d = d
        s = 1.0 / np.sqrt(d)
        g = rng(2)
        self.Wq, self.Wk, self.Wv = (g.normal(0, s, (d, d)) for _ in range(3))

    def forward(self, x: np.ndarray, causal: bool = True) -> np.ndarray:
        q, k, v = x @ self.Wq, x @ self.Wk, x @ self.Wv
        s = (q @ k.T) / np.sqrt(self.d)
        if causal:
            s = np.where(np.tril(np.ones_like(s, dtype=bool)), s, -1e9)
        return _softmax(s) @ v


class HybridCross:
    """全局 token + 采样 token（q 个）对合并后序列（R 个）做交叉注意力。

    **这里用的是 softmax** —— 与 HSTU 的分水岭。
    """

    def __init__(self, d: int = 32, n_query: int = 100):
        self.d, self.nq = d, n_query
        s = 1.0 / np.sqrt(d)
        g = rng(3)
        self.Wq, self.Wk, self.Wv = (g.normal(0, s, (d, d)) for _ in range(3))
        self.q_init = g.normal(0, 1, (n_query, d))

    def forward(self, merged: np.ndarray) -> np.ndarray:
        q = self.q_init @ self.Wq
        k, v = merged @ self.Wk, merged @ self.Wv
        return _softmax((q @ k.T) / np.sqrt(self.d)) @ v


def _softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


# ======================================================================
def demo() -> None:
    title("LONGER：扁矩阵 + softmax + token merge")

    d, K = 32, 4
    L, nq = 4096, 100
    x = rng(4).normal(0, 1, (L, d))

    merge = TokenMerge(d, K)
    inner = InnerTrans(d, K)
    msa = MergedSelfAttn(d)
    cross = HybridCross(d, nq)

    merged = merge.forward(x)
    step("原始历史", L=L, d=d)
    step("token merge", K=K, R=L // K, note="合并后长度")
    step("inner（组内）", pairs=inner.pairs(L), note="R·K(K+1)/2，线性于 L")
    step("merged self", pairs=(L // K) * (L // K + 1) // 2, note="R²/2，二次但 R 小")
    step("hybrid cross", pairs=nq * (L // K), note="q·R，扁矩阵")

    # 真实前向：inner 在原始 L 上做（组内），然后 merge，再 cross
    x_inner = inner.forward(x) + x
    merged = merge.forward(x_inner)
    y = cross.forward(merged)
    step("inner 输出", shape=x_inner.shape)
    step("merge 输出", shape=merged.shape)
    step("cross 输出", shape=y.shape, note=f"q={nq} 行，只剩 q 个 token")

    # ---------------- 与全量 self-attn 的对比 ----------------
    print("\n  [对比] 同样 L 个 token，全量自注意力要多少 pair：")
    rows = []
    for LL in (1024, 2048, 4096, 8192, 16384):
        full = LL * (LL + 1) // 2
        merged_pairs = (LL // K) * (LL // K + 1) // 2
        inner_p = (LL // K) * K * (K + 1) // 2
        cross_p = nq * (LL // K)
        longer = merged_pairs + inner_p + cross_p
        rows.append((LL, f"{full/1e6:.2f} M", f"{merged_pairs/1e6:.2f} M",
                     f"{inner_p/1e6:.2f} M", f"{cross_p/1e6:.2f} M",
                     f"{longer/1e6:.2f} M", f"{full/longer:.1f}x"))
    print(table(rows, ("L", "全量 self", "merged self", "inner", "cross",
                       "LONGER 合计", "省了")))
    print("     => LONGER 不是「把 L² 变小」，而是「把 L² 换成 (L/K)² + L·K」。")
    print(f"        K={K} 时换算比约 {K}² = {K*K}x，这就是它敢上 10000 的原因。")

    # ---------------- 关键：q : L ----------------
    print("\n  [关键] 扁矩阵的形状（决定 CP 有没有对象）：")
    rows = []
    for LL, qq in ((1000, 100), (4000, 100), (10000, 100), (10000, 250)):
        rows.append((LL, qq, f"{qq/LL:.4f}",
                     f"{qq*(LL//K)/1e3:.0f} K",
                     f"{(LL//K)*(LL//K+1)//2/1e6:.2f} M"))
    print(table(rows, ("L", "q", "q/L", "cross pair", "merged self pair")))
    print("     => q/L 在 1e-2 量级。切 Q 之后每卡只剩几十个 query，")
    print("        而 merged KV 仍要全量 ⇒ 省不下大头，还多付固定开销。")
    print("        这正是 Amdahl 上限只有 1.2~1.3× 的来源。")

    # ---------------- softmax vs 纯加 ----------------
    print("\n  [分水岭] LONGER 用 softmax，所以 CP 需要 LSE 合并：")
    q = rng(5).normal(0, 1, (2, d))
    kv = rng(6).normal(0, 1, (64, d))
    a_full = _softmax((q @ kv.T) / np.sqrt(d))
    a_half = _softmax((q @ kv[:32].T) / np.sqrt(d))
    step("softmax 行和（全量）", val=f"{a_full.sum(-1)[0]:.6f}")
    step("softmax 行和（半量）", val=f"{a_half.sum(-1)[0]:.6f}")
    step("两段直接相加", val=f"{a_half.sum(-1)[0]*2:.6f}",
         note="错：必须按 LSE rescale")
    print("     => 对比 HSTU：SiLU 下两段 partial 直接相加就是精确解。")
    print("        所以 LONGER 是唯一「真的需要复杂 CP」的 GR 模型 ——")
    print("        而它恰好也是唯一真的需要长序列的模型。这就是那个悖论。")

    # ---------------- KV 字节：容量判据 ----------------
    print("\n  [容量判据] 合并后的 KV 有多大（决定要不要切）：")
    rows = []
    for LL in (4096, 10000, 32768, 100000, 1000000):
        R = LL // K
        kv_bytes = R * d * 2 * 2 * 32          # R · d · K/V · bf16 · 32 层
        rows.append((f"{LL/1000:.0f}K" if LL >= 1000 else LL, R,
                     f"{kv_bytes/1024**2:.1f} MiB",
                     "装得下" if kv_bytes < 8e9 else "装不下"))
    print(table(rows, ("L", "R = L/K", "合并 KV（32 层）", "8GB 预算")))
    print("     => 注意 d=32，所以 KV ≈ L × 1 KB。**要到 L≈800 万才装不下。**")
    print("        而 LONGER 的部署长度是 2K~10K ⇒ 容量判据永远说「不需要切」。")

    # ---------------- KV cache serving ----------------
    print("\n  [KV cache serving] 论文的工程优化（吞吐降级 40% -> 6.8%）：")
    rows = [
        ("每个候选都重算用户序列", "候选数 × L", "最慢"),
        ("缓存用户序列的 K/V", "候选数 × q_cand", "快"),
        ("解耦用户侧与候选侧", "L 一次 + q_cand 多次", "论文方案"),
    ]
    print(table(rows, ("做法", "计算量", "备注")))
    print("     => 这本质上就是「把 self-attn 改写成 cross-attn」，")
    print("        与 HyFormer 的 Query Decoding 是同一个技巧（见 09）。")


if __name__ == "__main__":
    demo()
