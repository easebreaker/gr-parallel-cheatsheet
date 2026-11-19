"""HSTU 最小实现 —— 只保留决定「并行性」的那一半。

论文: Zhai et al., "Actions Speak Louder than Words: Trillion-Parameter
      Sequential Transducers for Generative Recommendations" (ICML'24)

=======================================================================
一个 HSTU block 里有四个事实，前两个决定了「能不能 CP、要不要 CP」
=======================================================================

(1) 没有 softmax。
    HSTU 用 SiLU 代替 softmax：  A = SiLU(score)
    于是输出是   out_i = Σ_j w_ij · V_j        —— 一个「纯加」归约。

    后果极强：w_ij 不依赖键集合 J(i)。
    => 把 KV 切开，各卡算局部 Σ，跨卡直接相加就是精确结果。
    => 不需要 online softmax、不需要 LSE 配平、不需要专门 kernel。

(2) 分数除以 max_seq_len（不是除以 Σ）。
    论文里的 "average factor"。它必须在 SiLU 之前。
    放到后面会让输出尺度随 L 漂移，几百步后 NaN。

(3) rab（relative attention bias）由时间戳分桶查表得到：
    参数只有 O(N)，但前向张量是 O(N²)，必须实体化成 [B,N,N]。
    Meta 的生产路径（modules/stu.py）里没有 rab，只有窗口。

(4) 门控：out = (A @ V) ⊙ SiLU(U)。U 是第四个投影，不是 V。

=======================================================================
复杂度（一个 block，单样本）
=======================================================================
    pairs  = N²                （无 mask）
    FLOPs  = 2·N²·d·h
    rab    = N²·4 字节

这确实是真的二次 —— 但见 mtgr_min.py 的结论：
GR 的 KV 在 N ≤ 32K 时单卡装得下 ⇒ 「不需要切」。
"""

from __future__ import annotations

import numpy as np

from _core import count_pairs, flops_attn, rng, silu, softmax, step, title


class HSTUBlock:
    def __init__(self, d: int = 64, n_heads: int = 4, n_buckets: int = 8):
        self.d = d
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.n_buckets = n_buckets
        g = rng(0)
        s = 1.0 / np.sqrt(d)
        # 四组投影：Q / K / V / U。U 是门控，不是 value。
        self.Wq, self.Wk, self.Wv, self.Wu = (g.normal(0, s, (d, d)) for _ in range(4))
        self.Wo = g.normal(0, s, (d, d))
        # rab：时间差 → 桶 → 一个标量。参数 O(n_buckets)，前向张量 O(N²)。
        self.rab_table = g.normal(0, 0.1, (n_buckets,))

    # ------------------------------------------------------------------
    def rel_bias(self, tq: np.ndarray, tk: np.ndarray) -> np.ndarray:
        """[Nq] × [Nk] 时间戳 -> [Nq,Nk] 偏置。这一步就是 O(N²) 的来源。"""
        delta = np.abs(tq[:, None] - tk[None, :]).astype(np.float64)
        bucket = np.minimum((delta / 60.0).astype(np.int64), self.n_buckets - 1)
        return self.rab_table[bucket]

    def _heads(self, x: np.ndarray) -> np.ndarray:
        return x.reshape(x.shape[0], self.n_heads, self.head_dim).transpose(1, 0, 2)

    def weights(self, xq, xk, tq, tk, mask) -> np.ndarray:
        """只算权重矩阵 A = SiLU(scale·QK^T + rab)，不乘 V。

        单独拆出来，是为了下面演示「分块求和」时能看到 A 是可加的。
        """
        qh = self._heads(xq @ self.Wq)                      # [h, Nq, hd]
        kh = self._heads(xk @ self.Wk)                      # [h, Nk, hd]
        scores = qh @ kh.transpose(0, 2, 1)                 # [h, Nq, Nk]
        # (2) average factor：除以 max_seq_len，且在 SiLU 之前
        scores = scores / max(xq.shape[0], xk.shape[0])
        # (3) rab 实体化
        scores = scores + self.rel_bias(tq, tk)[None]
        # (1) 关键一行：SiLU 而不是 softmax —— 没有按列归一化
        return np.where(mask[None], silu(scores), 0.0)

    # ------------------------------------------------------------------
    def forward(self, x, timestamps, mask):
        n, d = x.shape
        attn = self.weights(x, x, timestamps, timestamps, mask)   # [h, N, N]
        vh = self._heads(x @ self.Wv)
        out = attn @ vh                                          # 纯加归约
        out = out.transpose(1, 0, 2).reshape(n, d)
        out = out * silu(x @ self.Wu)                            # (4) 门控
        return out @ self.Wo, attn


def make_causal(n: int) -> np.ndarray:
    return np.tril(np.ones((n, n), dtype=bool))


# ======================================================================
def demo() -> None:
    title("HSTU：为什么它是「纯加归约」，以及这对 CP 意味着什么")

    n, d, h = 64, 64, 4
    block = HSTUBlock(d=d, n_heads=h)
    x = rng(1).normal(0, 1, (n, d))
    ts = np.sort(rng(2).uniform(0, 600, n))
    mask = make_causal(n)

    out, attn = block.forward(x, ts, mask)

    step("输入", N=n, d=d, heads=h)
    step("q/k/v/u", shape=f"({n},{d}) x4", note="U 是门控")
    step("A = SiLU(...)", shape=tuple(attn.shape), note="O(N^2) 张量")
    step("rab", shape=(n, n), note="O(N^2) 张量，参数只有 8 个")
    step("可见 pair", pairs=count_pairs(mask), total=n * n)
    step("FLOPs", val=f"{flops_attn(count_pairs(mask), d // h, h) / 1e6:.2f} M")
    step("rab 字节", val=f"{n * n * 4 / 1024:.1f} KiB", note="与参数无关")

    # ---------------- 验证 1：纯加归约 ⇒ 分块求和精确 ----------------
    print("\n  [验证 1] 把 KV 切成两段，各段独立算 partial，再相加：")
    half = n // 2
    vh = block._heads(x @ block.Wv)
    a_all = block.weights(x, x, ts, ts, mask)
    partial = []
    for lo, hi in ((0, half), (half, n)):
        a = block.weights(x, x[lo:hi], ts, ts[lo:hi], mask[:, lo:hi])
        part = a @ vh[:, lo:hi, :]
        partial.append(part)
        step(f"rank {len(partial)-1} partial", pairs=count_pairs(mask[:, lo:hi]),
             bytes=f"{part.nbytes/1024:.1f} KiB")
    merged = partial[0] + partial[1]
    merged = merged.transpose(1, 0, 2).reshape(n, d) * silu(x @ block.Wu)
    merged = merged @ block.Wo
    err = np.abs(merged - out).max()
    step("整体 vs 分块", max_abs_err=f"{err:.3e}", verdict="精确相等" if err < 1e-6 else "不等")
    print("     => 跨卡只需要一次加法，不需要 LSE、不需要 rescale。这就是 CP 在")
    print("        HSTU/MTGR 上『零归约』的全部技术内容。")

    # ---------------- 验证 2：softmax 为什么不行 ----------------
    print("\n  [验证 2] 同一份 score 换成 softmax，分块就不成立了：")
    s = block._heads(x @ block.Wq) @ block._heads(x @ block.Wk).transpose(0, 2, 1)
    s = s / n + block.rel_bias(ts, ts)[None]
    s = np.where(mask[None], s, -1e9)
    print(f"     SiLU   某行权重和 = {attn[0, -1].sum():.4f}  （≠1，可加）")
    print(f"     softmax 该行权重和 = {softmax(s[0], -1)[-1].sum():.4f}  （恒 =1，必须先拿到全部 KV）")
    print("     => 有 softmax 就要 online softmax / LSE；没有就只是加法。")
    print("        LONGER 用了 softmax ⇒ 它是唯一需要复杂 CP 的 GR（见 docs/04）。")

    # ---------------- 验证 3：二次性 ----------------
    print("\n  [验证 3] 工作量随 N 的真实增长（决定 CP 有没有对象）：")
    rows = []
    for nn in (512, 1024, 2048, 4096):
        pairs = count_pairs(make_causal(nn))
        rows.append((nn, pairs, f"{flops_attn(pairs, 16, h) / 1e9:.2f} G",
                     f"{nn * nn * 4 / 1024 ** 2:.1f} MiB"))
    from _core import table
    print(table(rows, ("N", "pairs", "FLOPs", "rab 张量")))


if __name__ == "__main__":
    demo()
