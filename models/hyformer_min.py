"""HyFormer 最小实现 —— 「用十几个 query 读整段历史」。

论文: "HyFormer: Revisiting the Roles of Sequence Modeling and Feature
      Interaction in CTR Prediction", arXiv 2601.12681（SIGIR'26）

=======================================================================
它把「序列建模」和「特征交叉」缝进了同一个 block
=======================================================================

每一层两个模块，交替：

  ① Query Decoding（序列建模）
       Q  = N 个 semantic query（由非序列特征 MLP 生成，N 很小，~16）
       K,V= 长行为序列的逐层投影
       Q ← CrossAttn(Q, K, V)                ← 代价 ∝ N · L · d

  ② Query Boosting（特征交叉）
       Q' = [Q | 非序列特征 token]            ← T = N + M 个
       Q' ← MLP-Mixer 式 token mixing         ← **零参数、零 FLOPs**
       Q' ← per-token FFN                     ← 代价 ∝ T · d · d_ff

=======================================================================
为什么它和前面所有模型都不一样
=======================================================================

  **q : L 是四个模型里最小的。** N≈16，L 可达上万 ⇒ q/L ≈ 1e-3。

  于是 Query Decoding 的代价 ∝ N·L 很小，但那**不是因为它便宜，
  而是因为它把大部分工作搬给了「读 KV」**：
      算的部分   ∝ N · L · d      （N 小 → 小）
      读的部分   ∝ L · d · bytes  （L 大 → 带宽主导）

  ⇒ 这是唯一一个「计算不贵、带宽贵」的模型。
  ⇒ 也是唯一一个 CP（切 KV）**真的有用**、但**不带来算力收益**的模型。
     它买的是 TPOT 上界，不是 FLOPs。

  这正是 Medha 的 KVP 的适用场景，也是 LLM decode 的形态。
  HyFormer 等于「把 LLM decode 的那个形状搬到了推荐里」。
"""

from __future__ import annotations

import numpy as np

from _core import rng, step, table, title


def _softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


# ======================================================================
class QueryDecoding:
    """CrossAttn(Q: N 个 query, K/V: L 长的行为序列)。"""

    def __init__(self, d: int = 48, n_query: int = 16):
        self.d, self.N = d, n_query
        s = 1.0 / np.sqrt(d)
        g = rng(0)
        self.Wq, self.Wk, self.Wv = (g.normal(0, s, (d, d)) for _ in range(3))

    def forward(self, q_tokens: np.ndarray, seq: np.ndarray) -> np.ndarray:
        q = q_tokens @ self.Wq
        k, v = seq @ self.Wk, seq @ self.Wv
        return q_tokens + _softmax((q @ k.T) / np.sqrt(self.d)) @ v

    def cost(self, L: int) -> tuple[int, int]:
        """返回 (pair 数, KV 读取的元素数)。两者是不同的东西。"""
        return self.N * L, L * self.d


class MultiHeadTokenMixing:
    """与 RankMixer 完全相同的零参数转置。"""

    def __init__(self, T: int, d: int, H: int | None = None):
        self.T, self.d, self.H = T, d, H or T

    def forward(self, x: np.ndarray) -> np.ndarray:
        T, d, H = self.T, self.d, self.H
        assert d % H == 0 and x.shape[0] == T
        return x.reshape(T, H, d // H).transpose(1, 0, 2).reshape(H, T * d // H)


class PerTokenFFN:
    def __init__(self, T: int, d: int, k: int = 4):
        self.d, self.k = d, k
        s = 1.0 / np.sqrt(d)
        g = rng(1)
        self.W1 = g.normal(0, s, (T, d, k * d))
        self.W2 = g.normal(0, s, (T, k * d, d))

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.einsum("td,tdk->tk", x, self.W1)
        h = h / (1.0 + np.exp(-h))
        return np.einsum("tk,tkd->td", h, self.W2)


class QueryBoosting:
    """MLP-Mixer 式 token mixing + per-token FFN。"""

    def __init__(self, T: int, d: int):
        self.mix = MultiHeadTokenMixing(T, d)
        self.ffn = PerTokenFFN(T, d)

    def forward(self, x: np.ndarray) -> np.ndarray:
        s = self.mix.forward(x)
        return x + self.ffn.forward(s)


class HyFormerLayer:
    def __init__(self, d: int = 48, n_query: int = 16, n_ns: int = 8):
        self.d, self.N, self.M = d, n_query, n_ns
        self.dec = QueryDecoding(d, n_query)
        self.boost = QueryBoosting(n_query + n_ns, d)

    def forward(self, q: np.ndarray, seq: np.ndarray) -> np.ndarray:
        q = self.dec.forward(q, seq)                      # ① 读序列
        q_ns = np.concatenate([q, self.ns], 0)            # ② 拼上非序列特征
        q_ns = self.boost.forward(q_ns)                   #    转置 + 逐 token FFN
        return q_ns[: self.N]


# ======================================================================
def demo() -> None:
    title("HyFormer：q 只有 16 个，所以廉价的是算、昂贵的是读")

    d, N, M, L = 48, 16, 8, 4096   # d 必须能被 T=N+M 整除（H=T）
    dec = QueryDecoding(d, N)
    boost = QueryBoosting(N + M, d)

    q = rng(2).normal(0, 1, (N, d))
    seq = rng(3).normal(0, 1, (L, d))
    out = dec.forward(q, seq)
    step("非序列 query", shape=q.shape, note=f"N={N}")
    step("行为序列", shape=seq.shape, note=f"L={L}")
    step("Query Decoding 输出", shape=out.shape)
    step("Query Boosting 输出", shape=boost.forward(np.concatenate([out, np.zeros((M, d))], 0)).shape)

    # ---------------- 算 vs 读 ----------------
    F_EFF, BW = 150e12, 800e9        # 有效算力（FLOP/s）、HBM 带宽（B/s）

    print("\n  [核心] Query Decoding 里，「算」与「读」随 L 的走势不同")
    print("         （下表是**单层单样本**；乘上 层数×batch 才是整网，但比值不变）：")
    rows = []
    for LL in (512, 1024, 4096, 16384, 65536):
        pairs, _ = dec.cost(LL)
        t_compute = 2 * pairs * d / F_EFF * 1e6            # us
        t_read = 4 * LL * d / BW * 1e6                     # K+V，bf16
        rows.append((LL, f"{pairs/1e3:.0f} K", f"{t_compute:.2f} us",
                     f"{4*LL*d/1024**2:.1f} MiB", f"{t_read:.2f} us",
                     f"{t_read/t_compute:.0f}x"))
    print(table(rows, ("L", "pair", "计算", "KV 读取", "读耗时", "读/算")))
    print("     => **两条线都 ∝ L，比值却是常数**：")
    print("            t_算 / t_读 = N · BW / (2 · F)   （与 L、d 都无关）")
    print(f"            N=16, BW=800GB/s, F=150TFLOPS  ->  {(16*BW/(2*F_EFF)):.3f}")
    print("        所以读永远主导，且 N 越小越糟。这就是 LLM decode 的形状（q=1）。")

    # ---------------- N 的敏感性 ----------------
    print("\n  [N 的作用] query 数只动「算」端，改不了「读」端：")
    rows = []
    t_read = 4 * 16384 * d / BW * 1e6
    for NN in (8, 16, 32, 128, 512):
        pairs = NN * 16384
        t_compute = 2 * pairs * d / F_EFF * 1e6
        rows.append((NN, f"{pairs/1e3:.0f} K", f"{t_compute:.2f} us",
                     f"{t_read:.2f} us",
                     f"{t_compute/t_read:.2f}"))
    print(table(rows, ("N", "pair @ L=16K", "计算", "读", "算/读")))
    print("     => N 从 8 到 512（64 倍），总时间只从 ~1.00 涨到 ~1.34 倍。")
    print("        因为读那一端没变。所以「优化 query 数」在这里是白费力气 ——")
    print("        与 LONGER 恰好相反（那边减少 q 能省 46% FLOPs）。")

    # ---------------- CP 对它的意义 ----------------
    print("\n  [CP 的意义] 切 KV 之后，两条线怎么变：")
    rows = []
    for C in (1, 2, 4, 8):
        P = 32768
        t_compute = 2 * (N * P / C) * d / F_EFF * 1e6
        t_read = 4 * (P / C) * d / BW * 1e6
        t_merge = N * d * 4 / 200e9 * 1e6           # LSE 合并，∝ N·d，与 P 无关
        rows.append((C, f"{t_compute:.2f}", f"{t_read:.2f}",
                     f"{t_merge:.2f}", f"{t_compute+t_read+t_merge:.2f}"))
    print(table(rows, ("C", "计算 us", "读 us", "合并 us", "合计 us")))
    print("     => 时间确实按 1/C 降，但**降的是读取，不是算力**。")
    print("        所以它买到的是「TPOT 上界」，不是「吞吐」。")
    print("        这与 LONGER 的 1.2~1.3× Amdahl 是两种不同的收益，别混。")

    # ---------------- 两段交叉注意力的形状 ----------------
    print("\n  [易混点] HyFormer 有两段交叉注意力，KV 来源不同：")
    rows = [
        ("decode", "short_len (~16)", "short_len (~16)", "两段都小，几乎不计",
         "Q 与 KV 一起变"),
        ("seq_cross", "short_len (~16)", "kv_len (可达 8K)", "∝ short_len · kv_len",
         "只有 KV 变"),
    ]
    print(table(rows, ("阶段", "Q", "KV", "代价", "扫描时动谁")))
    print("     => 扫 short_len 时两段一起动，扫 kv_len 时只有 seq_cross 动。")
    print("        这个差异是「辨识」的关键：自变量不成比例，才有独立信息。")

    # ---------------- 并行含义 ----------------
    print("\n  [并行含义]：")
    rows = [
        ("CP", "**真的有用，但只是延迟**", "q/L≈1e-3 ⇒ 算的少；读的多 ⇒ 切 KV 是唯一出路"),
        ("TP", "有用", "切 d，减轻权重与 KV 带宽"),
        ("EP", "仅当 MoE", "原论文是 dense LRM，没有专家"),
        ("beam", "不适用", "判别式，没有自回归 decode"),
    ]
    print(table(rows, ("轴", "结论", "原因")))


if __name__ == "__main__":
    demo()
