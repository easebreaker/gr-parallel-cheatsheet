"""RankMixer 最小实现 —— 「把 attention 换成免费的数据搬运」。

论文: Zhu et al., "RankMixer: Scaling Up Ranking Models in Industrial
      Recommenders", arXiv 2507.15551（CIKM'25，字节）

=======================================================================
它到底做了什么（两句话）
=======================================================================
1. Multi-head Token Mixing (MHTM)：把每个 token 沿特征维切成 H 份，
   再把所有 token 的第 h 份拼成一个新 token。
   —— 这是一次 reshape + transpose，**零参数、零 FLOPs**。

2. Per-token FFN：每个 token 用**自己专属**的 FFN 参数。
   —— 参数是共享 FFN 的 T 倍，但 FLOPs 完全一样。

   +-----------------------------------------------------------------+
   |  输入  T 个 token × D 维                                        |
   |    x_0 = [h0|h1|h2|h3]      <- 每个 token 切成 H=4 份           |
   |    x_1 = [h0|h1|h2|h3]                                          |
   |    x_2 = [h0|h1|h2|h3]                                          |
   |                                                                 |
   |  MHTM（转置，免费）                                             |
   |    s_0 = [x0.h0 | x1.h0 | x2.h0]   <- 同一个 head 跨 token 拼接  |
   |    s_1 = [x0.h1 | x1.h1 | x2.h1]                                |
   |    s_2 = [x0.h2 | x1.h2 | x2.h2]                                |
   |                                                                 |
   |  逐 token FFN（各自独立参数）                                    |
   |  ---------------------------------------------------------------|
   +-----------------------------------------------------------------+

=======================================================================
为什么这一页是「并行性」而不是「精度」的全部关键
=======================================================================

  **T 不是序列长度。**
  T 是「特征子空间」的个数，是模型超参（论文里 H=T，典型 16 左右）。
  用户行为历史 L 根本不进这个模块 —— L 在序列侧被别的模块消化。

  所以：
    pairs（query,key 对）  = 0        <- 没有 attention，没有 pair
    复杂度                 ∝ T·D²·k   <- 线性于 T，与 L 无关
    转置成本               ∝ T·D     <- MTE/拷贝，可被融合成 0

  => CP 在 RankMixer 上**没有对象**：不存在「按序列切」这件事。
     能切的是 T（特征子空间）和 D（隐藏维），以及 MoE 的专家。

  论文自己的数字说明了同一件事：MFU 4.5% -> 45%。
  它把模型从 memory-bound 推到 compute-bound —— 也就是说，
  瓶颈从「搬数据」变成了「算 GEMM」，而那正是 TP / EP 的强项。
"""

from __future__ import annotations

import numpy as np

from _core import rng, step, table, title


# ======================================================================
class MultiHeadTokenMixing:
    """零参数的 token 混合：split-head -> transpose -> concat。"""

    def __init__(self, T: int, D: int, H: int | None = None):
        assert D % (H or T) == 0
        self.T, self.D = T, D
        self.H = H or T          # 论文：H = T，保证 token 数不变，残差能加

    def forward(self, x: np.ndarray) -> np.ndarray:
        """[T, D] -> [H, T*D/H]。纯 reshape/transpose。"""
        T, D, H = self.T, self.D, self.H
        dh = D // H
        return x.reshape(T, H, dh).transpose(1, 0, 2).reshape(H, T * dh)

    def inverse(self, s: np.ndarray) -> np.ndarray:
        """[H, T*D/H] -> [T, D]。残差相加前必须转回去。"""
        T, D, H = self.T, self.D, self.H
        dh = D // H
        return s.reshape(H, T, dh).transpose(1, 0, 2).reshape(T, D)


class PerTokenFFN:
    """每个 token 一套参数，输入是它自己那一路。"""

    def __init__(self, n_tokens: int, D: int, k: int = 4):
        self.D, self.k = D, k
        g = rng(0)
        s = 1.0 / np.sqrt(D)
        self.W1 = g.normal(0, s, (n_tokens, D, k * D))
        self.W2 = g.normal(0, s, (n_tokens, k * D, D))

    def forward(self, x: np.ndarray) -> np.ndarray:
        # 第 i 个 token 只跟自己那组权重做乘 —— 参数隔离
        h = np.einsum("td,tdk->tk", x, self.W1)
        h = h * (1.0 / (1.0 + np.exp(-h)))          # SiLU/GELU 均可
        return np.einsum("tk,tkd->td", h, self.W2)

    @property
    def params(self) -> int:
        return self.W1.size + self.W2.size


class RankMixerBlock:
    def __init__(self, T: int, D: int, k: int = 4):
        self.mix = MultiHeadTokenMixing(T, D)
        self.pffn = PerTokenFFN(T, D, k)

    def forward(self, x: np.ndarray) -> np.ndarray:
        s = self.mix.forward(x)
        s = s + self.pffn.forward(s)                 # 残差（形状都是 [T, D]）
        return s


# ======================================================================
class ReLURouter:
    """Sparse-MoE 的路由：ReLU 门 + L1 惩罚。

    与 Top-k 的区别：每个 token 激活的专家数**不同**，
    高信息量的 token 可以自己多要几个专家。
    代价是负载更难均衡 —— 这正是 EP 的输入端。
    """

    def __init__(self, D: int, n_experts: int, l1: float = 1e-3):
        self.n_experts, self.l1 = n_experts, l1
        self.Wg = rng(9).normal(0, 1 / np.sqrt(D), (D, n_experts))

    def forward(self, x: np.ndarray):
        gate = np.maximum(x @ self.Wg, 0.0)          # ReLU，不是 softmax
        active = (gate > 0).sum(axis=1)              # 每个 token 的激活数不同
        penalty = self.l1 * np.abs(gate).sum()       # L1 把稀疏度压出来
        return gate, active, penalty


def moe_ffn(gate: np.ndarray, x: np.ndarray, experts: list[np.ndarray]) -> np.ndarray:
    """稀疏计算：只有 gate>0 的 (token, expert) 对才做 FFN。"""
    out = np.zeros_like(x)
    for e, W in enumerate(experts):
        rows = gate[:, e] > 0
        if rows.any():
            out[rows] += (x[rows] @ W) * gate[rows, e][:, None]
    return out


# ======================================================================
def demo() -> None:
    title("RankMixer：没有 L，没有 pair，所以 CP 没有对象")

    T, D, k = 16, 64, 4
    blk = RankMixerBlock(T, D, k)
    x = rng(1).normal(0, 1, (T, D))
    y = blk.forward(x)

    step("输入", shape=x.shape, note="T 个特征 token，D 维")
    step("MHTM 输出", shape=blk.mix.forward(x).shape)
    step("MHTM 参数", val=0, note="纯 reshape/transpose")
    step("PFFN 参数", val=f"{blk.pffn.params/1e6:.3f} M",
         note=f"共享 FFN 的 {T} 倍")
    step("PFFN FLOPs", val=f"{2*T*D*k*D*2/1e6:.3f} M")
    print("     => 注意：**没有 Q、没有 K、没有 pair**。")

    # ---------------- 转置到底在做什么 ----------------
    print("\n  [看懂 MHTM] 用小例子打印 head 的去向（T=3, D=6, H=3）：")
    small = np.arange(18, dtype=float).reshape(3, 6)
    mix = MultiHeadTokenMixing(3, 6, 3)
    s = mix.forward(small)
    print("     输入 x[t, d]：")
    for t in range(3):
        print(f"       token{t}: {small[t]}")
    print("     输出（每个新 token 是『同一个 head 跨 token』拼出来的）：")
    for h in range(3):
        print(f"       s{h}: {s[h]}")
    print(f"     inverse(forward(x)) == x ? {np.allclose(mix.inverse(s), small)}")
    print("     => 一次 0-FLOP 的转置，把『token 之间』的信息换了个位置。")

    # ---------------- 与 self-attention 的代价对比 ----------------
    print("\n  [代价] 同样 T 个 token，三种混合方式的成本：")
    rows = []
    for TT in (16, 64, 256):
        attn = 2 * TT * TT * D * 2               # QK^T + AV
        mixv = 0                                  # 参数free
        pffn = 2 * TT * D * k * D * 2
        rows.append((TT, f"{mixv} (纯搬运)", f"{attn/1e6:.2f} M", f"{pffn/1e6:.2f} M",
                     f"{attn/pffn:.2f}x"))
    print(table(rows, ("T", "MHTM", "self-attn", "PFFN", "attn/PFFN")))
    print("     => T 越大，self-attention 越亏；MHTM 一直免费。")
    print("        这就是论文说的『replace quadratic self-attention』。")

    # ---------------- 关键：T 与序列长度无关 ----------------
    print("\n  [核心事实] 用户历史 L 从 512 涨到 8192，这个模块的工作量不变：")
    rows = []
    for L in (512, 1024, 2048, 4096, 8192):
        rows.append((L, f"{2*T*D*k*D*2/1e6:.3f} M", f"{T*D*8/1024:.1f} KiB",
                     "0"))
    print(table(rows, ("用户历史 L", "本模块 FLOPs", "激活字节", "pairs")))
    print("     => 4 列里有 3 列与 L 无关。")
    print("        所以『按序列切（CP）』在 RankMixer 上没有定义。")

    # ---------------- Sparse-MoE：这才是 EP 的入口 ----------------
    print("\n  [Sparse-MoE] ReLU 路由让每个 token 激活不同数量的专家：")
    router = ReLURouter(D, n_experts=8)
    gate, active, penalty = router.forward(y)
    step("gate 形状", shape=gate.shape, note="[T, n_experts]")
    step("每 token 激活专家数", val=list(active), note="不固定")
    step("平均激活", val=f"{active.mean():.2f} / 8")
    step("L1 惩罚", val=f"{penalty:.4f}", note="把稀疏度压出来")
    step("稀疏 FFN 的 (token,expert) 对", val=int((gate > 0).sum()),
         note="这就是 EP 的 all-to-all token 数")

    print("\n  [并行含义] RankMixer 的四条结论：")
    rows = [
        ("CP", "无对象", "没有 L 维，T 是特征子空间数（~16）"),
        ("TP", "可以，但 T 太小", "切 D 有效；切 T 会打破 per-token FFN 的意义"),
        ("EP", "**主战场**", "Sparse-MoE 的专家数才是参数量所在"),
        ("beam", "无关", "RankMixer 是判别式，没有自回归 decode"),
    ]
    print(table(rows, ("轴", "状态", "原因")))


if __name__ == "__main__":
    demo()
