"""beam 的三次提速 —— 每一层都只改一件事。

对应工作: vLLM PagedAttention (COW 共享, 55% 显存)
          Trie-based beam decoding (arXiv 2502.00085)
          xGR / NVIDIA GR-Inference（ContextKV + BeamKV + BeamPath）
          华为 "Optimize the Cob, Then the Kernel" (arXiv 2611.23339)

=======================================================================
同一个 beam search，三种摆法
=======================================================================

  (0) 朴素：beam 摆在 batch 维
      Q = [B, d]，KV = 每条 beam 各一份 [P + t, d]
      => 同一份 prompt KV 被读 B 遍

  (1) Tree Attention：beam 摆在**序列维**
      把 B 个新 token 拼到 prompt 后面：[P + B·t]
      用一个树形掩码限制「每个 token 只能看自己的祖先」
      => prompt KV 只读 1 遍。**这是 25× 的主要来源。**

  (2) 静态 KV + 掩码代替重排
      预分配 [P + B·(N−1)] 的连续空间，不做 block 管理
      beam 被剪掉时不做 KV 重排，只更新掩码
      => 消掉每步的 KV copy 与 PagedAttention 的元数据开销

  (3) 循环展开 + 图
      3 步 decode 全部塞进一张图，beam 选择留在设备上
      => 消掉 host 同步（vLLM 每步一次 engine invocation）

=======================================================================
三层的收益量级（华为 910B4，OpenOneRec-1.7B，L=1024）
=======================================================================
    BW=128 : vLLM 244.7 ms -> 59.1 ms   (4.1x)
    BW=512 : vLLM 1989  ms -> 79.0 ms   (25x)
    BW 从 128 涨到 512（4x），时间只涨 1.69x（14.2 -> 24.0 ms）

最后一行是最重要的事实：**beam 不是线性成本**。
把 t(B) = F + a·B + b·B² 拿两个点解，得到 F ≥ 10.93 ms，
于是「无限卡」时的 Amdahl 上限：
    B=128: 14.2 / 10.93 = 1.30x
    B=512: 24.0 / 10.93 = 2.20x
=> beam 并行不是能效杠杆，它是一个**天花板很低的轴**。
"""

from __future__ import annotations

import numpy as np

from _core import mib, rng, step, table, title


# ======================================================================
def _softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(-1, keepdims=True)


def _attn(q, k, v, mask):
    """mask: [nq, nk] bool。标准 softmax attention。"""
    s = (q @ k.T) / np.sqrt(q.shape[-1])
    s = np.where(mask, s, -1e9)
    return _softmax(s) @ v


# ======================================================================
class BeamSearch:
    """一个小到能一眼看完的 beam search。

    prompt: [P, d]          每步的 K/V 来自这里
    parents[t][i]: 第 t 步第 i 条 beam 来自上一步的第几个父 beam
    """

    def __init__(self, P=8, d=8, B=4, vocab=8, steps=3, seed=0):
        self.P, self.d, self.B, self.V, self.steps = P, d, B, vocab, steps
        g = rng(seed)
        self.prompt = g.normal(0, 1, (P, d))
        self.Wq, self.Wk, self.Wv = (
            g.normal(0, 1 / np.sqrt(d), (d, d)) for _ in range(3)
        )
        self.emb = g.normal(0, 1, (vocab, d))
        self.head = g.normal(0, 1 / np.sqrt(d), (d, vocab))
        # 固定一份父指针，保证 naive 与 tree 走同一棵搜索树
        self.parents = [np.arange(B)]
        for _ in range(steps - 1):
            self.parents.append(g.integers(0, B, B))

    # ------------------------------------------------------------------
    def _path(self, pick: list[int]) -> list[int]:
        """一条 beam 祖先的 (step, beam) 序列。pick[t] = 第 t 步取了哪条。"""
        return list(zip(range(len(pick)), pick))

    # ------------------------------------------------------------------
    def naive_step(self, t: int, tokens: list[list[int]]):
        """(0) 每条 beam 当成独立序列，各算一遍。"""
        outs, kv_read = [], 0
        for b in range(self.B):
            pk = self._ancestor_tokens(t, tokens, b)
            kv = np.concatenate([self.prompt, self.emb[pk]], 0)
            kv_read += kv.shape[0]
            q = self.emb[tokens[t][b]][None]
            out = _attn(q @ self.Wq, kv @ self.Wk, kv @ self.Wv,
                        np.ones((1, kv.shape[0]), dtype=bool))
            outs.append(out[0])
        return np.stack(outs), kv_read

    def _ancestor_tokens(self, t: int, tokens: list[list[int]], b: int) -> list[int]:
        path, cur = [], b
        for step in range(t, 0, -1):
            path.append(tokens[step - 1][cur])
            cur = int(self.parents[step][cur])
        return list(reversed(path))

    # ------------------------------------------------------------------
    def tree_step(self, t: int, tokens: list[list[int]]):
        """(1) 一个 buffer + 树掩码，prompt 只读一遍。"""
        # buffer 布局：prompt | step0 的 B 个 | step1 的 B 个 | ... | step(t-1) 的 B 个
        buf = [self.prompt]
        slot = {}                                  # (step, beam) -> 行号
        for s in range(t):
            buf.append(self.emb[tokens[s]])
            for b in range(self.B):
                slot[(s, b)] = self.P + s * self.B + b
        kv = np.concatenate(buf, 0)

        # 掩码：每行只看 prompt + 自己的祖先链
        nq = self.B
        mask = np.zeros((nq, kv.shape[0]), dtype=bool)
        mask[:, : self.P] = True
        for b in range(self.B):
            # 祖先链：从第 t 条 beam 回溯
            cur = b
            for s in range(t - 1, -1, -1):
                mask[b, slot[(s, cur)]] = True
                cur = int(self.parents[s + 1][cur])

        q = self.emb[tokens[t]] @ self.Wq
        out = _attn(q, kv @ self.Wk, kv @ self.Wv, mask)
        return out, kv.shape[0], mask

    # ------------------------------------------------------------------
    def run(self):
        """跑完 steps 步，同时统计两种实现的 KV 读取量。"""
        tokens = [[int(rng(9).integers(0, self.V)) for _ in range(self.B)]
                  for _ in range(self.steps)]
        naive_kv = tree_kv = 0
        naive_out = tree_out = None
        mask_bytes = 0
        for t in range(self.steps):
            naive_out, r0 = self.naive_step(t, tokens)
            tree_out, r1, mask = self.tree_step(t, tokens)
            naive_kv += r0
            tree_kv += r1
            mask_bytes += mask.nbytes
        return naive_out, tree_out, naive_kv, tree_kv, mask_bytes


# ======================================================================
def demo() -> None:
    title("beam：三次提速，每一层只改一件事")

    bs = BeamSearch(P=8, d=8, B=4, vocab=8, steps=3)
    naive, tree, naive_kv, tree_kv, mask_b = bs.run()

    print("\n  [正确性] 两种摆法必须算出同一个结果：")
    step("B", val=bs.B, note="先用小尺寸验证，避免数值噪声")
    step("naive vs tree", max_err=f"{np.abs(naive - tree).max():.2e}",
         verdict="一致" if np.allclose(naive, tree, atol=1e-12) else "不一致")
    print("     => tree attention 不是近似，它是把同一批 (query,key) 对换了个布局。")

    # ---------------- 三层的字节账 ----------------
    print("\n  [第一层] KV 读取量（B=4, P=8, 3 步）：")
    step("naive", val=f"{naive_kv} 行", note="每条 beam 各读一遍 prompt")
    step("tree", val=f"{tree_kv} 行", note="prompt 只读一遍")
    step("比值", val=f"{naive_kv / tree_kv:.2f}x")
    print("     => beam 越大，这个比值越接近 B。这就是 25× 的机制。")

    print("\n  [第二层] 静态 KV 与『掩码代替重排』：")
    rows = []
    for B in (128, 512, 900):
        P, d, steps = 1024, 2048, 3
        kv_alloc = (P + B * (steps - 1)) * d * 2 * 2      # K+V, bf16
        reorder = B * P * d * 2 * 2                       # 最坏情况整段搬
        mask = B * (P + B * steps) * 1                    # bool
        rows.append((B, mib(kv_alloc), mib(reorder), mib(mask),
                     f"{reorder / mask:.0f}x"))
    print(table(rows, ("B", "静态 KV 分配", "vLLM 重排(最坏)", "树掩码", "重排/掩码")))
    print("     => 掩码比 KV 重排小两三个数量级。")
    print("        所以『不搬 KV，改掩码』是这一层收益的全部来源。")

    print("\n  [第三层] host 同步：vLLM 每步一次 engine invocation：")
    print(table([(3, 3, 1), (4, 4, 1), (5, 5, 1)],
                ("decode 步数", "vLLM 调用次数", "展开后")))
    print("     => 展平成一张图之后，3 步变成 1 次 launch。")
    print("        这就是为什么『图编译』和『算子融合』的收益比『优化 attention kernel』大。")

    # ---------------- 成本律 ----------------
    print("\n  [成本律] t(B) = F + a·B + b·B^2，用公开的两点反解：")
    t128, t512 = 14.2, 24.0
    # 两点三未知：可以严格给出 F 的下界
    F_min = (512 * t128 - 128 * t512) / (512 - 128)
    step("F 的下界", val=f"{F_min:.2f} ms", note="取 b >= 0")
    step("B=128 天花板", val=f"{t128 / F_min:.2f}x")
    step("B=512 天花板", val=f"{t512 / F_min:.2f}x")
    print("     => 这是『无限张卡也拿不到』的上限。")
    print("        结论：beam 并行不是能效杠杆，天花板只有 1.3~2.2x。")

    print("\n  [结论表] 三层各自解决什么：")
    rows = [
        ("(1) Tree Attention", "prompt KV 被读 B 遍", "B 倍"),
        ("(2) 静态 KV + 掩码", "每步 KV 重排 + PagedAttention 元数据", "1~2 个数量级"),
        ("(3) 循环展开 + 图", "每步一次 host 同步", "取决于步数"),
        ("(4) SID 受限解码", "在全词表上算 logits", "21.5x（8192 vs 176384）"),
    ]
    print(table(rows, ("手段", "消掉的东西", "量级")))


if __name__ == "__main__":
    demo()
