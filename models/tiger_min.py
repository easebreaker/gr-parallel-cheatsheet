"""TIGER 最小实现 —— 「3 个 token 的 SID + 小 beam + 合法元组约束」。

论文: Rajput et al., "Recommender Systems with Generative Retrieval",
      arXiv 2305.05065（NeurIPS'23，Google）

=======================================================================
三个部件
=======================================================================
A. RQ-VAE（不是 RQ-KMeans）：3 层 × 256 码字，SID = 3 个 token
   与 OneRec 的 RQ-KMeans 几乎是同一个东西，区别只在码本怎么学。

B. T5 式 encoder-decoder
   encoder 吃「历史 item 的 SID 拼接」⇒ 长度 = 3·S（S = session 长度）
   decoder 吐 3 个 token
   注意：**历史长度被乘了 3**，这是 SID 方案的一个隐性成本。

C. **约束 beam search**
   SID 三元组只有极少一部分能映射回真实 item
   ⇒ 生成出来的非法元组直接丢弃，记为 invalid generation
   ⇒ 这是最早的「SID 受限解码」，后来的 xGR / FlashTrie / 华为那篇都在做同一件事

=======================================================================
它在这张矩阵里的位置：**所有东西都很小**
=======================================================================

  S（session 长度）   ≈ 20        ⇒ encoder 只有 3S ≈ 60 个 token
  beam                = 10~20     ⇒ 比 OneRec 的 128~900 小一个半数量级
  MoE                 无          ⇒ 没有 EP 的入口
  模型                 ~几十 M     ⇒ 不需要 TP

  ⇒ **CP / EP / beam 三个轴在这里全部没有对象。**
  它是「反面基线」：说明 GR 的并行问题不是「模型叫 GR」就有，
  而是由 (q, L, B, 归约代数) 这四个数决定的。
"""

from __future__ import annotations

import numpy as np

from _core import rng, step, table, title


def _softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


# ======================================================================
class RQVAE:
    """残差量化。与 RQ-KMeans 同构，只是码本由 VAE 一起学。"""

    def __init__(self, n_levels: int = 3, codebook: int = 256, dim: int = 32, seed: int = 0):
        g = rng(seed)
        self.codebooks = [g.normal(0, 1, (codebook, dim)) for _ in range(n_levels)]
        self.n_levels = n_levels

    def encode(self, z: np.ndarray) -> list[int]:
        r = np.asarray(z, dtype=np.float64).copy()
        out = []
        for C in self.codebooks:
            idx = int(np.argmin(((C - r) ** 2).sum(axis=1)))
            out.append(idx)
            r = r - C[idx]
        return out

    def decode(self, codes: list[int]) -> np.ndarray:
        return sum(C[c] for C, c in zip(self.codebooks, codes))


class T5Seq2Seq:
    """只保留形状：encoder 吃 3S token，decoder 逐步吐 3 个。"""

    def __init__(self, d: int = 64, vocab: int = 256, seed: int = 1):
        self.d, self.vocab = d, vocab
        g = rng(seed)
        s = 1.0 / np.sqrt(d)
        self.emb = g.normal(0, 1, (vocab, d))
        self.Wq, self.Wk, self.Wv = (g.normal(0, s, (d, d)) for _ in range(3))
        self.head = g.normal(0, s, (d, vocab))

    def encode(self, codes: np.ndarray) -> np.ndarray:
        x = self.emb[codes]
        q, k, v = x @ self.Wq, x @ self.Wk, x @ self.Wv
        s = (q @ k.T) / np.sqrt(self.d)
        s = np.where(np.tril(np.ones_like(s, dtype=bool)), s, -1e9)
        return _softmax(s) @ v

    def decode_logits(self, prefix: list[int], mem: np.ndarray) -> np.ndarray:
        x = self.emb[prefix[-1] if prefix else 0][None]
        # cross-attn 到 encoder 记忆（真实是 L 层，这里 1 步）
        ctx = _softmax(((x @ self.Wq) @ (mem @ self.Wk).T) / np.sqrt(self.d)) @ (mem @ self.Wv)
        return (x + ctx) @ self.head


# ======================================================================
def constrained_beam_search(model: T5Seq2Seq, mem: np.ndarray, valid: set[tuple],
                            beam: int, n_steps: int = 3):
    """约束 beam：只保留「前缀仍能扩展成合法元组」的候选。

    这是 TIGER 与 OneRec 的最大差异之一：OneRec 用 SID 词表约束（投影只算 8192），
    TIGER 用**合法三元组集合**约束（直接查表），后者更紧、更省。
    """
    beams = [(0.0, [])]
    pools, kept, dropped = [], [], []
    for t in range(n_steps):
        cands = []
        for score, toks in beams:
            logits = model.decode_logits(toks, mem)[0]
            lp = logits - logits.max() - np.log(np.exp(logits - logits.max()).sum())
            for tok in range(model.vocab):
                nt = toks + [tok]
                # 前缀合法性：存在某个合法元组以 nt 开头
                ok = any(tuple(v[: len(nt)]) == tuple(nt) for v in valid)
                if ok:
                    cands.append((score + lp[tok], nt))
        pools.append(len(beams) * model.vocab)
        dropped.append(len(beams) * model.vocab - len(cands))
        cands.sort(key=lambda x: -x[0])
        beams = cands[:beam]
        kept.append(len(beams))
    return beams, pools, kept, dropped


# ======================================================================
def demo() -> None:
    title("TIGER：3-token SID + 小 beam + 合法元组约束")

    # ---------------- A. RQ-VAE ----------------
    print("\n  [A] RQ-VAE：3 层 × 256 码字 -> 3 个 token")
    rqvae = RQVAE(n_levels=3, codebook=256, dim=32)
    z = rng(2).normal(0, 1, 32)
    codes = rqvae.encode(z)
    step("item 隐向量", shape=z.shape)
    step("SID", val=codes, note="3 个 token")
    step("码本大小", val=f"256³ = {256**3/1e6:.1f}M", note="SID 总空间 1677 万")
    step("重建误差", val=f"{np.linalg.norm(z - rqvae.decode(codes)):.3f}")

    print("\n    对比 OneRec（RQ-KMeans，8192 码字）：")
    rows = [("TIGER", 3, 256, f"{256**3/1e6:.1f}M", "论文 NeurIPS'23"),
            ("OneRec", 3, 8192, f"{8192**3/1e9:.0f}B", "工业级，词表大得多"),
            ("OpenOneRec", 3, 8192, f"{8192**3/1e9:.0f}B", "vocab_size=176384 含特殊 token")]
    print(table(rows, ("模型", "层数", "码本", "SID 空间", "备注")))
    print("     => 码本越大，非法元组越多，「约束解码」越重要。")

    # ---------------- B. 历史长度 ×3 ----------------
    print("\n  [B] SID 的隐性成本：历史长度被乘了 3")
    rows = []
    for S in (10, 20, 50, 100):
        rows.append((S, 3 * S, f"{3*S*(3*S+1)//2}", f"{S*(S+1)//2}",
                     f"{(3*S*(3*S+1)/2)/(S*(S+1)/2):.1f}x"))
    print(table(rows, ("session 长度 S", "encoder token = 3S", "causal pair",
                       "若 1 token/item", "代价倍数")))
    print("     => 用 SID 表示历史，encoder 的 pair 数会涨 **9 倍**（3²）。")
    print("        TIGER 的 S 只有 ~20，所以绝对值仍是 1830 pair，可忽略。")
    print("        但换成工业级 L=10000，3L=30000 ⇒ pair 4.5e8 —— 这时才需要 CP。")

    # ---------------- C. 约束 beam ----------------
    print("\n  [C] 约束 beam：「非法元组直接丢」")
    S = 20
    model = T5Seq2Seq(d=64, vocab=256)
    hist_codes = rng(3).integers(0, 256, 3 * S)
    mem = model.encode(hist_codes)

    # 造一个稀疏的合法集合（真实系统是 SID 表的全部 item）
    valid = {tuple(rng(4).integers(0, 256, 3)) for _ in range(500)}
    beams, pools, kept, dropped = constrained_beam_search(model, mem, valid, beam=10)
    step("历史 token", val=3 * S, note=f"S={S}")
    step("合法元组数", val=len(valid), note="真实系统 = item 总数")
    step("每步候选池", val=pools, note="beam × vocab")
    step("被约束丢掉的", val=dropped, note="非法前缀")
    step("保留的 beam", val=kept)
    step("最终 top-1", val=beams[0][1] if beams else None)

    print("\n    三种约束方式的代价对比（beam=10, 3 步）：")
    rows = [
        ("无约束", "10 × 256 每步", "生成大量非法元组，全部作废", "最差"),
        ("SID 词表约束", "投影只算合法 token", "8192 vs 176384 = 21.5x（华为）", "中"),
        ("合法元组约束（TIGER）", "前缀查表", "候选池从 2560 掉到几十", "最好"),
    ]
    print(table(rows, ("方式", "做法", "省什么", "效果")))
    print("     => 这是最早的「受限解码」，后来的 xGR / FlashTrie / 华为那篇")
    print("        都在做同一件事，只是把查表从 CPU 搬到了 GPU。")

    # ---------------- D. 三个轴全都没有对象 ----------------
    print("\n  [D] 为什么三个并行轴在这里都失效：")
    rows = []
    for S_, beam_ in ((20, 10), (50, 20), (100, 20)):
        enc_pairs = 3 * S_ * (3 * S_ + 1) // 2
        kv = 3 * S_ * 64 * 2 * 2 * 12          # 3S·d·K/V·bf16·12 层
        rows.append((S_, beam_, f"{enc_pairs/1e3:.1f} K",
                     f"{kv/1024**2:.2f} MiB",
                     f"{beam_ * 3 * S_/1e3:.1f} K",
                     "全部装得下"))
    print(table(rows, ("S", "beam", "encoder pair", "encoder KV", "decode pair",
                       "容量判据")))
    print("     => encoder pair 只有几千，KV 只有几 MB，beam 只有 10~20。")
    print("        CP：KV 装得下 ⇒ 不需要；EP：没有专家；beam：太小，切了亏。")

    print("\n  [定位] TIGER 是这张矩阵的「反面基线」：")
    rows = [
        ("CP", "无对象", "3S≈60 token，KV≈MB 级"),
        ("EP", "不适用", "dense T5，没有专家"),
        ("TP", "不需要", "模型只有几十 M"),
        ("beam", "无对象", "B=10~20，全局 top-B 的通信与计算都可忽略"),
    ]
    print(table(rows, ("轴", "结论", "原因")))
    print("     => 结论：**「叫 GR」不等于「需要并行」。**")
    print("        决定权在 (q, L, B, 归约代数) 这四个数上，不在名字上。")


if __name__ == "__main__":
    demo()
