"""OpenOneRec / OneRec 最小实现 —— 「推荐」变成一个 3-token 的自回归生成问题。

论文: "OneRec: Unifying Retrieve and Rank with Generative Recommender and
      Preference Alignment", arXiv 2502.18965
      "OneRec Technical Report", arXiv 2506.13695
      "OpenOneRec Technical Report", arXiv 2512.24762

=======================================================================
三个部件，缺一不可
=======================================================================
A. Semantic ID (SID)：item 被压成 3 个离散 token
   三层 RQ-KMeans：每层取最近质心的下标，减掉质心，残差传给下一层。
   token 因此从粗到细（「运动 -> 球类 -> 篮球」），
   并且**相似 item 共享前缀** —— 后面 beam 的树结构全靠这个。

B. Encoder-Decoder
   encoder 吃用户行为历史（长，L）—— 与 LLM prefill 同构
   decoder 逐个吐 SID token（只有 3 个）—— 与 LLM decode 同构
   中间是 cross-attention

C. Beam search
   B 条候选同时推进。GR 里 B = 128~900（不是 LLM 的 4~8）。
   这是「beam 不是次要参数，而是主维度」的来源。

=======================================================================
推理的三段成本（并行分析的入口）
=======================================================================

  阶段 1  prefill (encoder)
      pairs ∝ L²          长！用户历史几千到几万
      每请求只做 1 次
      并行轴：CP（切 L）/ TP（切 d）

  阶段 2  decode (decoder, 3 步)
      每步 q = B，KV = P + B·t
      pair ∝ B·(P + B·t)   B 进入了成本
      并行轴：beam（切 B）/ CP（切 P）

  阶段 3  beam 排序
      B·V 个 logits -> 每 beam 取 2B -> 全局 top-B
      只跟 B、V 有关，与 d、L 无关
      并行轴：几乎没有（一次 topk）

关键事实：**q = B**。
LLM 的 decode 是 q=1，所以「切 KV」永远不划算（搬得比算得多）。
GR 的 decode 是 q=512，同一份 KV 搬一次能喂 512 个 query ——
这是 CP 在 GR 推理上唯一可能有对象的场景（见 parallel/cp_min.py）。
"""

from __future__ import annotations

import numpy as np

from _core import rng, step, table, title


# ======================================================================
# A. RQ-KMeans：把 item 压成 3 个 token
# ======================================================================
class RQKMeans:
    """残差量化：找最近质心 -> 取下标 -> 减质心 -> 残差进下一层。"""

    def __init__(self, n_levels: int = 3, codebook: int = 16, dim: int = 32, seed: int = 0):
        g = rng(seed)
        self.centroids = [g.normal(0, 1, (codebook, dim)) for _ in range(n_levels)]

    def encode(self, item_emb: np.ndarray) -> list[int]:
        r = np.asarray(item_emb, dtype=np.float64).copy()
        tokens = []
        for C in self.centroids:
            idx = int(np.argmin(((C - r) ** 2).sum(axis=1)))
            tokens.append(idx)
            r = r - C[idx]
        return tokens

    def decode(self, tokens: list[int]) -> np.ndarray:
        return sum(C[t] for C, t in zip(self.centroids, tokens))


# ======================================================================
# B. Encoder-Decoder（只保留形状，不追精度）
# ======================================================================
class OneRec:
    def __init__(self, d: int = 64, vocab: int = 16, n_experts: int = 4, top_k: int = 1):
        self.d, self.vocab, self.top_k = d, vocab, top_k
        g = rng(1)
        s = 1.0 / np.sqrt(d)
        self.Wq, self.Wk, self.Wv = (g.normal(0, s, (d, d)) for _ in range(3))
        self.emb = g.normal(0, 1, (vocab, d))
        self.head = g.normal(0, s, (d, vocab))
        # MoE FFN：只有 top_k 个专家被激活（OneRec: 13% 参数激活）
        self.experts = [
            (g.normal(0, s, (d, 4 * d)), g.normal(0, s, (4 * d, d)))
            for _ in range(n_experts)
        ]
        self.router = g.normal(0, s, (d, n_experts))

    # ------------------------------------------------------------------
    def encode(self, hist_ids: np.ndarray) -> np.ndarray:
        """用户历史 -> 一份 mem。真实是 L 层 self-attn，这里 1 层。"""
        x = self.emb[hist_ids]
        q, k, v = x @ self.Wq, x @ self.Wk, x @ self.Wv
        a = _softmax((q @ k.T) / np.sqrt(self.d))
        return a @ v                                    # [L, d]

    def decode_step(self, last_tok: int, mem: np.ndarray, cache: dict) -> np.ndarray:
        """一个 decode step。mem 是 encoder 输出，作为 cross-attn 的 KV。"""
        x = self.emb[last_tok][None]                    # [1, d]
        cache.setdefault("k", []).append(x @ self.Wk)
        cache.setdefault("v", []).append(x @ self.Wv)
        k = np.concatenate(cache["k"], 0)
        v = np.concatenate(cache["v"], 0)
        a = _softmax(((x @ self.Wq) @ k.T) / np.sqrt(self.d))
        z = x + a @ v
        # cross-attn 与 mem 相关项（这里只保留形状语义）
        z = z + 0.0 * ((z @ self.Wq) @ (mem @ self.Wk).T @ mem)
        # MoE FFN
        gate = z @ self.router
        pick = np.argsort(-gate[0])[: self.top_k]
        h = sum(
            np.maximum(z @ self.experts[e][0], 0) @ self.experts[e][1] for e in pick
        )
        return z + h

    def logits(self, h: np.ndarray) -> np.ndarray:
        return h @ self.head


def _softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


def _log_softmax(x: np.ndarray) -> np.ndarray:
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())


# ======================================================================
# C. beam search（朴素版：beam 摆在 batch 维）
#    后面 parallel/beam_min.py 用 tree attention 重做一遍并对比 KV 字节
# ======================================================================
def beam_search(model: OneRec, hist_ids: np.ndarray, beam: int, n_steps: int = 3):
    mem = model.encode(hist_ids)                        # 1 次 prefill
    beams = [(0.0, [])]
    pool_sizes = []
    kv_reads = 0
    for _ in range(n_steps):
        cands = []
        for score, toks in beams:
            h = model.decode_step(toks[-1] if toks else 0, mem, {"k": [], "v": []})
            kv_reads += len(hist_ids) + len(toks)       # 每条 beam 都完整读一遍
            lp = _log_softmax(model.logits(h)[0])
            for tok in range(model.vocab):
                cands.append((score + lp[tok], toks + [tok]))
        pool_sizes.append(len(cands))
        cands.sort(key=lambda x: -x[0])
        beams = cands[:beam]
    return beams, mem, pool_sizes, kv_reads


# ======================================================================
def demo() -> None:
    title("OpenOneRec：3 个 token 的自回归生成，B 条 beam 同时推进")

    # ---- A. SID ----
    print("\n  [A] RQ-KMeans：一个 item -> 3 个 token")
    rq = RQKMeans(n_levels=3, codebook=16, dim=32)
    item = rng(2).normal(0, 1, 32)
    toks = rq.encode(item)
    step("item 向量", shape=item.shape)
    step("SID", val=toks, note="3 层，从粗到细")
    step("重建误差", val=f"{np.linalg.norm(item - rq.decode(toks)):.3f}",
         note="每层只拟合剩下的残差")
    print("     => 相似 item 的 token[0] 往往相同 => beam 的树有共享前缀。")

    # 两个相近 item 共享前缀
    near = item + rng(20).normal(0, 0.05, 32)
    step("相邻 item 的 SID", val=rq.encode(near), note="前缀相同是常态")

    # ---- B. 形状轨迹 ----
    print("\n  [B] 一次前向的形状轨迹：")
    model = OneRec(d=64, vocab=16, n_experts=4, top_k=1)
    L = 256
    hist = rng(3).integers(0, 16, L)
    mem = model.encode(hist)
    step("用户历史", shape=hist.shape, note=f"L={L}")
    step("encoder 输出 mem", shape=mem.shape)
    step("decode 一步", shape=(1, 64), note="q 只有 1 行")
    print("     => encoder 吃 L，decoder 吃 1，cross-attn 的 KV 长 L。")

    # ---- C. beam ----
    print("\n  [C] beam search 的真实工作量（V=16, 3 步）：")
    rows = []
    for B in (1, 8, 128, 512):
        pool = B * model.vocab
        kv_len = L + B * 2
        pairs = B * kv_len
        rows.append((B, pool, pairs, f"{pairs / L:.1f}x"))
    print(table(rows, ("B", "候选池 B·V", "每步 pair", "相对 B=1")))
    print("     => pair ∝ B·(L + B·t)。")
    print("        B·L 项 = B 个 query 共享同一份 prompt KV（可摊薄）")
    print("        B²·t 项 = beam 之间互相看（beam 独有，随 B 二次）")

    # ---- D. 朴素 beam 的 KV 冗余 ----
    print("\n  [D] 朴素实现（beam 摆 batch 维）读了几遍 KV：")
    beams, _, pools, kv_reads = beam_search(model, hist, beam=8, n_steps=3)
    shared = L * 3                                       # tree attention：每步 1 遍
    step("朴素 KV 读取", val=f"{kv_reads} 个 token")
    step("tree attention", val=f"{shared} 个 token", note="每步只读一次")
    step("冗余倍数", val=f"{kv_reads / shared:.1f}x", note="≈ beam 宽度")
    print("     => 这就是 vLLM 慢 25 倍的机制性原因（见 parallel/beam_min.py）。")

    # ---- E. 关键：q = B ----
    print("\n  [核心] 为什么 GR 的 decode 和 LLM 不一样：")
    rows = [
        ("LLM decode", 1, "q=1，切 KV 永远不划算"),
        ("GR decode (B=128)", 128, "KV 搬一次的收益被 128 倍摊薄"),
        ("GR decode (B=512)", 512, "CP 开始可能有对象"),
        ("GR decode (B=900)", 900, "生产上限（NVIDIA GR-Inference）"),
    ]
    print(table(rows, ("场景", "q", "含义")))
    print("     => 同一个 cross-attn：LLM 里 q=1，GR 里 q=512。")
    print("        这就是『beam 让 CP 重新有对象』的全部内容。")

    # ---- F. 生产配置 ----
    print("\n  [生产配置对照]（公开报道）")
    rows = [
        ("OneRec（闭源）", "0.94B/2.6B", "24 专家 top-2/4", "128", "MoE"),
        ("OneRec-V2（闭源）", "1B", "—", "512", "MoE"),
        ("OpenOneRec-1.7B", "1.7B", "dense Qwen3", "128~512", "**无专家**"),
        ("OpenOneRec-8B", "8B", "dense Qwen3", "128~512", "**无专家**"),
        ("美团 SLM", "~0.5B", "—", "~100", "—"),
        ("NVIDIA GR-Inference", "0.6B", "—", "20→900", "—"),
    ]
    print(table(rows, ("系统", "规模", "FFN", "beam", "有没有专家")))

    # ---- G. 谁是 MoE ---- 这一节纠正一个很容易搞错的点
    print("\n  [G] 「OneRec 有专家」这句话，对开源版不成立：")
    one_rec = [("OneRec-0.015B", "Dense", "4", "128", "—", "—"),
               ("OneRec-0.121B", "Dense", "8", "1024", "—", "—"),
               ("OneRec-0.935B", "MoE", "8", "1024", "24 / 2", "Decoder"),
               ("OneRec-2.633B", "MoE", "24", "1024", "24 / 4", "Enc & Dec")]
    print("    闭源 OneRec 系列（arXiv 2506.13695, Table 1）：")
    print("    " + table(one_rec, ("模型", "类型", "层数", "隐藏维",
                                  "专家(总/激活)", "MoE 位置")).replace("\n", "\n    "))

    print("\n    开源 OpenOneRec / OneRec-Foundation（Qwen3 backbone）：")
    cfg = [("OneRec-1.7B", "Qwen3-1.7B", 2048, 28, 16, 8, 6144, "dense"),
           ("OneRec-8B", "Qwen3-8B", 4096, 36, 32, 8, 12288, "dense")]
    print("    " + table(cfg, ("模型", "backbone", "hidden", "layers",
                              "Q heads", "KV heads", "FFN dim", "FFN 类型")).replace("\n", "\n    "))

    print("\n    依据：HF config.json 为 `Qwen3ForCausalLM`，没有 num_experts /")
    print("    moe_intermediate_size 等字段；官方模型卡写明「architecture remains")
    print("    strictly consistent with the original Qwen3」，唯一改动是扩展 itemic token 词表。")
    print("    而 Qwen3 的 0.6B/1.7B/4B/8B 都是 dense（MoE 只有 30B-A3B / 235B-A22B）。")

    print("\n    两条推论：")
    print("      1. EP 对 OpenOneRec 不适用 —— 没有专家可切。")
    print("      2. KV heads = 8，而一张卡 8 芯 ⇒ TP=8 恰好覆盖 KV，")
    print("         再往上加 KVP/CP 在显存上**没有任何增量收益**（判据 A 的实测确认）。")


if __name__ == "__main__":
    demo()
