# 04 · OpenOneRec / OneRec

> **一句话**：把推荐变成一个 3-token 的自回归生成问题，
> 于是它是这四个模型里**唯一有 beam 这一维**的，也是唯一让 `q` 变大的。

## 论文

- *OneRec: Unifying Retrieve and Rank with Generative Recommender and Preference
  Alignment*, **arXiv 2502.18965**（快手）
- *OneRec Technical Report*, **arXiv 2506.13695**
- *OpenOneRec Technical Report*, **arXiv 2512.24762**（开源版 + benchmark）

## 结构

```
  用户行为历史 (L 个 item)                     候选 SID
        │                                        │
        ▼                                        │
  ┌──────────────┐                               │
  │   Encoder    │  ← L 层 self-attention         │
  │  (prefill)   │     复杂度 O(L²)               │
  └──────┬───────┘                               │
         │ mem [L, d]                            │
         ▼                                       │
  ┌──────────────────────────────────────────────┴───┐
  │   Decoder（3 步）                                 │
  │      step 1: tok1 ~ P(· | mem)                    │
  │      step 2: tok2 ~ P(· | mem, tok1)              │
  │      step 3: tok3 ~ P(· | mem, tok1, tok2)        │
  │      ↑ 每步都维护 B 条 beam                        │
  └───────────────────────────────────────────────────┘
         │
         ▼
   SID (3 个 token) ──► 反查 item ──► 排序 / 返回
```

## 三个部件

### A. Semantic ID：3 个 token 表示一个 item

三层 RQ-KMeans（残差量化）：

```
item 向量 r0
   │  找最近质心 → token1 = argmin‖C1 − r0‖
   │  r1 = r0 − C1[token1]          ← 残差
   │  找最近质心 → token2 = argmin‖C2 − r1‖
   │  r2 = r1 − C2[token2]
   │  ...
   ▼  token3
```

两个后果：

1. **从粗到细**：token1 是「运动」，token2 是「球类」，token3 是「篮球」
2. **相似 item 共享前缀** —— 这是 beam 的树结构能共享 KV 的前提

### B. Encoder-Decoder

| 部分 | 吃什么 | 复杂度 | 对应 LLM |
| --- | --- | --- | --- |
| Encoder | 用户历史 L | O(L²) | prefill |
| Decoder | 3 个 token × B 条 beam | O(B·(L+B·t)) | decode |
| Cross-attn | query=B，KV=L | 同上 | —— |

### C. Beam search

GR 的 beam 和 LLM 的 beam 不是一个东西：

| | LLM | GR |
| --- | --- | --- |
| beam 宽度 | 4~8 | **128~900** |
| 输出长度 | 几十~几千 | **3~5** |
| 目的 | 找一个更好的句子 | 找**一批**不同的候选 item |
| 词表约束 | 无 | 只能出合法 SID（受限解码） |

## 推理的三段成本

| 阶段 | 公式 | 谁主导 |
| --- | --- | --- |
| 1 prefill | `pairs ∝ L²` | 长序列 |
| 2 decode | `pairs ∝ B·(L + B·t)` | beam 宽度 |
| 3 beam 排序 | `∝ B·V` | 词表大小 |

decode 的展开：

```
pairs = B·L  +  B²·t
        ↑        ↑
     可摊薄     beam 独有
```

- `B·L`：B 个 query 共享同一份 prompt KV ⇒ **每 query 的读取量 ∝ 1/B**
- `B²·t`：beam 之间互相看（tree attention 下的树内交互）

## 关键事实：q = B

```
LLM decode        q = 1      切 KV 永远不划算（搬得比算得多）
GR decode B=128   q = 128    KV 搬一次的收益被摊薄 128 倍
GR decode B=512   q = 512    CP 开始可能有对象
GR decode B=900   q = 900    生产上限
```

**同一个 cross-attention，LLM 里 q=1，GR 里 q=512。**
这一个数字改变了 CP 的全部结论 —— 见 `05-CP.md` 的场景 2。

## 公开的生产配置

| 系统 | 规模 | FFN | beam | 有没有专家 |
| --- | --- | --- | --- | --- |
| OneRec（闭源） | 0.94B / 2.6B | **MoE**（24 专家，top-2/4） | 128 | 有 |
| OneRec-V2（闭源） | 1B | **MoE** | 512 | 有 |
| **OpenOneRec-1.7B** | 1.7B | **dense Qwen3** | 128~512 | **无** |
| **OpenOneRec-8B** | 8B | **dense Qwen3** | 128~512 | **无** |
| 美团 SLM 引擎 | ~0.5B | — | ~100 | — |
| NVIDIA GR-Inference | 0.6B | — | 20→900 | — |

## MoE：谁有，谁没有

这一节纠正一个非常容易搞错的点。**「OneRec 用了 MoE」这句话只对闭源版成立。**

闭源 OneRec 系列（arXiv 2506.13695, Table 1）：

| 模型 | 类型 | 层数 | 隐藏维 | 专家（总/激活） | MoE 位置 |
| --- | --- | --- | --- | --- | --- |
| OneRec-0.015B | Dense | 4 | 128 | — | — |
| OneRec-0.121B | Dense | 8 | 1024 | — | — |
| OneRec-0.935B | **MoE** | 8 | 1024 | **24 / 2** | Decoder |
| OneRec-2.633B | **MoE** | 24 | 1024 | **24 / 4** | Enc & Dec |

开源 OpenOneRec / OneRec-Foundation：

| 模型 | backbone | hidden | layers | Q heads | KV heads | FFN dim | FFN 类型 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OneRec-1.7B | Qwen3-1.7B | 2048 | 28 | 16 | **8** | 6144 | **dense** |
| OneRec-8B | Qwen3-8B | 4096 | 36 | 32 | **8** | 12288 | **dense** |

依据：

- HF `config.json` 的 `architectures` 是 `Qwen3ForCausalLM`，**没有** `num_experts` /
  `moe_intermediate_size` 等字段
- 官方模型卡原文：「The architecture remains **strictly consistent with the original Qwen3**」，
  唯一的结构改动是扩展 itemic token 的词表（`vocab_size = 176384`）
- Qwen3 的 0.6B / 1.7B / 4B / 8B 都是 dense；MoE 只有 30B-A3B 与 235B-A22B

### 两条推论（都会改变并行结论）

1. **EP 对 OpenOneRec 不适用** —— 没有专家可切。
   因此「beam × EP」那条线只能落在闭源 OneRec（24 专家）或 RankMixer 上。
2. **KV heads = 8，而一张卡 8 芯** ⇒ TP=8 恰好覆盖 KV，
   再加 KVP / CP 在**显存上没有增量收益**。
   这是 `05-CP.md` 判据 A 的一个真实数据点：
   `n_kv = 8 ≥ N = 8` ⇒ ② 的价值 ≈ 0。

## 并行含义

| 轴 | 结论 | 原因 |
| --- | --- | --- |
| **CP** | **有对象** | prefill 的 L 可以切；decode 的 q=B 让切 KV 也划算 |
| **EP** | **开源版不适用** | OpenOneRec 是 dense Qwen3，没有专家。闭源 OneRec 才有 |
| **beam** | **主战场** | B 从 128 到 900，是真正的缩放维 |
| TP | 可以，但 8 卡触顶 | `n_kv = 8`，TP>8 会复制 KV |

## 常见误解

| 误解 | 修正 |
| --- | --- |
| 「GR 就是小 LLM，并行直接用 LLM 的」 | q=512 vs 1；只有 3 步 vs 上千步；这两个差异改变全部结论 |
| 「beam search 只是个解码参数」 | 在 GR 里它是**主维度**，决定了成本形状 |
| 「3 步很短，所以 decode 不贵」 | decode 的 pair 数是 `B·(L+B·t)`，B=512 时是 L 的 512 倍 |
| 「输出只有 3 个 token ⇒ 等于一次前向」 | 是 3 次，而且每次有 B 条 beam |
| **「OneRec 是 MoE，所以能上 EP」** | **闭源 OneRec 才是；开源的 OpenOneRec 是 dense Qwen3，没有专家** |

## 对应代码

`models/onerec_min.py`

- RQ-KMeans：一个 item → 3 个 token，并显示相邻 item 共享前缀
- 形状轨迹：encoder 吃 L、decoder 吃 1、cross-attn 的 KV 长 L
- beam 的工作量表：`B·(L + B·t)`
- 朴素 beam 的 KV 冗余倍数（≈ beam 宽度）
- **`[G]` 一节：闭源 OneRec（MoE 24 专家）vs 开源 OpenOneRec（dense Qwen3）的对照**
