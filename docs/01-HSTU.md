# 01 · HSTU

> **一句话**：把 softmax 换成 SiLU，于是 attention 变成一次普通的加法归约 ——
> 这一行改动同时决定了它的速度、它的 CP 可行性、以及它为什么不需要 CP。

## 论文

- Zhai et al., *Actions Speak Louder than Words: Trillion-Parameter Sequential
  Transducers for Generative Recommendations*, **ICML 2024**（Meta）
- 代码：`facebookresearch/generative-recommenders`（本仓库 `_vendor/` 下有）

## 结构

```
 用户行为序列 (item_id, action, timestamp)
        │
        ▼
   ┌─ 逐 token 嵌入 ─┐
   │  item emb       │
   │  action emb     │  ← 三类特征分别嵌入后相加
   │  time bucket    │
   └────────┬────────┘
            ▼
      ┌───────────────────────────────┐
      │  HSTU block  × L              │
      │                               │
      │   X ──► Q,K,V,U  (四组投影)   │
      │        │                      │
      │        │  score = QKᵀ / N     │  ← N = max_seq_len
      │        │  score += rab(t)     │  ← 时间差查表，O(N²) 张量
      │        │  A = SiLU(score)     │  ← 关键：不是 softmax
      │        │  out = A @ V         │  ← 纯加归约
      │        │  out *= SiLU(U)      │  ← 门控
      │        └─► + 残差 / LN        │
      └────────┬──────────────────────┘
               ▼
         逐 token 打分 (pointwise) 或 序列聚合 (non-pointwise)
```

## 四个机制

### 1. SiLU 代替 softmax（最重要）

```
LLM:     A = softmax(score)        A 的每一行和 = 1
HSTU:    A = SiLU(score)           A 的每一行和 = 任意

out_i = Σ_j A_ij · V_j
```

softmax 的行和恒为 1 ⇒ `out_i` 依赖**整个键集合**。少看一半键，结果的**尺度就错了**。

SiLU 没有归一化 ⇒ `A_ij` 与键集合无关 ⇒ 输出就是加权和的一部分，**可以分块算再相加**。

> 这是 CP 在 HSTU 上「零归约」的全部技术内容。`models/hstu_min.py` 的 `[验证 1]` 用一个
> `max_abs_err ≈ 1e-16` 的对比把它证明了。

### 2. average factor（除以 N，而不是除以 Σ）

论文把这个因子叫做 normalization，防止注意力权重随序列长度变大。它有两个实现陷阱：

- **必须在 SiLU 之前**：放到后面会让权重尺度随 L 漂移
- **必须只除一次**：在 CP 下如果每张卡都除一次 N，结果会被放大

### 3. rab：relative attention bias

时间戳差值分桶 → 查表 → 得到一个标量，加到 score 上。

- 参数只有 `num_buckets + 1` 个（**O(1)**）
- 前向张量是 `[B, N, N]`（**O(N²)**）

> Meta 的生产实现（`modules/stu.py:205`）里**没有 timestamps、没有 rab，只有窗口**。
> 也就是说：rab 是参考实现的特性，生产上被窗口替代了。
> 这条事实经常被误读成「rab 是结构性缺陷」。

### 4. u-v 门控

`out = (A @ V) ⊙ SiLU(U)`。U 是**第四组投影**，不是 V 的复用。
少一组投影不会报错，但会让模型少一个自由度 —— 这类 bug 只影响精度，不影响 shape。

## 成本

| 量 | 公式 | N=4096 时 |
| --- | --- | --- |
| 可见 pair | `N²`（无 mask）/ `N(N+1)/2`（因果） | 8.4 M |
| FLOPs | `2 · pairs · d · h` | 1.07 G |
| rab 张量 | `N² · 4` 字节 | 64 MiB |
| KV 字节 | `N · d · 2 · 2 · 层数` | 见判据 3 |

## 并行含义

| 轴 | 结论 | 原因 |
| --- | --- | --- |
| **CP** | **数学可行，但不需要** | 纯加归约 ⇒ 零归约；但 N≤32K 时 KV 单卡装得下 |
| TP | 可以 | 切 head / 切 d，通用 |
| EP | 只有 MoE 版本才有意义 | 原版 HSTU 是 dense |
| beam | 无关 | 判别式，没有自回归 |

**为什么「不需要」值得单独强调**：CP 在 LLM 上的全部难点来自「KV 必须分片」。
HSTU 的 KV 在 32K 以内单卡装得下 ⇒ 这个前提不成立 ⇒ 四个难点（online softmax、
负载配平、自定义归约原语、跨卡状态）全部消失，**一个都不剩**。

## 常见误解

| 误解 | 修正 |
| --- | --- |
| 「HSTU 是纯加归约」 | 对 HSTU 成立，但**不适用于 LONGER**（LONGER 用 softmax） |
| 「rab 必须物化，所以是结构性缺陷」 | Meta 生产路径用窗口绕开了 rab；是已解决的实现代价 |
| 「二次复杂度 ⇒ 一定要 CP」 | 二次是真的，但容量判据说不用切 |
| 「SiLU 替换 softmax 只是省了一点算力」 | 它改变的是**归约的代数性质**，不是速度 |

## 对应代码

`models/hstu_min.py`

- `[验证 1]` 分块求和 vs 整体，`max_abs_err ≈ 1e-16`
- `[验证 2]` 同一份 score 下 SiLU 行和 ≠ 1、softmax 行和 = 1
- `[验证 3]` pair 与 rab 张量随 N 的真实增长
