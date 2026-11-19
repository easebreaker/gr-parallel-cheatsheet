# 03 · RankMixer

> **一句话**：它把 quadratic self-attention 换成了**一次转置**。
> 代价为零，收益是 MFU 从 4.5% 涨到 45% —— 同时这也意味着 CP 没有定义。

## 论文

- Zhu et al., *RankMixer: Scaling Up Ranking Models in Industrial Recommenders*,
  **arXiv 2507.15551**（CIKM'25，字节）
- 关键数字：MFU **4.5% → 45%**，参数量放大两个数量级而延迟基本不变，
  1B dense 全量上线（+0.3% 用户活跃天数、+1.08% 使用时长）

## 结构

```
 拼接后的特征向量 e_input  (N 个特征组)
        │  按语义切成 T 个 token
        ▼
   x_0  x_1  ...  x_{T-1}           每个 T×D
        │
   ┌────┴─────────────────────────────────────┐
   │  Multi-Head Token Mixing（零参数）        │
   │                                          │
   │  每个 token 沿 D 切成 H 份 (H = T)        │
   │  把所有 token 的第 h 份拼成一个新 token    │
   │      s_h = [x_0.h | x_1.h | ... | x_{T-1}.h] │
   │  => reshape + transpose，没有矩阵乘        │
   └────┬─────────────────────────────────────┘
        ▼
   s_0  s_1  ...  s_{H-1}
        │
   ┌────┴─────────────────────────────────────┐
   │  Per-token FFN（每个 token 独立参数）      │
   │  参数 = 共享 FFN × T，FLOPs 不变           │
   │  ↓ 可替换为 Sparse-MoE（ReLU 路由 + L1）   │
   └────┬─────────────────────────────────────┘
        ▼
      残差 → 下一层 ... → mean pooling → 多任务头
```

## 两个机制

### 1. Multi-head Token Mixing：转置即混合

为什么它能替代 attention：**attention 的作用是让不同 token 的特征互相看见**。
转置也做到了这件事，只是方式不同：

| | attention | token mixing |
| --- | --- | --- |
| 做什么 | 每个 token 用**内积**加权聚合别的 token | 每个 token 的**特征块**被搬到另一个 token 里 |
| 参数 | Q/K/V 三组 | **0** |
| FLOPs | `2·T²·D` | **0**（只有数据搬运） |
| 前提 | token 在同一嵌入空间 | 无 |

论文的论证很硬：推荐的特征空间**异构**（用户 ID 空间上亿，行为序列稠密），
跨空间的**内积没有意义**。所以 attention 在这里不但贵，而且不比免费的转置更好。

小例子（T=3, D=6, H=3）：

```
输入                             输出
token0: [0 1 | 2 3 | 4 5]        s0: [0 1 | 6 7 | 12 13]   ← 所有 token 的第 0 份
token1: [6 7 | 8 9 | 10 11]      s1: [2 3 | 8 9 | 14 15]   ← 所有 token 的第 1 份
token2: [12 13|14 15|16 17]      s2: [4 5 |10 11| 16 17]   ← 所有 token 的第 2 份
```

`H = T` 是为了让 token 数不变，残差才能直接相加。

### 2. Per-token FFN：参数隔离

每个 token 有**自己的一套** FFN 权重。

- 参数量 = 共享 FFN × T（T=16 就是 16 倍）
- FLOPs **完全不变**（每个 token 仍然只做一次矩阵乘）

所以它是「加参数不加算力」的纯容量手段。论文的动机是
**特征子空间支配问题**：共享 FFN 会让高频特征淹没长尾特征。

### 3. Sparse-MoE 变体

把每个 token 的 dense FFN 换成 MoE：

- **ReLU 门 + L1 惩罚**，不是 top-k softmax
- 每个 token 激活的专家数**不同**（高信息量的 token 自己多要几个）

## 成本

| 量 | 公式 | T=16, D=64, k=4 |
| --- | --- | --- |
| token mixing | `0` | 0 |
| Per-token FFN | `2·T·D·(kD)·2` | 1.05 M FLOPs |
| 参数 | `T · D · kD · 2` | 0.52 M |
| 可见 pair | **不存在** | 0 |

## 并行含义

| 轴 | 结论 | 原因 |
| --- | --- | --- |
| **CP** | **无定义** | 没有序列维 L。T 是特征子空间数（~16），不是长度 |
| TP | 可以，但 T 太小 | 切 D 有效；切 T 会打破 per-token FFN 的意义 |
| **EP** | **主战场** | Sparse-MoE 的专家数才是参数量所在 |
| beam | 无关 | 判别式，没有自回归 |

### 这条为什么最值得记住

用户历史 L 从 512 涨到 8192，这个模块的：

- FLOPs：**不变**
- 激活字节：**不变**
- pair 数：**恒为 0**

**四列里三列与 L 无关。** 所以「按序列切（CP）」在 RankMixer 上没有定义 ——
不是「不需要」，是**没有那个维度**。

### MFU 从 4.5% 到 45% 的含义

这 10 倍不是「优化」，是**换了瓶颈**：

- 之前：一堆手工特征交叉（CPU 时代遗产），memory-bound，MFU 4.5%
- 之后：干净的大 GEMM，compute-bound，MFU 45%

**从 memory-bound 变成 compute-bound，正好是 TP/EP 开始有效的条件。**

## 常见误解

| 误解 | 修正 |
| --- | --- |
| 「RankMixer 也有 T 个 token，所以也能切序列」 | T 是特征维度，与用户历史 L 无关 |
| 「token mixing 是一种 attention」 | 它是转置。零参数零 FLOPs，没有内积 |
| 「per-token FFN 是 MoE」 | 不是。每个 token 看**不同**输入，MoE 专家看**同一**输入 |
| 「MFU 提升意味着模型变快」 | 它意味着硬件利用率变高；绝对延迟不一定降 |

## 对应代码

`models/rankmixer_min.py`

- token mixing 的小例子（打印 head 的去向）+ `inverse(forward(x)) == x`
- 三种混合方式的成本对比（MHTM 恒为 0）
- 「L 变化但本模块工作量不变」的四列表
- ReLU 路由：每个 token 激活数不同 + L1 惩罚
