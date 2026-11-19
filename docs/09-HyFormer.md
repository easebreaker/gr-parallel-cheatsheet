# 09 · HyFormer

> **一句话**：用 **16 个 query** 去读整段历史。
> 于是「算」变得很便宜，「读」变成唯一瓶颈 —— 这是 **LLM decode 的形状被搬进了推荐**。

## 论文

- *HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction
  in CTR Prediction*, **arXiv 2601.12681**（SIGIR'26）

## 结构

```
 非序列特征 (M 个)                 长行为序列 (L)
      │                                 │
      │   MLP 生成 N 个 semantic query   │
      ▼                                 ▼
  Q ∈ [N, d]  ────────► ① Query Decoding ◄──── K,V ∈ [L, d]
                              CrossAttn(N, L)
                                 │
                                 ▼  [N, d]
                     ┌───────────┴───────────┐
                     │ ② Query Boosting      │
                     │  Q' = [Q | NS tokens] │  T = N + M
                     │  ① MLP-Mixer 式转置    │  ← 零参数、零 FLOPs
                     │  ② per-token FFN       │  ← 每个 token 一套参数
                     └───────────┬───────────┘
                                 ▼
                          下一层（stacked）
```

**两个模块交替，一层就是一次「读序列 → 混特征」。**

## 三个机制

### 1. Query Decoding：`q` 极小

`N ≈ 16`，`L` 可达上万 ⇒ **`q/L ≈ 1e-3`**，是七个模型里最小的。

代价 `∝ N·L·d`，看起来「很便宜」—— 但它便宜的原因是**把工作搬给了读取**：

```
算的部分  ∝ N · L · d      （N 小 → 小）
读的部分  ∝ L · d · bytes  （L 大 → 带宽主导）
```

**闭式结论**（`models/hyformer_min.py` 验证）：

```
t_算 / t_读 = N · BW / (2 · F)      与 L、d 都无关
```

N=16、BW=800GB/s、F=150TFLOPS ⇒ 比值 **0.043**，即读是算的 23 倍。

| L | pair | 计算 | KV 读取 | 读耗时 | 读/算 |
| --- | --- | --- | --- | --- | --- |
| 512 | 8 K | 0.01 us | 0.1 MiB | 0.12 us | 23× |
| 4096 | 66 K | 0.04 us | 0.8 MiB | 0.98 us | 23× |
| 65536 | 1049 K | 0.67 us | 12.0 MiB | 15.73 us | 23× |

（单层单样本；乘上 层数×batch 才是整网，但比值不变。）

### 2. N 越多并不更慢（在读主导区间）

| N | pair @ L=16K | 计算 | 读 | 算/读 |
| --- | --- | --- | --- | --- |
| 8 | 131 K | 0.08 us | 3.93 us | 0.02 |
| 16 | 262 K | 0.17 us | 3.93 us | 0.04 |
| 128 | 2097 K | 1.34 us | 3.93 us | 0.34 |
| 512 | 8389 K | 5.37 us | 3.93 us | 1.37 |

N 涨 64 倍，总时间只涨约 1.34 倍。

> **这与 LONGER 恰好相反**：那边减少 `q`（250→100）可以省 46% FLOPs。
> HyFormer 的 `q` 优化是白费力气，因为瓶颈在读取。

### 3. Query Boosting = RankMixer 的两个部件

- **MLP-Mixer 式 token mixing**：零参数转置（与 RankMixer 的 MHTM 完全相同）
- **per-token FFN**：每个 token 一套参数（参数量 ×T，FLOPs 不变）

论文明确写了 "inspired by RankMixer (Zhu et al., 2025)"。
所以 `03-RankMixer.md` 里的机制分析可以直接搬过来。

## 一个容易混的点：两段交叉注意力

| 阶段 | Q | KV | 代价 | 扫描时动谁 |
| --- | --- | --- | --- | --- |
| `decode` | short_len (~16) | short_len (~16) | 两段都小，几乎不计 | Q 与 KV 一起变 |
| `seq_cross` | short_len (~16) | kv_len (可达 8K) | `∝ short_len · kv_len` | 只有 KV 变 |

**这个差异是「辨识」的关键**：扫 `short_len` 时两段一起动，扫 `kv_len` 时只有
`seq_cross` 动。自变量不成比例，才有独立信息（对应 `执行时间预测` 里的辨识分析）。

## 并行含义

| 轴 | 结论 | 原因 |
| --- | --- | --- |
| **CP** | **真的有用，但只是延迟** | `q/L≈1e-3` ⇒ 算的少、读的多 ⇒ 切 KV 是唯一出路 |
| TP | 有用 | 切 d，同时减轻权重与 KV 带宽 |
| EP | 仅当 MoE | 原论文是 dense LRM，没有专家 |
| beam | 不适用 | 判别式，没有自回归 decode |

### CP 在这里买到的是什么

切 KV 之后（C=2/4/8，P=32K）：

| C | 计算 us | 读 us | 合并 us | 合计 us |
| --- | --- | --- | --- | --- |
| 1 | 0.34 | 7.86 | 0.02 | 8.22 |
| 2 | 0.17 | 3.93 | 0.02 | 4.12 |
| 4 | 0.08 | 1.97 | 0.02 | 2.07 |
| 8 | 0.04 | 0.98 | 0.02 | 1.04 |

时间确实按 `1/C` 降，但**降的是读取，不是算力**。

> **它买到的是 TPOT 上界，不是吞吐。**
> 这与 LONGER 的 1.2~1.3× Amdahl 是**两种不同的收益**，不能混着说。

这正是 Medha 的 KVP 的适用场景 —— 也说明 **HyFormer 是七个模型里唯一
「CP 有真实正面价值」的判别式模型**（HSTU/MTGR 的容量判据说不需要）。

## 常见误解

| 误解 | 修正 |
| --- | --- |
| 「HyFormer 的 attention 便宜，所以不需要并行」 | 便宜的是**算**；读是瓶颈，CP 正是切读 |
| 「q 小 ⇒ 和 LONGER 一样没法 CP」 | 不一样。LONGER 是算主导（q·L），HyFormer 是读主导 |
| 「Query Boosting 是新机制」 | 就是 RankMixer 的 MHTM + per-token FFN |
| 「两段 cross-attn 可以合并算」 | KV 长度不同（short_len vs kv_len），必须分开 |

## 对应代码

`models/hyformer_min.py`

- `QueryDecoding` / `MultiHeadTokenMixing` / `PerTokenFFN` / `QueryBoosting` / `HyFormerLayer`
- 算 vs 读的比值闭式（`N·BW/(2F)`）与实测表
- N 的敏感性（64 倍 N 只涨 1.34 倍）
- CP 的 `1/C` 曲线（降的是读取）
- 两段 cross-attn 的对照
