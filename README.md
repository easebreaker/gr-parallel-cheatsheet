# GR 模型 × 并行轴 机制速查

七个生成式推荐（GR）模型 × 四个并行轴（CP / EP / TP / beam），
每个都配一份**能跑起来的最小实现**和对应论文的机制说明。

在线查看：<https://easebreaker.github.io/gr-parallel-cheatsheet/>

## 模型

| 模型 | 论文 | 一句话 |
| --- | --- | --- |
| HSTU | ICML'24 | SiLU 代替 softmax ⇒ 纯加归约 |
| MTGR | CIKM'25 | 三段掩码 ⇒ 静态段是方阵 |
| LONGER | RecSys'25 | 扁矩阵 + softmax |
| HyFormer | SIGIR'26 | 16 个 query 读整段 |
| TIGER | NeurIPS'23 | SID + 小 beam，三个轴都没有对象 |
| RankMixer | CIKM'25 | 转置代替 attention，没有 L 维 |
| OpenOneRec | arXiv 2512.24762 | dense Qwen3 + beam |

## 并行轴

| 轴 | 切什么 | 通信量 |
| --- | --- | --- |
| CP | 序列 / prompt KV | ring `2(N−1)Pd`；cp_q 只搬 `B·d` 的 LSE |
| EP | 专家 | `2·B·k·d` 两次 all-to-all（是 TP 的 k 倍） |
| TP | 权重 | `2·B·d` all-reduce |
| beam | 搜索宽度 / tree KV | 全局 top-B，MB 级 |

## 三条判据

1. **归约代数** —— 是 `Σ_j w_ij V_j` 纯加吗？纯加 ⇒ 分块求和精确，零归约。
2. **掩码拓扑** —— 切 N 块后失衡多少？方阵 ⇒ 1.00；因果 ⇒ `2 − 1/N`。
3. **单卡容量** —— KV 装得下吗？装得下 ⇒ 不需要切。

前两条说「能切」，第三条才说「要不要切」。

## 本地运行

```bash
python3 run_all.py        # 全部 demo 跑一遍
python3 build.py          # 重新生成 index.html
```

只依赖 numpy。

## 更新线上页面

在本目录执行：

```bash
./deploy.sh
```
