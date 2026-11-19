# 机制速查

七个 GR 模型（HSTU / MTGR / LONGER / HyFormer / TIGER / RankMixer / OpenOneRec）
× 四个并行轴（CP / EP / TP / beam）+ 混合并行的最小可运行实现、机制说明、交互式网页。

## 在线查看（手机可直接打开）

**<https://easebreaker.github.io/gr-parallel-cheatsheet/>**

窄屏下侧栏会收成左上角的 ☰ 抽屉，表格可横向滑动，手机上直接能读。

更新线上页面：

```bash
./deploy.sh          # 会自动重跑 build.py，再同步到 Pages 仓库
```

## 怎么用

**看网页（推荐）**

双击 `index.html`。自包含、零外部依赖、不需要联网。

- 左侧栏按「总览 / 模型 / 并行轴 / 交叉」分组
- 每个条目三个标签页：**说明 / 代码 / 运行输出**
- `交叉 → 模型 × 轴 矩阵`：点模型名进模型页，点格子进对应的轴
- 按 `/` 搜索，`Esc` 退出搜索

**跑代码**

```bash
python3 run_all.py              # 全部跑一遍，末尾打印总表
python3 run_all.py hstu         # 只跑一个
python3 models/longer_min.py    # 单个文件的完整输出
```

只依赖 `numpy`，不需要 torch，不需要多卡，不需要 NPU。
「分布式」用单进程的假 ProcessGroup 模拟：机制是真的，通信时间是估算的。

**改完源码后重新生成网页**

```bash
python3 build.py
```

`build.py` 会把 `.py` 和 `docs/*.md` 内联进 `index.html`，并真的跑一遍 demo
把 stdout 也内联进去 —— 所以网页里的「运行输出」永远是真实的。

> 注意：macOS 系统 Python + numpy 在 stdout 被重定向到文件时会 SIGSEGV（exit 139）。
> `build.py` 已经用 pty 绕开了；万一看到 `out of pty devices`，在普通终端里再跑一次即可。

## 目录

```
机制速查/
├── index.html                 交互式查看器（自包含）
├── build.py                   生成 index.html
├── run_all.py                 一次跑完所有 demo
├── _core.py                   shape/成本打印 + 假 ProcessGroup
├── models/
│   ├── hstu_min.py            SiLU 代替 softmax ⇒ 纯加归约
│   ├── mtgr_min.py            三段掩码 + GLN
│   ├── longer_min.py          token merge + 扁矩阵 + softmax
│   ├── hyformer_min.py        16 个 query 读整段
│   ├── tiger_min.py           RQ-VAE + 小 beam + 约束解码
│   ├── rankmixer_min.py       零参数 token mixing
│   └── onerec_min.py          SID + encoder-decoder + beam
├── parallel/
│   ├── cp_min.py              cp_q / cp_qk / zigzag 三方案实测
│   ├── ep_min.py              四步 + 三个量 + 路由相关性
│   ├── beam_min.py            naive vs tree attention vs 静态 KV
│   └── mixed_min.py           CP×EP×TP×beam 的组合规则
└── docs/
    ├── 00-地图.md             先读这个
    ├── 01-HSTU.md
    ├── 02-MTGR.md
    ├── 03-RankMixer.md
    ├── 04-OpenOneRec.md
    ├── 05-CP.md
    ├── 06-EP.md
    ├── 07-beam.md
    ├── 08-LONGER.md
    ├── 09-HyFormer.md
    ├── 10-TIGER.md
    └── 11-混合并行.md
```

## 三条判据（全篇的骨架）

任何「能不能并行」的问题，按顺序问三句：

1. **归约代数** —— 是不是 `Σ_j w_ij V_j` 这种纯加？纯加 ⇒ 分块求和精确，零归约。
2. **掩码拓扑** —— 切 N 块后失衡多少？方阵 ⇒ 1.00；因果 ⇒ `2 − 1/N`。
3. **单卡容量** —— KV 装得下吗？装得下 ⇒ 不需要切。

前两条说「能切」，第三条才说「要不要切」。大部分争论失败在把前两条当成了第三条。

## 一句话结论

| 模型 | 一句话 |
| --- | --- |
| TIGER | S=20 / B=10~20 ⇒ **四个轴全都没有对象**。它是「叫 GR 不等于需要并行」的反面基线 |
| HSTU | 无 softmax ⇒ 切 KV 也只需加法，但单卡装得下 ⇒ 不需要 |
| MTGR | 静态段是方阵 ⇒ 失衡 1.00，比纯因果更好切 |
| LONGER | **唯一用 softmax**；`q/L≈1e-2` ⇒ 收益上限只 1.2~1.3× |
| RankMixer | T 是特征子空间数 ⇒ 没有序列维 ⇒ CP 无定义；EP 是主战场 |
| HyFormer | q≈16 ⇒ **算便宜、读贵**，CP 切的是读（买到延迟，不是吞吐） |
| OneRec（闭源） | **MoE 24 专家** ⇒ EP 有入口，beam 让路由相关性放大 |
| OpenOneRec（开源） | q=B=512 ⇒ beam 是主战场；但权重是 **dense Qwen3**，没有专家 ⇒ EP 不适用 |

## 混合并行的一句话

**每一项成本只有唯一一个轴能切** ⇒ 混合加速是 `1/Σ(f_i/n_i)`，不是加速比的乘积。

- CP 切 prompt KV（`∝P`）
- beam 切 tree 自注意力（`∝B²·t`）
- TP 切 attention 权重
- EP 切 FFN / 专家
- fixed 与 scoring **没有轴**

细节在 `docs/11-混合并行.md`，代码在 `parallel/mixed_min.py`。
