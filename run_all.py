"""一次跑完所有 demo，最后打印一张「模型 × 并行轴」总表。

用法：
    cd 机制速查 && python3 run_all.py
    python3 run_all.py hstu            # 只跑一个
    python3 run_all.py --quiet         # 只打印总表
"""

from __future__ import annotations

import importlib
import io
import os
import sys
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "models"))
sys.path.insert(0, os.path.join(HERE, "parallel"))

TARGETS = [
    ("tiger", "models.tiger_min"),
    ("hstu", "models.hstu_min"),
    ("mtgr", "models.mtgr_min"),
    ("longer", "models.longer_min"),
    ("rankmixer", "models.rankmixer_min"),
    ("hyformer", "models.hyformer_min"),
    ("onerec", "models.onerec_min"),
    ("cp", "parallel.cp_min"),
    ("ep", "parallel.ep_min"),
    ("beam", "parallel.beam_min"),
    ("mixed", "parallel.mixed_min"),
]

MATRIX = [
    ("TIGER",      "无对象",     "不适用",   "无对象",   "S=20 / B=10~20 ⇒ 全是小量"),
    ("HSTU",       "能切但不需要", "仅 MoE",   "无关",     "纯加归约，但 KV 装得下"),
    ("MTGR",       "有对象",     "有条件",   "无关",     "静态段方阵 ⇒ 失衡 1.00"),
    ("LONGER",     "对象小",     "未用",     "无关",     "q/L≈1e-2 ⇒ 上限 1.2~1.3x"),
    ("RankMixer",  "无定义",     "主战场",   "无关",     "T 是特征子空间数，没有 L"),
    ("HyFormer",   "买延迟",     "仅 MoE",   "不适用",   "q≈16，算便宜读贵"),
    ("OpenOneRec", "有对象",     "闭源才有", "**主战场**", "q=B=512"),
]

AXES = ("模型", "CP", "EP", "beam", "一句话")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    quiet = "--quiet" in sys.argv
    targets = [t for t in TARGETS if not args or t[0] in args]

    for name, modpath in targets:
        mod = importlib.import_module(modpath)
        if quiet:
            buf = io.StringIO()
            with redirect_stdout(buf):
                mod.demo()
        else:
            mod.demo()

    import _core
    _core.title("总表：模型 × 并行轴")
    print(_core.table(MATRIX, AXES))
    print("\n  *  TIGER 是反面基线：四个轴全无对象 —— 「叫 GR」不等于「需要并行」。")
    print("  *  HSTU 的 CP 数学可行，但 N ≤ 32K 时 KV 单卡装得下 ⇒ 不需要。")
    print("  ** HSTU/MTGR/LONGER/HyFormer/RankMixer 是判别式，没有 beam 这一维。")
    print("  *** OpenOneRec（开源）是 dense Qwen3，没有专家；只有闭源 OneRec 是 MoE。")
    print("\n  读法：先看「一句话」列，再去 docs/ 里找对应的结构原因。")
    print("  网页版：直接双击 index.html（分组导航 + 搜索 + 7×4 交叉矩阵）。")


if __name__ == "__main__":
    main()
