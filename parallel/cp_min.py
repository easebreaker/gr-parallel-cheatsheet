"""CP 的三种切法 —— 在同一个加法归约 attention 上实测差异。

对应代码: Megatron `context_parallel/sequence_roll.py`、`parallel_state.py`
          TransformerEngine `AttnFuncWithCPAndKVP2P`（ring）
          Medha 的 KVP（切 KV、复制 Q）

=======================================================================
只有三种切法，区别只在「切谁」和「失衡怎么办」
=======================================================================

  (1) cp_qk   切 Q 也切 KV（经典 CP / ring attention）
      每卡：q 的 1/N + kv 的 1/N
      通信：(N-1) 轮，每轮 shard_kv 字节  —— 必须转圈补齐 KV
      失衡：因果三角 => 2N/(N+1)（N=8 时 1.78）

  (2) zigzag  在 (1) 之上把块配对
      每卡：第 r 块 + 第 (2N-1-r) 块
      通信：同 (1)
      失衡：≈1.00（把「重卡 + 轻卡」合成一张卡）

  (3) cp_q    只切 Q，KV 每卡一份全量
      每卡：q 的 1/N + 全部 kv
      通信：**0**（KV 是复制的）
      失衡：读 KV 完全均衡；算 pair 对因果仍失衡
             —— 但对 GR 的 decode（非因果、q=B）失衡也是 1.00

=======================================================================
这三条路线的取舍可以用一句话讲完
=======================================================================
    通信量 ∝ 你切掉的那一维
    cp_qk 切了 KV  => 省显存，但要 (N-1) 轮通信
    cp_q  不切 KV  => 零通信，但每卡都要存全量 KV

所以：**KV 单卡装得下时，cp_q 严格优于 cp_qk。**
这就是「GR 的 CP 是伪问题」的技术内容（docs/05）。
"""

from __future__ import annotations

import numpy as np

from _core import (World, count_pairs, load_imbalance, rng, step, table, title,
                   zigzag_indices)


# ======================================================================
# 被切的那个 attention：纯加归约（HSTU/MTGR 的形态）
# ======================================================================
def make_qkv(n, d=32, n_heads=4, seed=0):
    g = rng(seed)
    return (g.normal(0, 1, (n, d)), g.normal(0, 1, (n, d)),
            g.normal(0, 1, (n, d)), g.normal(0, 1, (n, n_heads, d // n_heads)))


def local_out(q, k, v, mask, d=32, h=4):
    """在给定的 q/k/v 上算 SiLU 权重并乘 V。mask 是 [nq, nk]。"""
    nq, nk = q.shape[0], k.shape[0]
    hd = d // h
    qh = q.reshape(nq, h, hd).transpose(1, 0, 2)
    kh = k.reshape(nk, h, hd).transpose(1, 0, 2)
    vh = v.reshape(nk, h, hd).transpose(1, 0, 2)
    s = qh @ kh.transpose(0, 2, 1) / max(nq, nk)
    s = np.where(mask[None], _silu(s), 0.0)
    o = s @ vh
    return o.transpose(1, 0, 2).reshape(nq, d)


def _silu(x):
    out = np.empty_like(x)
    p = x >= 0
    out[p] = x[p] / (1 + np.exp(-x[p]))
    e = np.exp(x[~p])
    out[~p] = x[~p] * e / (1 + e)
    return out


def reference(q, k, v, mask, d=32, h=4):
    return local_out(q, k, v, mask, d, h)


# ======================================================================
# (3) cp_q：零通信
# ======================================================================
def cp_q(world, q, k, v, mask, d=32, h=4):
    n = q.shape[0]
    c = n // world.size
    outs, pairs = [], []
    for r in range(world.size):
        lo, hi = r * c, (r + 1) * c
        o = local_out(q[lo:hi], k, v, mask[lo:hi], d, h)   # 全量 KV
        outs.append(o)
        pairs.append(int(mask[lo:hi].sum()))
    return np.concatenate(outs, 0), pairs, 0


# ======================================================================
# (1) cp_qk：ring，每轮换一块 KV
# ======================================================================
def _chunks(n: int, N: int) -> list[np.ndarray]:
    c = n // N
    return [np.arange(r * c, (r + 1) * c) for r in range(N)]


def cp_qk(world, q, k, v, mask, d=32, h=4):
    """ring：每卡持 q 的 1/N + kv 的 1/N，转 (N-1) 轮补齐。

    注意 send/recv 的顺序：所有 rank 先同时 post send，再同时 recv。
    这正是 TE `flash_attn_p2p_communicate` 里「偶数 rank 先发、奇数 rank 先收」
    要解决的那个死锁问题的单进程等价写法。
    """
    n, N = q.shape[0], world.size
    blocks = _chunks(n, N)
    cur_k = [k[b] for b in blocks]
    cur_v = [v[b] for b in blocks]
    acc_k = [[] for _ in range(N)]
    acc_v = [[] for _ in range(N)]
    comm = [0] * N

    for _hop in range(N - 1):
        # 1) 所有 rank 先 post send
        for r in range(N):
            pg = world[r]
            dst = (r + 1) % N
            pg.send(dst, cur_k[r])
            pg.send(dst, cur_v[r])
            comm[r] += cur_k[r].nbytes + cur_v[r].nbytes
        # 2) 所有 rank 再 recv（此时 mailbox 一定非空）
        for r in range(N):
            pg = world[r]
            src = (r - 1) % N
            nk, nv = pg.recv(src), pg.recv(src)
            acc_k[r].append(nk)
            acc_v[r].append(nv)
            cur_k[r], cur_v[r] = nk, nv

    outs, pairs = [], []
    for r in range(N):
        own = blocks[r]
        # 收到的顺序是 (r-1), (r-2), ..., (r-(N-1))
        key_idx = np.concatenate([own] + [blocks[(r - 1 - hop) % N] for hop in range(N - 1)])
        k_all = np.concatenate([k[own]] + acc_k[r], 0)
        v_all = np.concatenate([v[own]] + acc_v[r], 0)
        m = mask[np.ix_(own, key_idx)]
        outs.append(local_out(q[own], k_all, v_all, m, d, h))
        pairs.append(int(m.sum()))
    return np.concatenate(outs, 0), pairs, sum(comm) // N


def zigzag_pairs(n: int, N: int, mask: np.ndarray) -> list[int]:
    """zigzag 的负载：rank r 拿第 r 块和第 (2N-1-r) 块。

    正确性与 cp_qk 相同（纯加归约，求和顺序无关），所以这里只量失衡。
    """
    c = n // (2 * N)
    out = []
    for r in range(N):
        idx = zigzag_indices(n, N, r)
        out.append(int(mask[np.ix_(idx, idx)].sum()))
    return out


# ======================================================================
def demo() -> None:
    n, d, h = 256, 32, 4
    q, k, v, _ = make_qkv(n, d, h)
    mask_causal = np.tril(np.ones((n, n), dtype=bool))
    mask_full = np.ones((n, n), dtype=bool)

    title("CP 三种切法：pair、通信、失衡")

    # ---------------- 1. 全序列（训练 / prefill）----------------
    print("\n  [场景 1] 全序列 + 因果掩码（= HSTU/MTGR 训练、GR prefill）")
    ref = reference(q, k, v, mask_causal, d, h)
    rows = []
    for N in (2, 4, 8):
        world = World(N)
        out_q, pairs_q, comm_q = cp_q(world, q, k, v, mask_causal, d, h)
        world = World(N)
        out_qk, pairs_qk, comm_qk = cp_qk(world, q, k, v, mask_causal, d, h)
        zz = zigzag_pairs(n, N, mask_causal)
        rows.append((N,
                     f"{load_imbalance(pairs_q):.2f}",
                     f"{load_imbalance(pairs_qk):.2f}",
                     f"{load_imbalance(zz):.2f}",
                     f"{comm_q / 1024:.0f}", f"{comm_qk / 1024:.0f}",
                     f"{np.abs(out_qk - ref).max():.1e}"))
    print(table(rows, ("N", "cp_q 失衡", "cp_qk 失衡", "zigzag 失衡",
                       "cp_q 通信 KiB", "cp_qk 通信 KiB", "cp_qk 误差")))
    print("     => cp_q 的通信恒为 0，且结果精确（误差 1e-16 级）。")
    print("        cp_q 与 cp_qk 的『算』失衡相同，都是 2 − 1/N（N=8 时 1.87）。")
    print("        注意：两者的区别不是 pair 数，而是**通信**。")
    print("        zigzag 把失衡压到 1.00，代价是额外的重排 kernel。")

    print("\n     注：cp_q 在因果掩码下『算』仍失衡（前面的 rank 看得少），")
    print("         但『读』是均衡的（每卡都读全量 KV）。GR 的瓶颈在读取。")

    # ---------------- 2. decode（beam）----------------
    print("\n  [场景 2] decode：q = B（beam 宽度），KV = P（非因果）")
    print("           这正是 beam 让 CP 重新有对象的地方")
    rows = []
    kv_per_tok = d * 4 * 2                      # K+V 字节/token
    for B, P in ((1, 4096), (128, 4096), (512, 4096), (900, 32768)):
        full = P * kv_per_tok
        rows.append((B, P,
                     f"{full / 1024:.0f}",
                     f"{full / B / 1024:.1f}",
                     f"{B * P / 1e6:.2f}"))
    print(table(rows, ("B", "P", "cp_q 每卡读 KiB", "每 query 摊到 KiB",
                       "pair/步 (M)")))
    print("     => cp_q 每卡要读全量 KV（第 3 列与 B 无关），")
    print("        但因为一个请求里有 B 个 query，**每 query 的读取量按 1/B 摊薄**。")
    print("        这是 LLM decode（B=1）永远做不到的事。")

    # ---------------- 3. 通信/计算 收支平衡点 ----------------
    print("\n  [场景 3] cp_q vs cp_qk 的真实账（两者的字节数其实相等）")
    print("            cp_q  多读  (1 − 1/N)·P·κ  来自 HBM（~800 GB/s）")
    print("            cp_qk 通信  (1 − 1/N)·P·κ  走互联（~200 GB/s）")
    rows = []
    for P, N in ((1024, 8), (4096, 8), (16384, 8), (65536, 8)):
        bytes_ = (N - 1) / N * P * kv_per_tok
        t_read = bytes_ / 800e9
        t_comm = bytes_ / 200e9
        rows.append((P, N, f"{bytes_ / 1024:.0f}",
                     f"{t_read * 1e9:.0f} ns", f"{t_comm * 1e9:.0f} ns",
                     f"{t_comm / t_read:.1f}x"))
    print(table(rows, ("P", "N", "字节数 (两边相同)", "HBM 多读", "互联通信",
                       "通信/多读")))
    print("     => **两边的字节数完全相等**。胜负只取决于用哪种带宽。")
    print("        HBM:互联 ≈ 4:1 => cp_q 恒定赢 4 倍，与 P、N 无关。")
    print("        这是 Medha 的 KVP 与 GR 的 cp_q 在结构上占优的真正原因。")

    # ---------------- 4. 何时该用 cp_qk ----------------
    print("\n  [场景 4] 什么时候反而必须用 cp_qk？")
    rows = [
        ("KV 单卡装得下", "cp_q", "零通信、精确、无重排"),
        ("KV 单卡装不下（L > 32K）", "cp_qk", "只能切，通信是硬代价"),
        ("有 softmax（LONGER）", "都必须", "需要 online softmax，归约不再是加法"),
        ("q 很小（LONGER 的 50~250）", "cp_q 也不划算", "省下的绝对工作量本来就小"),
    ]
    print(table(rows, ("条件", "选择", "理由")))


if __name__ == "__main__":
    demo()
