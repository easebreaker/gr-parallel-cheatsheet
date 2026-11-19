"""所有 demo 共用的极小数。

只做三件事：
1. 打印 shape / 成本轨迹（并行性问题的全部信息都在这里面）
2. 一个假的 ProcessGroup，把「跨卡」变成「跨 list」
3. 计数：pair 数、字节数

刻意不依赖 torch.distributed —— 本文件的目标是「能在任何一台机器上跑」，
所以 collective 全部用直接搬运模拟。机制是真的，通信时间是估算的。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np

BAR = "=" * 78


def title(text: str) -> None:
    print(f"\n{BAR}\n{text}\n{BAR}")


def step(name: str, **info) -> None:
    """打一行轨迹。所有 demo 都用它，保证输出格式一致。"""
    body = "  ".join(f"{k}={v}" for k, v in info.items())
    print(f"  {name:<28s} {body}")


def dtype_bytes(dtype) -> int:
    return np.dtype(dtype).itemsize


# --------------------------------------------------------------------------
# 假的 ProcessGroup
# --------------------------------------------------------------------------
@dataclass
class FakePG:
    """模拟一个 process group。

    真实的 isend/irecv 在这里变成「往 (src, dst) 的队列里塞一个数组」。
    mailbox 由 World 持有并被所有 rank 共享 —— 这就是为什么 send 之后
    对端能 recv 到：单进程里没有真的网络，只有一张表。
    """

    size: int
    rank: int = 0
    mailbox: dict = field(default_factory=dict)

    def send(self, dst: int, tensor: np.ndarray) -> None:
        self.mailbox.setdefault((self.rank, dst), []).append(np.array(tensor, copy=True))

    def recv(self, src: int) -> np.ndarray:
        q = self.mailbox.setdefault((src, self.rank), [])
        if not q:
            raise RuntimeError(
                f"rank {self.rank} 从 {src} recv 时 mailbox 为空（send/recv 顺序有误）"
            )
        return q.pop(0)

    def all_to_all(self, split: Sequence[np.ndarray]) -> List[np.ndarray]:
        """split[i] 是发给 rank i 的分片；返回收到的东西。"""
        for dst, shard in enumerate(split):
            if shard is not None and shard.size:
                self.send(dst, shard)
        return [self.recv(i) for i in range(self.size)]

    def all_reduce(self, tensor: np.ndarray) -> np.ndarray:
        for dst in range(self.size):
            if dst != self.rank:
                self.send(dst, tensor)
        acc = np.array(tensor, copy=True)
        for src in range(self.size):
            if src != self.rank:
                acc = acc + self.recv(src)
        return acc

    def all_gather(self, tensor: np.ndarray) -> np.ndarray:
        for dst in range(self.size):
            if dst != self.rank:
                self.send(dst, tensor)
        parts = []
        for src in range(self.size):
            parts.append(tensor if src == self.rank else self.recv(src))
        return np.concatenate(parts, axis=0)


class World:
    """把 N 个 rank 装进一个进程，方便写「看起来像分布式」的代码。"""

    def __init__(self, size: int):
        self.size = size
        self.mailbox: dict = {}                       # 所有 rank 共享
        self.pgs = [FakePG(size=size, rank=r, mailbox=self.mailbox) for r in range(size)]

    def __getitem__(self, rank: int) -> FakePG:
        return self.pgs[rank]


# --------------------------------------------------------------------------
# CP / beam 用的两个小工具
# --------------------------------------------------------------------------
def contiguous_slice(L: int, n: int, r: int) -> slice:
    c = L // n
    return slice(r * c, (r + 1) * c)


def zigzag_indices(L: int, n: int, r: int) -> np.ndarray:
    """MCore / TE 的 zigzag 切分：rank r 拿第 r 块和第 (2n-1-r) 块。"""
    c = L // (2 * n)
    front = np.arange(r * c, (r + 1) * c)
    back = np.arange((2 * n - 1 - r) * c, (2 * n - r) * c)
    return np.concatenate([front, back])


def load_imbalance(work: Sequence[float]) -> float:
    return max(work) / (sum(work) / len(work))


def gib(x: float) -> str:
    return f"{x / 1024 ** 3:.3f} GB"


def mib(x: float) -> str:
    return f"{x / 1024 ** 2:.2f} MiB"


def table(rows: List[Sequence], headers: Sequence[str]) -> str:
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    out = [line, "  ".join("-" * w for w in widths)]
    for r in rows:
        out.append("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
    return "\n".join(out)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def silu(x: np.ndarray) -> np.ndarray:
    """数值稳定的 SiLU：x·sigmoid(x)，用两段式避免 exp 溢出。"""
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = x[pos] / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = x[~pos] * ex / (1.0 + ex)
    return out


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def count_pairs(mask: np.ndarray) -> int:
    """可见 (query, key) 对数。CP 的全部工作量都由它决定。"""
    return int(mask.sum())


def flops_attn(pairs: int, head_dim: int, n_heads: int) -> float:
    """2 * H * d 每个 pair：QK^T 一次 + attn@V 一次。"""
    return 2.0 * pairs * head_dim * n_heads
