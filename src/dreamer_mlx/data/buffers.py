"""A generic sequence replay buffer for MLX.

Stores arbitrary pytrees (nested dicts / lists / tuples) of arrays: images, vectors,
scalars, discrete actions, flags, ... Each leaf gets its own preallocated ring buffer
with its native shape and dtype (e.g. ``uint8`` images stay ``uint8``).

Storage is NumPy on the host rather than ``mx.array``: MLX arrays are immutable, so
in-place writes would rebuild the buffer or grow a lazy graph on every ``add``. On
Apple silicon memory is unified, so converting a sampled batch to ``mx.array`` is cheap.
Sampling returns pytrees of ``mx.array``.

Conventions:
- ``add`` takes a single step (no batch dim); ``add_batch`` takes ``n`` consecutive
  steps stacked on axis 0.
- ``sample(B)`` returns leaves of shape ``(B, *shape)``.
- ``sample_sequences(B, L)`` returns leaves of shape ``(B, L, *shape)``, made of
  ``L`` consecutive steps that never straddle the ring-buffer write head.
- If ``terminal_key`` is given, sequences also never cross an episode boundary: a
  terminal step may only appear as the *last* step of a sequence.
- One buffer holds one stream. For several parallel environments, use one buffer
  per environment.
"""

from typing import Any

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten


class SequenceReplayBuffer:
    """Ring-buffer replay memory over arbitrary pytrees, with sequence sampling.

    Args:
        capacity: Maximum number of steps stored. Oldest steps are overwritten first.
        example: A single step (no batch dimension) used to infer the structure,
            shapes and dtypes. Leaves may be NumPy/MLX arrays or Python/NumPy scalars.
            ``float64`` leaves are stored as ``float32`` (MLX GPU has no float64).
        terminal_key: Optional path of a leaf marking episode ends (e.g. ``"done"`` or
            ``"info.done"`` for nested dicts). Must hold a single value per step.
        seed: Seed for the sampling RNG (NumPy ``Generator``).
    """

    def __init__(
        self,
        capacity: int,
        example: Any,
        *,
        terminal_key: str | None = None,
        seed: int | None = None,
    ):
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self.capacity = capacity
        self.terminal_key = terminal_key
        self._rng = np.random.default_rng(seed)
        self._ptr = 0
        self._size = 0

        self._store: dict[str, np.ndarray] = {}
        for path, leaf in tree_flatten(example):
            a = np.asarray(leaf)
            dtype = np.float32 if a.dtype == np.float64 else a.dtype
            self._store[path] = np.zeros((capacity, *a.shape), dtype=dtype)
        if not self._store:
            raise ValueError("example contains no array leaves.")

        if terminal_key is not None:
            if terminal_key not in self._store:
                raise KeyError(
                    f"terminal_key {terminal_key!r} not in leaves {list(self._store)}."
                )
            if self._store[terminal_key][0].size != 1:
                raise ValueError("terminal_key leaf must hold a single value per step.")

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size == self.capacity

    @property
    def _start(self) -> int:
        """Physical index of the oldest step."""
        return self._ptr if self.is_full else 0

    def clear(self) -> None:
        self._ptr = 0
        self._size = 0

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #
    def add(self, step: Any) -> None:
        """Add a single step (leaves have no batch dimension)."""
        batched = tree_unflatten(
            [(p, np.asarray(x)[None]) for p, x in tree_flatten(step)]
        )
        self.add_batch(batched)

    def add_batch(self, steps: Any) -> None:
        """Add ``n`` consecutive steps, stacked on axis 0 of every leaf."""
        flat = {p: np.asarray(x) for p, x in tree_flatten(steps)}
        if flat.keys() != self._store.keys():
            raise ValueError(
                f"Structure mismatch: got {sorted(flat)}, expected {sorted(self._store)}."
            )

        n = None
        for path, a in flat.items():
            store = self._store[path]
            if a.ndim == 0 or a.shape[1:] != store.shape[1:]:
                raise ValueError(
                    f"Leaf {path!r}: expected shape (n, *{store.shape[1:]}), "
                    f"got {a.shape}."
                )
            if not np.can_cast(a.dtype, store.dtype, casting="same_kind"):
                raise TypeError(
                    f"Leaf {path!r}: cannot store dtype {a.dtype} as {store.dtype}."
                )
            if n is not None and a.shape[0] != n:
                raise ValueError("Leaves have different lengths along axis 0.")
            n = a.shape[0]
        if not n:
            return

        # If more steps than capacity are supplied, only the newest ``capacity`` survive.
        skip = max(0, n - self.capacity)
        idx = (self._ptr + skip + np.arange(n - skip)) % self.capacity
        for path, a in flat.items():
            self._store[path][idx] = a[skip:]
        self._ptr = (self._ptr + n) % self.capacity
        self._size = min(self._size + n, self.capacity)

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #
    def sample(self, batch_size: int) -> Any:
        """Uniformly sample individual steps (with replacement)."""
        self._require(1)
        idx = self._rng.integers(0, self._size, size=batch_size)
        return self._gather(idx)

    def sample_sequences(self, batch_size: int, seq_len: int) -> Any:
        """Uniformly sample ``batch_size`` windows of ``seq_len`` consecutive steps.

        Returns leaves of shape ``(batch_size, seq_len, *shape)``, in chronological
        order along axis 1.
        """
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1.")
        self._require(seq_len)

        n_starts = self._size - seq_len + 1
        if self.terminal_key is None:
            starts = self._rng.integers(0, n_starts, size=batch_size)
        else:
            valid = self._valid_starts(seq_len)
            if valid.size == 0:
                raise ValueError(
                    f"No sequence of length {seq_len} without an internal episode "
                    "boundary is available yet."
                )
            starts = self._rng.choice(valid, size=batch_size)

        logical = starts[:, None] + np.arange(seq_len)  # (B, L), chronological
        return self._gather((self._start + logical) % self.capacity, physical=True)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        """Save contents to a ``.npz`` file."""
        np.savez(path, __ptr__=self._ptr, __size__=self._size, **self._store)

    def load(self, path: str) -> None:
        """Load contents saved by ``save`` into this (identically configured) buffer."""
        with np.load(path) as f:
            for p, store in self._store.items():
                if p not in f or f[p].shape != store.shape:
                    raise ValueError(f"Saved buffer is incompatible at leaf {p!r}.")
                store[:] = f[p]
            self._ptr = int(f["__ptr__"])
            self._size = int(f["__size__"])

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _require(self, n: int) -> None:
        if self._size < n:
            raise ValueError(f"Need at least {n} stored steps, have {self._size}.")

    def _gather(self, idx: np.ndarray, physical: bool = False) -> Any:
        if not physical:  # idx is logical (chronological); the start offset is 0 or ptr
            idx = (self._start + idx) % self.capacity
        return tree_unflatten([(p, mx.array(a[idx])) for p, a in self._store.items()])

    def _valid_starts(self, seq_len: int) -> np.ndarray:
        """Logical start indices whose window has no terminal before its last step."""
        order = (self._start + np.arange(self._size)) % self.capacity
        done = self._store[self.terminal_key][order].reshape(self._size, -1).any(-1)
        c = np.concatenate([[0], np.cumsum(done)])  # c[k] = number of dones in done[:k]
        # window [s, s+L-1]: dones in its first L-1 steps = c[s+L-1] - c[s]
        n_starts = self._size - seq_len + 1
        return np.flatnonzero(c[seq_len - 1 : self._size] - c[:n_starts] == 0)


if __name__ == "__main__":
    example = {
        "obs": {"image": np.zeros((64, 64, 3), np.uint8),
        "state": np.zeros(4, np.float32)},
        "action": np.int32(0),
        "reward": np.float32(0),
        "done": np.bool_(False),
        "t": np.int32(0),
    }
    buf = SequenceReplayBuffer(capacity=1000, example=example, terminal_key="done", seed=0)

    rng = np.random.default_rng(1)
    for t in range(1500):  # overfill so the ring wraps; an episode ends every 50 steps
        buf.add(
            {
                "obs": {
                    "image": rng.integers(0, 256, (64, 64, 3), dtype=np.uint8),
                    "state": rng.normal(size=4).astype(np.float32),
                },
                "action": np.int32(t % 3),
                "reward": np.float32(0.1),
                "done": np.bool_(t % 50 == 49),
                "t": np.int32(t),
            }
        )

    print(len(buf), buf.is_full)
    batch = buf.sample_sequences(batch_size=8, seq_len=16)
    print({k: v.shape for k, v in tree_flatten(batch)})
    print(batch["obs"]["image"].dtype)

    # Sequences must be consecutive in time and never cross an episode boundary.
    for _ in range(200):
        b = buf.sample_sequences(32, 16)
        t, done = np.array(b["t"]), np.array(b["done"])
        assert (np.diff(t, axis=1) == 1).all()
        assert not done[:, :-1].any()
        assert t.min() >= 500  # only the newest 1000 steps survive
    print("ok")
