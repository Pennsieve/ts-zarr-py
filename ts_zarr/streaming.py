"""Bounded-memory streaming generators that drive the folds over a source."""

from collections.abc import Callable, Iterable, Iterator
from typing import Protocol

import numpy as np
import numpy.typing as npt

from ts_zarr.constants import DECIMATION_FACTOR
from ts_zarr.fold import fold_raw_block
from ts_zarr.protocols import ContinuousChannelSource


class BlockReadableArray(Protocol):
    """Minimal structural view of an on-disk array iterated along axis 0.

    Decouples streaming from zarr: any object exposing a length-bearing shape
    and axis-0 slicing (a zarr Array, a numpy array) satisfies this.
    """

    @property
    def shape(self) -> tuple[int, ...]:
        """Array shape; axis 0 is the iterated sample/bin axis."""
        ...

    def __getitem__(self, item: slice) -> npt.NDArray[np.float32]:
        """Return the rows in the given axis-0 slice."""
        ...


def _rebuffer_and_fold(
    blocks: Iterable[npt.NDArray[np.float32]],
    fold_fn: Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]],
    group: int = DECIMATION_FACTOR,
) -> Iterator[npt.NDArray[np.float32]]:
    """Fold a stream of blocks into the next coarser level, buffering across boundaries.

    The blocks (rank-1 raw runs or rank-2 (min, max) runs) are conceptually
    concatenated and folded by fold_fn in aligned groups of group rows; the
    sub-group leftover at each block boundary is carried onto the next block so
    the split into blocks does not affect the result. The concatenation of the
    yielded arrays equals fold_fn applied to the whole concatenated input, but
    computed in bounded memory, carrying at most group-1 rows at a time.
    """
    carry: npt.NDArray[np.float32] | None = None
    for block in blocks:
        buffer = (
            block if carry is None else np.concatenate([carry, block], axis=0)
        )
        n_full = buffer.shape[0] // group
        split = n_full * group
        if n_full:
            yield fold_fn(buffer[:split])
        carry = buffer[split:]
    if carry is not None and carry.shape[0]:
        yield fold_fn(carry)


def iter_raw_blocks(
    source: ContinuousChannelSource,
    block_samples: int,
) -> Iterator[npt.NDArray[np.float32]]:
    """Yield successive read_samples windows covering the source's raw samples.

    Walks [0, source.num_samples()) in block_samples-sized windows and yields
    each source.read_samples(start, stop); the final window may be shorter.
    Yields nothing when the source has no samples. Raises ValueError if
    block_samples is not positive.
    """
    if block_samples <= 0:
        raise ValueError("block_samples must be positive")
    n = source.num_samples()
    for start in range(0, n, block_samples):
        yield source.read_samples(start, min(start + block_samples, n))


def iter_level0_to_level1(
    source: ContinuousChannelSource, block_samples: int
) -> Iterator[npt.NDArray[np.float32]]:
    """Yield level-1 (min, max) pairs folded from the source's raw samples.

    Reads the source in block_samples-sized windows and folds them across block
    boundaries into one (min, max) row per 4 raw samples (keep-tail). The
    concatenated output equals folding the whole series at once, regardless of
    block_samples. Raises ValueError if block_samples is not positive.
    """
    if block_samples <= 0:
        raise ValueError("block_samples must be positive")
    yield from _rebuffer_and_fold(
        iter_raw_blocks(source, block_samples), fold_raw_block
    )


def iter_array_blocks(
    array: BlockReadableArray, block_len: int
) -> Iterator[npt.NDArray[np.float32]]:
    """Yield successive axis-0 windows of an on-disk array.

    Walks [0, array.shape[0]) in block_len-sized windows and yields each
    array[start:stop] slice; the final window may be shorter (keep-tail).
    Works for rank-1 raw and rank-2 (min, max) arrays alike. Yields nothing
    when the array is empty. Raises ValueError if block_len is not positive.
    """
    if block_len <= 0:
        raise ValueError("block_len must be positive")
    n = array.shape[0]
    for start in range(0, n, block_len):
        stop = min(n, start + block_len)
        yield array[start:stop]
