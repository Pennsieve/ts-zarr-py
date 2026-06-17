"""Min/max pyramid folding: reduce a level to the next coarser one in blocks of 4."""

import numpy as np
import numpy.typing as npt

from processor.constants import DECIMATION_FACTOR, ENVELOPE_PAIR_SIZE


def _block_split(
    length: int, block: int = DECIMATION_FACTOR
) -> tuple[int, int]:
    """Return (n_full, tail_len) for splitting length items into blocks.

    Raise ValueError if block is not positive.

    n_full is the count of complete blocks of size block; tail_len is the
    leftover count (0..block-1), kept as a final partial block (no padding).
    """
    if block <= 0:
        raise ValueError("block must be positive")
    return (length // block, length % block)


def fold_raw_block(raw: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Fold raw samples into (min, max) envelope pairs over disjoint blocks of 4.

    Take a 1-D float32 array and return a rank-2 float32 array with one row per
    block of 4 input samples (row k is (min, max) of samples 4k..4k+3), keeping a
    final partial block of 1-3 samples as one row. Uses plain min/max so NaN
    propagates into its bin (the reader treats non-finite bins as no-data).
    """
    block = DECIMATION_FACTOR
    n_full, tail_len = _block_split(raw.shape[0], block)
    split = n_full * block
    out = np.empty(
        (n_full + (1 if tail_len else 0), ENVELOPE_PAIR_SIZE), dtype=np.float32
    )
    if n_full:
        full = raw[:split].reshape(n_full, block)
        out[:n_full, 0] = full.min(axis=1)
        out[:n_full, 1] = full.max(axis=1)
    if tail_len:
        tail = raw[split:]
        out[n_full, 0] = tail.min()
        out[n_full, 1] = tail.max()
    return out


def fold_pair_block(pairs: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Fold (min, max) pairs into coarser pairs over disjoint blocks of 4.

    Take a rank-2 float32 array of (min, max) rows and return a rank-2 float32
    array with one row per block of 4 input rows: its min is the smallest of the
    4 mins (column 0) and its max is the largest of the 4 maxes (column 1). A
    final partial block of 1-3 rows is kept as one row. Plain min/max, so NaN
    propagates.
    """
    block = DECIMATION_FACTOR
    n_full, tail_len = _block_split(pairs.shape[0], block)
    split = n_full * block
    out = np.empty(
        (n_full + (1 if tail_len else 0), ENVELOPE_PAIR_SIZE), dtype=np.float32
    )
    if n_full:
        full = pairs[:split].reshape(n_full, block, ENVELOPE_PAIR_SIZE)
        out[:n_full, 0] = full[:, :, 0].min(axis=1)
        out[:n_full, 1] = full[:, :, 1].max(axis=1)
    if tail_len:
        tail = pairs[split:]
        out[n_full, 0] = tail[:, 0].min()
        out[n_full, 1] = tail[:, 1].max()
    return out


def fold_block(arr: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Fold one pyramid level to the next coarser one, dispatching on rank.

    A rank-1 array is treated as raw samples and folded by fold_raw_block; a
    rank-2 array is treated as (min, max) pairs and folded by fold_pair_block.
    Raises ValueError for any other rank.
    """
    match arr.ndim:
        case 1:
            return fold_raw_block(arr)
        case 2:
            return fold_pair_block(arr)
        case _:
            raise ValueError(f"unsupported array rank: {arr.ndim}")
