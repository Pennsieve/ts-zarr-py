"""Compose streaming and zarr I/O to write one continuous channel's levels."""

from collections.abc import Iterable
from typing import cast

import numpy as np
import numpy.typing as npt

from processor.attrs import channel_group_attrs, level_array_attrs
from processor.constants import FLOAT32_BYTES
from processor.fold import fold_block
from processor.planning import level0_period_us, plan_levels
from processor.protocols import ContinuousChannelSource
from processor.sizing import chunk_and_shard
from processor.streaming import (
    BlockReadableArray,
    _rebuffer_and_fold,
    iter_array_blocks,
    iter_raw_blocks,
)
from processor.types import ChunkShard, LevelPlan, WriteOpts
from processor.zarr_io import (
    ZarrArray,
    ZarrGroup,
    create_array,
    create_group_with_attrs,
    write_region,
)


def _write_blocks(
    array: ZarrArray, blocks: Iterable[npt.NDArray[np.float32]]
) -> None:
    """Write each block to array at its running axis-0 offset."""
    start = 0
    for block in blocks:
        write_region(array, start, block)
        start += block.shape[0]


def write_level0(
    group: ZarrGroup,
    source: ContinuousChannelSource,
    plan: LevelPlan,
    sizing: ChunkShard,
    zstd_level: int,
) -> ZarrArray:
    """Create the level-0 raw array under group and stream the source into it.

    Creates the array named "0" with shape plan.shape, float32 dtype, the chunk
    and shard shapes from sizing, and the level period_us as its sole attribute,
    then writes the source's raw samples in chunk-sized blocks at their running
    axis-0 offset. Returns the created array. An empty source (no samples)
    creates the array and writes nothing. Raises ValueError if plan does not
    describe level 0 (raw).
    """
    if not plan.is_raw:
        raise ValueError("plan level must be raw")

    array = create_array(
        group,
        str(plan.level),
        plan.shape,
        np.float32,
        sizing.chunk_shape,
        sizing.shard_shape,
        level_array_attrs(plan.period_us),
        zstd_level,
    )
    _write_blocks(array, iter_raw_blocks(source, sizing.chunk_shape[0]))
    return array


def write_level_from_previous(
    group: ZarrGroup,
    prev: ZarrArray,
    plan: LevelPlan,
    sizing: ChunkShard,
    zstd_level: int,
) -> ZarrArray:
    """Create the level named plan.level by folding the previous level into it.

    Creates the array named str(plan.level) with shape plan.shape, float32
    dtype, the chunk and shard shapes from sizing, and the level period_us as
    its sole attribute. Reads prev in chunk-sized axis-0 blocks and folds them
    across block boundaries into one (min, max) row per 4 input rows (min of
    mins, max of maxes when prev is itself an envelope level), writing each
    folded block at its running axis-0 offset. Returns the created array. An
    empty prev (no rows) creates the array and writes nothing. Raises ValueError
    if plan describes level 0; this path builds only envelope levels (>= 1).
    """
    if plan.is_raw:
        raise ValueError("level must not be raw")

    array = create_array(
        group,
        str(plan.level),
        plan.shape,
        np.float32,
        sizing.chunk_shape,
        sizing.shard_shape,
        level_array_attrs(plan.period_us),
        zstd_level,
    )

    _write_blocks(
        array,
        _rebuffer_and_fold(
            iter_array_blocks(
                cast("BlockReadableArray", prev), sizing.chunk_shape[0]
            ),
            fold_block,
        ),
    )
    return array


def write_continuous_channel(
    parent: ZarrGroup,
    index: int,
    source: ContinuousChannelSource,
    *,
    opts: WriteOpts,
) -> None:
    """Write one continuous channel as the subgroup named str(index).

    Creates the channel group under parent carrying its continuous-kind
    attributes (id, rate, start), plans the pyramid from the source's sample
    count and rate, writes level 0 from the source, then folds each coarser
    level from the array written just below it. Every level array is sized and
    compressed per opts. Returns nothing. A channel that plans to a single
    level writes only level 0.
    """
    attributes = channel_group_attrs(
        source.id, source.rate_hz(), source.start_us(), "continuous"
    )
    group = create_group_with_attrs(parent, str(index), attributes)
    previous = None
    for plan in plan_levels(
        source.num_samples(),
        level0_period_us(source.rate_hz()),
        opts.max_levels,
        opts.min_bins,
    ):
        sizing = chunk_and_shard(
            level_shape=plan.shape,
            dtype_size=FLOAT32_BYTES,
            inner_len=opts.inner_len,
            target_shard_bytes=opts.target_shard_bytes,
        )
        if previous is None:
            previous = write_level0(
                group=group,
                source=source,
                plan=plan,
                sizing=sizing,
                zstd_level=opts.zstd_level,
            )
        else:
            previous = write_level_from_previous(
                group=group,
                prev=previous,
                plan=plan,
                sizing=sizing,
                zstd_level=opts.zstd_level,
            )
