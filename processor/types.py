"""Shared value types for the writer."""

from dataclasses import dataclass
from typing import Literal

from processor.constants import INNER_CHUNK_SAMPLES, TARGET_SHARD_BYTES

type ChannelKind = Literal["continuous", "unit"]


@dataclass(frozen=True, slots=True)
class LevelPlan:
    """Resolved shape and time resolution of one pyramid level of a channel.

    Level 0 holds raw samples, shape (n_bins,). Levels >= 1 hold (min, max)
    envelopes, shape (n_bins, 2) with the trailing axis being the pair.

    period_us is the wall-clock microseconds that one bin spans, widening with
    each coarser level. Float because the rate need not divide evenly into
    microseconds.
    """

    level: int
    shape: tuple[int, ...]
    period_us: float

    @property
    def is_raw(self) -> bool:
        """Whether this level holds raw samples rather than min/max envelopes.

        Lets the continuous channel writer choose the level-0 path (stream raw
        samples in) over the envelope path (fold from the level below).
        """
        return self.level == 0

    @property
    def name(self) -> str:
        """Canonical Zarr array key for this level.

        Levels are stored under their decimal level number, so level 0 maps to
        "0", level 1 to "1", etc.
        """
        return str(self.level)


@dataclass(frozen=True, slots=True)
class ChunkShard:
    """Inner-chunk and outer-shard shapes for one Zarr array.

    chunk_shape is the inner (compressed) chunk; shard_shape is the outer shard
    that groups whole chunks under the ZEP2 sharding codec.
    """

    chunk_shape: tuple[int, ...]
    shard_shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class WriteOpts:
    """Tunable settings shared by the channel writers.

    zstd_level is the Zstd compression level for every array. max_levels and
    min_bins bound the pyramid (passed to plan_levels). inner_len and
    target_shard_bytes size the inner chunk and outer shard (passed to
    chunk_and_shard).
    """

    zstd_level: int = 5
    max_levels: int = 8
    min_bins: int = 1024
    inner_len: int = INNER_CHUNK_SAMPLES
    target_shard_bytes: int = TARGET_SHARD_BYTES
