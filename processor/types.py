"""Shared value types for the writer."""

from dataclasses import dataclass
from typing import Literal

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


@dataclass(frozen=True, slots=True)
class ChunkShard:
    """Inner-chunk and outer-shard shapes for one Zarr array.

    chunk_shape is the inner (compressed) chunk; shard_shape is the outer shard
    that groups whole chunks under the ZEP2 sharding codec.
    """

    chunk_shape: tuple[int, ...]
    shard_shape: tuple[int, ...]
