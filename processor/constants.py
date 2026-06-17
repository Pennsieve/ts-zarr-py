"""Format constants for the pyramid bundle."""

from typing import Final

DECIMATION_FACTOR: Final = 4
"""Samples folded into one bin per pyramid level (4x coarser each level)."""

ENVELOPE_PAIR_SIZE: Final = 2
"""Length of the trailing (min, max) axis on every coarser pyramid level."""

FLOAT32_BYTES: Final = 4
"""Byte width of one float32 sample, for shard-size computation."""

MICROSECONDS_PER_SECOND: Final = 1_000_000.0
"""Microseconds in one second, for converting a sample rate to a period."""

INNER_CHUNK_SAMPLES: Final = 2**18
"""Target inner Zarr chunk length in samples (~256K)."""

TARGET_SHARD_BYTES: Final = 16 * 2**20
"""Target outer shard size in bytes (~16 MiB) for sharded pyramid arrays."""
