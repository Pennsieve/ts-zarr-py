"""Format constants for the pyramid bundle."""

from typing import Final

DECIMATION_FACTOR: Final = 4
"""Samples folded into one bin per pyramid level (4x coarser each level)."""

INNER_CHUNK_SAMPLES: Final = 2**18
"""Target inner Zarr chunk length in samples (~256K)."""

TARGET_SHARD_BYTES: Final = 16 * 2**20
"""Target outer shard size in bytes (~16 MiB) for sharded pyramid arrays."""
