"""All Zarr v3 calls for the writer (the only module importing zarr)."""

from pathlib import Path
from typing import Any, cast

import numpy.typing as npt
import zarr
from zarr import Array, Group
from zarr.codecs.zstd import ZstdCodec
from zarr.core.common import JSON
from zarr.storage import LocalStore

# Handle types for the other IO modules, so the literal "import zarr" stays
# confined to this module while callers can still name what they pass and get.
type ZarrGroup = Group
type ZarrArray = Array[Any]


def open_group(path: Path) -> Group:
    """Create or open the Zarr v3 group rooted at a local filesystem path.

    Returns a v3 group backed by a local store at path, creating the directory
    and group metadata when path is empty or absent and opening the existing
    group in place otherwise. Always v3, never v2.
    """
    return zarr.open_group(store=LocalStore(path), mode="a", zarr_format=3)


def create_group_with_attrs(
    parent: Group, name: str, attrs: dict[str, object]
) -> Group:
    """Create a named child group under parent carrying the given attributes.

    Returns a new v3 subgroup at name directly beneath parent (sharing parent's
    store), with its zarr.json attributes set to attrs verbatim and unprefixed.
    An empty attrs leaves the group with no custom attributes.
    """
    return parent.create_group(name, attributes=attrs)


def create_array(
    group: Group,
    name: str,
    shape: tuple[int, ...],
    dtype: npt.DTypeLike,
    chunk_shape: tuple[int, ...],
    shard_shape: tuple[int, ...],
    attrs: dict[str, object],
    zstd_level: int,
) -> Array[Any]:
    """Create a v3 sharded, Zstd-compressed array under group with attributes.

    Returns a new v3 array at name beneath group with the given shape and dtype.
    The outer shard is shard_shape, split into inner chunks of chunk_shape by the
    ZEP2 sharding codec; chunk_shape must divide shard_shape along each axis. The
    inner chunks are Zstd-compressed at zstd_level. The attrs dict is set verbatim
    and unprefixed in the array's zarr.json.
    """
    return group.create_array(
        name=name,
        shape=shape,
        dtype=dtype,
        chunks=chunk_shape,
        shards=shard_shape,
        compressors=ZstdCodec(level=zstd_level),
        attributes=cast("dict[str, JSON]", attrs),
    )


def write_region(
    array: Array[Any], start: int, block: npt.NDArray[Any]
) -> None:
    """Write a contiguous block of rows into array at axis-0 offset start.

    Writes block over the half-open axis-0 range from start to start plus the
    block's leading length. The block's trailing dimensions must match the
    array's (rank-1 raw samples, or rank-2 (rows, 2) min/max envelopes); the
    region must lie within the array's bounds. Returns nothing.
    """
    array[start : start + block.shape[0]] = block


def consolidate(root: Group) -> None:
    """Write the v3 inline consolidated_metadata block into the root group.

    Gathers the metadata of every descendant group and array into the root's
    own zarr.json as a single inline consolidated_metadata block (the v3 form),
    not a v2 .zmetadata sidecar, so the reader fetches one file. Returns nothing.
    """
    zarr.consolidate_metadata(root.store)
