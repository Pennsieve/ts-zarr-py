"""Top-level bundle orchestration."""

import shutil
from collections.abc import Sequence
from pathlib import Path

from ts_zarr.protocols import ContinuousChannelSource, UnitChannelSource
from ts_zarr.types import WriteOpts
from ts_zarr.write_continuous import write_continuous_channel
from ts_zarr.write_unit import write_unit_channel
from ts_zarr.zarr_io import ZarrGroup, consolidate, open_group


def assign_indices(
    continuous: Sequence[ContinuousChannelSource],
    units: Sequence[UnitChannelSource],
) -> list[tuple[int, ContinuousChannelSource | UnitChannelSource]]:
    """Assign each channel a digit index, continuous first then unit.

    Returns (index, source) pairs with contiguous indices 0..N-1 (N = total
    channels), in a deterministic order: the continuous sources in input order,
    then the unit sources in input order. These indices become the bundle's
    digit-named channel-group directories.
    """
    return list(enumerate([*continuous, *units]))


def atomic_publish(staging_dir: Path, final_dir: Path) -> None:
    """Move a fully-staged bundle to its final path in one atomic rename.

    Renames staging_dir onto final_dir so a reader sees either the old bundle or
    the complete new one, never a partial write. Replaces final_dir if it
    already exists. Both paths must be on the same filesystem; a cross-device
    move raises OSError (the copy-then-swap fallback is the caller's concern).
    Returns nothing.
    """
    if not final_dir.exists():
        staging_dir.replace(final_dir)
        return

    # Move the old bundle aside before renaming the new one in, so the old
    # bundle survives until the swap succeeds (a single os.replace cannot
    # overwrite a non-empty directory). The final path is briefly absent
    # between the two renames; a reader then sees no bundle, never a partial.
    backup = final_dir.with_name(final_dir.name + ".old")
    final_dir.replace(backup)
    try:
        staging_dir.replace(final_dir)
    except OSError:
        backup.replace(final_dir)
        raise
    shutil.rmtree(backup)


def write_all_channels(
    root: ZarrGroup,
    indexed: Sequence[tuple[int, ContinuousChannelSource | UnitChannelSource]],
    opts: WriteOpts,
) -> None:
    """Write every indexed channel under root, dispatching by source type.

    Writes each (index, source) pair as its own channel group beneath root: a
    continuous source goes through the continuous-channel writer, a unit source
    through the unit-channel writer, each governed by opts. An empty list writes
    nothing. Returns nothing.
    """
    for index, source in indexed:
        if isinstance(source, UnitChannelSource):
            write_unit_channel(root, index, source, opts=opts)
        else:
            write_continuous_channel(root, index, source, opts=opts)


def write_bundle(
    continuous: Sequence[ContinuousChannelSource],
    units: Sequence[UnitChannelSource],
    *,
    staging_dir: Path,
    final_dir: Path,
    opts: WriteOpts,
) -> None:
    """Build the whole viewer bundle and publish it atomically.

    Stages every continuous and unit channel into a fresh root group at
    staging_dir (continuous first, then unit, with contiguous digit indices),
    consolidates the root metadata, then renames the staged bundle onto
    final_dir so a reader never sees a partial write. Channel sizing and
    compression follow opts. Returns nothing.
    """
    root = open_group(staging_dir)
    write_all_channels(root, assign_indices(continuous, units), opts)
    consolidate(root)
    atomic_publish(staging_dir, final_dir)
