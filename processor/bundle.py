"""Top-level bundle orchestration."""

from collections.abc import Sequence

from processor.protocols import ContinuousChannelSource, UnitChannelSource


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
