"""Attribute-dict builders for the bundle (the only custom format surface)."""

from processor.types import ChannelKind


def root_attrs() -> dict[str, object]:
    """Return the custom attributes for the bundle's root group.

    The root group carries only Zarr's own consolidated_metadata; the format
    defines no custom root attributes, so this is always empty.
    """
    return {}


def channel_group_attrs(
    id: str,
    rate_hz: float,
    start_us: int,
    kind: ChannelKind,
    name: str = "",
    unit: str = "",
) -> dict[str, object]:
    """Return the channel-group zarr.json attributes for one channel.

    The six keys the reader joins on: id (opaque upstream identifier), rate_hz
    (sample rate; for unit channels the waveform rate), start_us (wall-clock
    microseconds of sample 0 / recording start), kind ("continuous" or "unit"),
    name (human-readable display label), and unit (physical unit of the stored
    samples, e.g. "uV") — the two that make the bundle self-describing. Values
    pass through unchanged.
    """
    return {
        "id": id,
        "rate_hz": rate_hz,
        "start_us": start_us,
        "kind": kind,
        "name": name,
        "unit": unit,
    }


def level_array_attrs(period_us: float) -> dict[str, object]:
    """Return the pyramid-level array attributes.

    period_us is the microseconds one bin spans at this level; the reader uses
    it to pick the level whose period best matches the requested pixel width.
    """
    return {"period_us": period_us}


def waveform_array_attrs(period_us: float) -> dict[str, object]:
    """Return the waveforms array attributes for a unit channel.

    period_us is the sample period within a spike waveform (waveform sample rate
    is 1e6 / period_us) — not a pyramid-level period.
    """
    return {"period_us": period_us}
