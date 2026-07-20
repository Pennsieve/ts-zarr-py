"""Concrete NWB adapters implementing the channel-source protocols."""

from collections.abc import Iterator
from datetime import datetime

import numpy as np
import numpy.typing as npt
from pynwb import NWBFile
from pynwb.ecephys import ElectricalSeries
from pynwb.misc import Units

from processor.constants import (
    MAX_UNIT_CLUSTERS,
    MICROSECONDS_PER_SECOND,
    UNIT_TO_UV,
)


class NwbContinuousSource:
    """A continuous channel backed by one column of an NWB ElectricalSeries."""

    def __init__(
        self,
        electrical_series: ElectricalSeries,
        channel_index: int,
        session_start_time: datetime,
    ) -> None:
        """Bind one channel of an ElectricalSeries as a continuous source.

        electrical_series holds the multichannel recording; channel_index selects
        the column this source exposes; session_start_time is the recording's
        wall-clock origin, combined with the series' own start offset to place
        sample 0 in absolute microseconds.
        """
        self._series = electrical_series
        self._channel_index = channel_index
        self._session_start_time = session_start_time

    @property
    def id(self) -> str:
        """Opaque upstream identifier for this channel.

        Derived from the selected electrode so the reader can join the channel
        to its display metadata.
        """
        electrodes = self._series.electrodes
        row_index = electrodes.data[self._channel_index]
        return str(electrodes.table.id[row_index])

    @property
    def name(self) -> str:
        """Human-readable display label for this channel.

        Taken from the selected electrode's channel_name column, falling back to
        its label column, then to the opaque id when the table carries neither.
        """
        electrodes = self._series.electrodes
        row_index = electrodes.data[self._channel_index]
        table = electrodes.table
        for column in ("channel_name", "label"):
            if column in table.colnames:
                return str(table[column][row_index])
        return self.id

    @property
    def unit(self) -> str:
        """Physical unit of the samples this source yields.

        Always microvolts ("uV"): read_samples normalizes every series to
        microvolts, so the stored unit is fixed regardless of the source's own.
        """
        return "uV"

    def rate_hz(self) -> float:
        """Return the channel's sample rate in hertz.

        Taken from the ElectricalSeries' own rate, or inferred from its explicit
        timestamps when no rate is set.
        """
        return float(self._series.rate)

    def start_us(self) -> int:
        """Return the wall-clock microseconds of sample index 0.

        The session start combined with the series' own start offset, rounded to
        whole microseconds.
        """
        start_s: float = (
            self._session_start_time.timestamp() + self._series.starting_time
        )
        return round(start_s * MICROSECONDS_PER_SECOND)

    def num_samples(self) -> int:
        """Return the total number of raw samples in the channel.

        The length of the series' time axis (axis 0 of its data), shared by
        every channel.
        """
        return int(self._series.data.shape[0])

    def read_samples(self, start: int, stop: int) -> npt.NDArray[np.float32]:
        """Return the half-open [start, stop) sample window as float32 microvolts.

        Reads this channel's column of the ElectricalSeries over the given
        axis-0 range in one HDF5 read and scales it to microvolts: the series'
        affine scaling (raw times the conversion, times the per-channel
        conversion when the series carries one, plus the offset) followed by the
        volts-family to microvolts factor for the series' unit. Raises ValueError
        if the series' unit is not a recognized volts family. An empty range
        (stop <= start) yields a length-0 array.
        """
        column: npt.NDArray[np.float64] = np.asarray(
            self._series.data[start:stop, self._channel_index],
            dtype=np.float64,
        )
        scaled = column * float(self._series.conversion)
        if self._series.channel_conversion is not None:
            scaled = scaled * float(
                self._series.channel_conversion[self._channel_index]
            )
        scaled = scaled + float(self._series.offset)
        unit = str(self._series.unit).lower()
        if unit not in UNIT_TO_UV:
            raise ValueError(f"unsupported ElectricalSeries unit: {unit!r}")
        return (scaled * UNIT_TO_UV[unit]).astype(np.float32)


class NwbUnitSource:
    """A unit (spike) channel backed by an NWB Units table.

    The NWB Units table is organized per unit: each row is one cluster with a
    ragged list of spike times and a single waveform_mean template. This adapter
    flattens it into the per-event streams the bundle stores: all units' spikes
    merged into one timestamp series sorted ascending, each event tagged with its
    cluster's dense id, and each event carrying its cluster's mean waveform.

    Cluster ids are assigned densely as 0..k-1 in the table's row order (not the
    upstream unit ids, which are display metadata fetched separately), so they
    fit the uint8 units array; a table of more than 256 units is rejected. The
    waveform sample rate has no standard home in the Units table, so it is passed
    in explicitly rather than guessed.
    """

    def __init__(
        self,
        units: Units,
        waveform_rate_hz: float,
        session_start_time: datetime,
    ) -> None:
        """Bind an NWB Units table as a flattened per-event spike source.

        units is the table to wrap; waveform_rate_hz is the sample rate within a
        waveform (the table carries no rate); session_start_time is the
        recording's wall-clock origin placing event timestamps in absolute
        microseconds. Raises ValueError if the table has more than 256 units
        (cluster ids would not fit uint8).
        """
        unit_count = len(units)
        if unit_count > MAX_UNIT_CLUSTERS:
            raise ValueError("a unit channel holds at most 256 clusters")

        self._id = str(units.name)
        self._rate_hz = float(waveform_rate_hz)
        self._start_us = round(
            session_start_time.timestamp() * MICROSECONDS_PER_SECOND
        )

        session_s = session_start_time.timestamp()
        times: list[npt.NDArray[np.float64]] = []
        clusters: list[npt.NDArray[np.uint8]] = []
        waveforms: list[npt.NDArray[np.float32]] = []
        for cluster_id in range(unit_count):
            spike_times: npt.NDArray[np.float64] = np.asarray(
                units.get_unit_spike_times(cluster_id), dtype=np.float64
            )
            mean: npt.NDArray[np.float32] = np.asarray(
                units["waveform_mean"][cluster_id], dtype=np.float32
            )
            count = spike_times.shape[0]
            times.append(spike_times)
            clusters.append(np.full(count, cluster_id, dtype=np.uint8))
            waveforms.append(np.broadcast_to(mean, (count, mean.shape[0])))

        all_times = np.concatenate(times)
        order = np.argsort(all_times, kind="stable")
        self._events = (
            ((all_times[order] + session_s) * MICROSECONDS_PER_SECOND)
            .round()
            .astype(np.int64)
        )
        self._units = np.concatenate(clusters)[order]
        self._waveforms = np.concatenate(waveforms)[order]

    @property
    def id(self) -> str:
        """Opaque upstream identifier for this unit channel.

        Derived from the Units table's name so the reader can join the channel
        to its display metadata.
        """
        return self._id

    @property
    def name(self) -> str:
        """Human-readable display label for this unit channel.

        The Units table's name, the only human-readable handle a Units table
        carries.
        """
        return self._id

    @property
    def unit(self) -> str:
        """Physical unit of the samples this source yields.

        Reported as microvolts ("uV") for consistency with continuous channels
        and the bundle's microvolt convention. A Units table's waveform_mean
        carries no unit metadata, so the waveform amplitudes are passed through
        unscaled rather than normalized from a guessed source unit.
        """
        return "uV"

    def rate_hz(self) -> float:
        """Return the waveform sample rate in hertz.

        The explicit rate bound at construction; the Units table itself carries
        no rate.
        """
        return self._rate_hz

    def start_us(self) -> int:
        """Return the wall-clock microseconds of the recording start.

        The session start rounded to whole microseconds; unit events are stored
        as absolute timestamps, so no per-event offset is applied beyond this.
        """
        return self._start_us

    def num_events(self) -> int:
        """Return the total number of spike events across all units."""
        return int(self._events.shape[0])

    def points_per_event(self) -> int:
        """Return the number of waveform samples stored per event.

        The width of the units' waveform_mean template, shared by every event.
        """
        return int(self._waveforms.shape[1])

    def read_events(self, start: int, stop: int) -> npt.NDArray[np.int64]:
        """Return the half-open [start, stop) window of event timestamps.

        Absolute-microsecond int64 timestamps over all units' spikes merged and
        sorted ascending. An empty range (stop <= start) yields a length-0 array.
        """
        return self._events[start:stop]

    def read_units(self, start: int, stop: int) -> npt.NDArray[np.uint8]:
        """Return the half-open [start, stop) window of per-event cluster ids.

        uint8 dense cluster ids (0..k-1 in table row order) aligned with the
        events at the same indices. An empty range yields a length-0 array.
        """
        return self._units[start:stop]

    def read_waveforms(self, start: int, stop: int) -> npt.NDArray[np.float32]:
        """Return float32 waveforms for events [start, stop).

        Shape (stop - start, points_per_event); row k is the waveform_mean of the
        cluster that produced event k (every spike of a cluster shares that
        cluster's mean template). An empty range yields a (0, points_per_event)
        array.
        """
        return self._waveforms[start:stop]


def _iter_units_tables(nwbfile: NWBFile) -> Iterator[Units]:
    """Yield every Units table in the file in deterministic discovery order.

    The root Units table first (when present), then the Units containers of
    each processing module, modules taken in name order and their containers in
    name order, so the same file always produces the same unit-channel order.
    """
    if nwbfile.units is not None:
        yield nwbfile.units
    for module_name in sorted(nwbfile.processing):
        module = nwbfile.processing[module_name]
        for container_name in sorted(module.data_interfaces):
            container = module.data_interfaces[container_name]
            if isinstance(container, Units):
                yield container


def build_sources_from_nwb(
    nwbfile: NWBFile,
) -> tuple[list[NwbContinuousSource], list[NwbUnitSource]]:
    """Discover the channel sources to write from an open NWB file.

    Returns the continuous sources and the unit sources, in that order. Every
    ElectricalSeries in the file's acquisition contributes one continuous source
    per channel column, in series order then channel order. Each Units table
    contributes one unit source, discovered as the root table then the tables in
    processing modules (modules by name, containers by name). The recording's
    session start time places all timestamps in absolute microseconds. Unit
    waveforms have no rate of their own, so they reuse the sample rate of the
    first ElectricalSeries. A file with no ElectricalSeries yields no continuous
    sources; one with no Units table yields no unit sources. Raises ValueError
    if any Units table is present but no ElectricalSeries supplies a waveform
    rate.
    """
    session_start = nwbfile.session_start_time
    series = [
        acq
        for acq in nwbfile.acquisition.values()
        if isinstance(acq, ElectricalSeries)
    ]
    continuous = [
        NwbContinuousSource(es, channel_index, session_start)
        for es in series
        for channel_index in range(es.data.shape[1])
    ]

    units: list[NwbUnitSource] = []
    for table in _iter_units_tables(nwbfile):
        if not series:
            raise ValueError("unit channels need an ElectricalSeries rate")
        waveform_rate_hz = float(series[0].rate)
        units.append(NwbUnitSource(table, waveform_rate_hz, session_start))

    return continuous, units
