from datetime import UTC, datetime

import numpy as np
import pytest
from pynwb.misc import Units
from pynwb.testing.mock.ecephys import mock_ElectricalSeries
from pynwb.testing.mock.file import mock_NWBFile

from processor.nwb_reader import (
    NwbContinuousSource,
    NwbUnitSource,
    build_sources_from_nwb,
)

STARTED = datetime(2021, 6, 1, 12, 0, tzinfo=UTC)


def _make_units(unit_specs, *, name="my_units"):
    """Build a pynwb Units table from (spike_times, waveform_mean) pairs."""
    units = Units(name=name)
    for spike_times, waveform_mean in unit_specs:
        units.add_unit(
            spike_times=list(spike_times),
            waveform_mean=np.asarray(waveform_mean, dtype=float),
        )
    return units


# Two units whose spikes interleave in time; dense cluster ids are 0 and 1.
# Sorted events: 0.0(u0) 0.1(u1) 0.2(u0) 0.3(u1) 0.5(u0).
_WM0 = [1.0, 2.0, 3.0, 4.0]
_WM1 = [5.0, 6.0, 7.0, 8.0]
_TWO_UNITS = [([0.0, 0.2, 0.5], _WM0), ([0.1, 0.3], _WM1)]
_SORTED_TIMES = [0.0, 0.1, 0.2, 0.3, 0.5]
_SORTED_CLUSTERS = [0, 1, 0, 1, 0]
_SORTED_WAVEFORMS = [_WM0, _WM1, _WM0, _WM1, _WM0]


def test_continuous_init_stores_references(electrical_series):
    es = electrical_series()
    started = datetime(2021, 6, 1, 12, 0, tzinfo=UTC)
    src = NwbContinuousSource(es, 2, started)
    assert src._series is es
    assert src._channel_index == 2
    assert src._session_start_time == started


def test_rate_hz_from_series_rate(electrical_series):
    es = electrical_series()  # mock rate is 30000.0 Hz
    started = datetime(2021, 6, 1, 12, 0, tzinfo=UTC)
    src = NwbContinuousSource(es, 2, started)
    assert src.rate_hz() == 30000.0


def test_num_samples_is_time_axis_length(electrical_series):
    es = electrical_series()  # mock data shape is (10, 5)
    started = datetime(2021, 6, 1, 12, 0, tzinfo=UTC)
    src = NwbContinuousSource(es, 2, started)
    assert src.num_samples() == 10


def test_start_us_is_session_start_plus_offset_in_microseconds(
    electrical_series,
):
    es = electrical_series()  # mock starting_time is 0.0 s
    started = datetime(2021, 6, 1, 12, 0, tzinfo=UTC)
    src = NwbContinuousSource(es, 2, started)
    expected = round(started.timestamp() * 1_000_000)
    assert src.start_us() == expected


def test_id_is_a_string(electrical_series):
    es = electrical_series()
    started = datetime(2021, 6, 1, 12, 0, tzinfo=UTC)
    src = NwbContinuousSource(es, 2, started)
    assert isinstance(src.id, str)


def test_read_samples_returns_the_channel_column(electrical_series):
    # The mock's scaling is identity (conversion 1, offset 0).
    es = electrical_series()
    src = NwbContinuousSource(es, 2, STARTED)
    expected = np.asarray(es.data[1:5, 2], dtype=np.float32)
    assert np.array_equal(src.read_samples(1, 5), expected)


def test_read_samples_dtype_is_float32(electrical_series):
    es = electrical_series()
    src = NwbContinuousSource(es, 2, STARTED)
    assert src.read_samples(0, 4).dtype == np.float32


def test_read_samples_shape_matches_window(electrical_series):
    es = electrical_series()
    src = NwbContinuousSource(es, 2, STARTED)
    assert src.read_samples(2, 7).shape == (5,)


def test_read_samples_empty_range_is_length_zero(electrical_series):
    es = electrical_series()
    src = NwbContinuousSource(es, 2, STARTED)
    assert src.read_samples(3, 3).shape == (0,)


def test_read_samples_applies_conversion_and_offset():
    data = np.arange(20, dtype=np.float64).reshape(4, 5)
    es = mock_ElectricalSeries(data=data, conversion=2.0, offset=10.0)
    src = NwbContinuousSource(es, 2, STARTED)
    expected = (data[0:4, 2] * 2.0 + 10.0).astype(np.float32)
    assert np.array_equal(src.read_samples(0, 4), expected)


def test_unit_num_events_sums_all_units_spikes():
    src = NwbUnitSource(_make_units(_TWO_UNITS), 30000.0, STARTED)
    assert src.num_events() == 5


def test_unit_points_per_event_is_waveform_mean_width():
    src = NwbUnitSource(_make_units(_TWO_UNITS), 30000.0, STARTED)
    assert src.points_per_event() == 4


def test_unit_rate_hz_is_the_passed_rate():
    src = NwbUnitSource(_make_units(_TWO_UNITS), 30000.0, STARTED)
    assert src.rate_hz() == 30000.0


def test_unit_start_us_is_session_start_in_microseconds():
    src = NwbUnitSource(_make_units(_TWO_UNITS), 30000.0, STARTED)
    assert src.start_us() == round(STARTED.timestamp() * 1_000_000)


def test_unit_id_is_the_table_name():
    src = NwbUnitSource(
        _make_units(_TWO_UNITS, name="probeA"), 30000.0, STARTED
    )
    assert src.id == "probeA"


def test_unit_read_events_is_ascending_absolute_microseconds():
    src = NwbUnitSource(_make_units(_TWO_UNITS), 30000.0, STARTED)
    base = STARTED.timestamp()
    expected = np.array(
        [round((base + t) * 1_000_000) for t in _SORTED_TIMES],
        dtype=np.int64,
    )
    events = src.read_events(0, 5)
    assert events.dtype == np.int64
    assert np.array_equal(events, expected)
    assert np.all(np.diff(events) >= 0)


def test_unit_read_events_window_is_a_subset():
    src = NwbUnitSource(_make_units(_TWO_UNITS), 30000.0, STARTED)
    assert np.array_equal(src.read_events(1, 3), src.read_events(0, 5)[1:3])


def test_unit_read_units_are_dense_cluster_ids_aligned_with_events():
    src = NwbUnitSource(_make_units(_TWO_UNITS), 30000.0, STARTED)
    units = src.read_units(0, 5)
    assert units.dtype == np.uint8
    assert np.array_equal(units, np.array(_SORTED_CLUSTERS, dtype=np.uint8))


def test_unit_read_waveforms_broadcasts_each_clusters_mean():
    src = NwbUnitSource(_make_units(_TWO_UNITS), 30000.0, STARTED)
    waveforms = src.read_waveforms(0, 5)
    assert waveforms.dtype == np.float32
    assert waveforms.shape == (5, 4)
    assert np.array_equal(
        waveforms, np.array(_SORTED_WAVEFORMS, dtype=np.float32)
    )


def test_unit_read_events_empty_range_is_length_zero():
    src = NwbUnitSource(_make_units(_TWO_UNITS), 30000.0, STARTED)
    assert src.read_events(2, 2).shape == (0,)


def test_unit_read_waveforms_empty_range_keeps_point_axis():
    src = NwbUnitSource(_make_units(_TWO_UNITS), 30000.0, STARTED)
    assert src.read_waveforms(2, 2).shape == (0, 4)


def test_unit_init_rejects_more_than_256_units():
    specs = [([float(i)], [0.0]) for i in range(257)]
    with pytest.raises(ValueError):
        NwbUnitSource(_make_units(specs), 30000.0, STARTED)


def test_build_yields_one_continuous_source_per_channel():
    nwb = mock_NWBFile()
    data = np.arange(50, dtype=np.float64).reshape(10, 5)
    mock_ElectricalSeries(nwbfile=nwb, data=data, rate=30000.0)
    continuous, units = build_sources_from_nwb(nwb)
    assert len(continuous) == 5
    assert units == []
    # Channel index 2 reads its own column.
    assert np.array_equal(
        continuous[2].read_samples(0, 10),
        data[:, 2].astype(np.float32),
    )


def test_build_yields_unit_source_with_first_series_rate():
    nwb = mock_NWBFile()
    mock_ElectricalSeries(nwbfile=nwb, rate=30000.0)
    nwb.units = _make_units(_TWO_UNITS, name="units")
    continuous, units = build_sources_from_nwb(nwb)
    assert len(units) == 1
    assert units[0].rate_hz() == 30000.0
    assert units[0].num_events() == 5


def test_build_with_no_electrical_series_yields_no_continuous():
    nwb = mock_NWBFile()
    continuous, units = build_sources_from_nwb(nwb)
    assert continuous == []
    assert units == []


def test_build_with_no_units_yields_no_unit_sources():
    nwb = mock_NWBFile()
    mock_ElectricalSeries(nwbfile=nwb, rate=30000.0)
    _, units = build_sources_from_nwb(nwb)
    assert units == []


def test_build_units_without_series_rate_raises():
    nwb = mock_NWBFile()
    nwb.units = _make_units(_TWO_UNITS, name="units")
    with pytest.raises(ValueError):
        build_sources_from_nwb(nwb)
