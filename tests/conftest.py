"""Shared pytest fixtures."""

import numpy as np
import pytest


class ArrayContinuousSource:
    """In-memory ContinuousChannelSource backed by a float32 array, for tests."""

    def __init__(self, samples, *, id="ch-0", rate_hz=32000.0, start_us=0):
        self._samples = np.asarray(samples, dtype=np.float32)
        self.id = id
        self._rate_hz = rate_hz
        self._start_us = start_us

    def rate_hz(self):
        return self._rate_hz

    def start_us(self):
        return self._start_us

    def num_samples(self):
        return int(self._samples.shape[0])

    def read_samples(self, start, stop):
        return self._samples[start:stop]


@pytest.fixture
def continuous_source():
    return ArrayContinuousSource


class ArrayUnitSource:
    """In-memory UnitChannelSource backed by arrays, for tests.

    events are int64 absolute microseconds; units default to zeros and
    waveforms to a (n_events, points_per_event) float32 ramp when not given.
    """

    def __init__(
        self,
        events,
        *,
        units=None,
        waveforms=None,
        points_per_event=4,
        id="unit-0",
        rate_hz=32000.0,
        start_us=0,
    ):
        self._events = np.asarray(events, dtype=np.int64)
        n = int(self._events.shape[0])
        self._points_per_event = points_per_event
        self._units = (
            np.zeros(n, dtype=np.uint8)
            if units is None
            else np.asarray(units, dtype=np.uint8)
        )
        self._waveforms = (
            np.zeros((n, points_per_event), dtype=np.float32)
            if waveforms is None
            else np.asarray(waveforms, dtype=np.float32)
        )
        self.id = id
        self._rate_hz = rate_hz
        self._start_us = start_us

    def rate_hz(self):
        return self._rate_hz

    def start_us(self):
        return self._start_us

    def num_events(self):
        return int(self._events.shape[0])

    def points_per_event(self):
        return self._points_per_event

    def read_events(self, start, stop):
        return self._events[start:stop]

    def read_units(self, start, stop):
        return self._units[start:stop]

    def read_waveforms(self, start, stop):
        return self._waveforms[start:stop]


@pytest.fixture
def unit_source():
    return ArrayUnitSource
