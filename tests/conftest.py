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
