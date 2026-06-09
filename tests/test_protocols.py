import numpy as np

from processor.protocols import ContinuousChannelSource


class FakeContinuous:
    id = "ch-0"

    def rate_hz(self):
        return 32000.0

    def start_us(self):
        return 0

    def num_samples(self):
        return 4

    def read_samples(self, start, stop):
        return np.zeros(stop - start, dtype=np.float32)


def test_conforming_instance_is_recognized():
    assert isinstance(FakeContinuous(), ContinuousChannelSource)


def test_missing_method_is_not_recognized():
    class MissingReadSamples:
        id = "ch-0"

        def rate_hz(self):
            return 32000.0

        def start_us(self):
            return 0

        def num_samples(self):
            return 4

    assert not isinstance(MissingReadSamples(), ContinuousChannelSource)


def test_missing_id_attr_is_not_recognized():
    class MissingId:
        def rate_hz(self):
            return 32000.0

        def start_us(self):
            return 0

        def num_samples(self):
            return 4

        def read_samples(self, start, stop):
            return np.zeros(stop - start, dtype=np.float32)

    assert not isinstance(MissingId(), ContinuousChannelSource)
