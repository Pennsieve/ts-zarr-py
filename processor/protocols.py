"""Typed source protocols the writer consumes for channel data."""

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


@runtime_checkable
class ContinuousChannelSource(Protocol):
    """A continuous channel the writer reads raw samples from.

    ``id`` is the opaque upstream identifier the reader joins on for display
    metadata. ``rate_hz`` is in Hz; ``start_us`` is the wall-clock microseconds
    of sample 0. ``read_samples(start, stop)`` returns the half-open
    ``[start, stop)`` sample window as float32.
    """

    id: str

    def rate_hz(self) -> float: ...
    def start_us(self) -> int: ...
    def num_samples(self) -> int: ...
    def read_samples(self, start: int, stop: int) -> npt.NDArray[np.float32]: ...
