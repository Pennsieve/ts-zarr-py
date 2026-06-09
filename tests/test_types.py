import dataclasses
from typing import get_args

import pytest

from processor.types import ChannelKind, ChunkShard, LevelPlan


def test_construction_stores_fields():
    plan = LevelPlan(level=0, shape=(1000,), period_us=31.25)
    assert plan.level == 0
    assert plan.shape == (1000,)
    assert plan.period_us == 31.25


def test_is_raw_true_at_level_0():
    assert LevelPlan(level=0, shape=(1000,), period_us=31.25).is_raw is True


@pytest.mark.parametrize("level", [1, 7])
def test_is_raw_false_above_level_0(level):
    plan = LevelPlan(level=level, shape=(250, 2), period_us=125.0)
    assert plan.is_raw is False


def test_frozen_rejects_mutation():
    plan = LevelPlan(level=0, shape=(1000,), period_us=31.25)
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.level = 5


def test_chunkshard_construction_stores_fields():
    cs = ChunkShard(chunk_shape=(262144,), shard_shape=(1048576,))
    assert cs.chunk_shape == (262144,)
    assert cs.shard_shape == (1048576,)


def test_chunkshard_construction_rank2():
    cs = ChunkShard(chunk_shape=(131072, 2), shard_shape=(524288, 2))
    assert cs.chunk_shape == (131072, 2)
    assert cs.shard_shape == (524288, 2)


def test_chunkshard_frozen_rejects_mutation():
    cs = ChunkShard(chunk_shape=(262144,), shard_shape=(1048576,))
    with pytest.raises(dataclasses.FrozenInstanceError):
        cs.chunk_shape = (1,)


def test_channelkind_literal_values():
    assert get_args(ChannelKind.__value__) == ("continuous", "unit")
