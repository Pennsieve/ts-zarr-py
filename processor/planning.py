"""Pyramid level planning: level counts, shapes, and time resolutions."""

from processor.types import LevelPlan


def level_count(num_samples: int, max_levels: int = 8, min_bins: int = 1024) -> int:
    """Return the number of pyramid levels for a channel of num_samples samples.

    Level 0 (raw) always counts, even when num_samples is 0. Each coarser level is
    included only while it would still hold at least min_bins bins, and the total
    is capped at max_levels.
    """
    count = 1
    for level in range(1, max_levels):
        if num_samples // 4**level < min_bins:
            break
        count += 1
    return count


def level_num_bins(num_samples: int, level: int) -> int:
    """Return the number of bins at a pyramid level for num_samples samples.

    Level 0 has one bin per raw sample. Each coarser level has one bin per
    disjoint block of 4 bins from the level below, with a partial final block
    kept as one bin so no samples are dropped.

    The count is computed with integer arithmetic, not by rounding a float
    division, so it stays exact even for very large num_samples.
    Float division loses precision past 2**53 samples.
    """
    samples_per_bin: int = 4**level
    return -(-num_samples // samples_per_bin)


def level_period_us(level0_period_us: float, level: int) -> float:
    """Return the microseconds that one bin spans at the given pyramid level.

    This is the level-0 sample period widened by the pyramid's 4x-per-level
    decimation, so level 0 returns level0_period_us unchanged and each higher
    level spans 4x more time than the one below.
    """
    return float(level0_period_us * 4**level)


def level_shape(num_samples: int, level: int) -> tuple[int, ...]:
    """Return the array shape for a pyramid level.

    Level 0 is rank-1 (a single axis of raw samples). Each coarser level is
    rank-2 with a trailing axis of length 2 holding the (min, max) pair, and a
    leading axis of one entry per bin.
    """
    return (num_samples,) if level == 0 else (level_num_bins(num_samples, level), 2)


def plan_levels(
    num_samples: int,
    level0_period_us: float,
    max_levels: int = 8,
    min_bins: int = 1024,
) -> list[LevelPlan]:
    """Return the pyramid plan for a channel: one LevelPlan per level.

    The number of levels comes from level_count; level 0 is the raw level and the
    rest are min/max envelope levels. Each entry carries that level's array shape
    and the wall-clock microseconds one of its bins spans.
    """
    return [
        LevelPlan(
            level=k,
            shape=level_shape(num_samples, k),
            period_us=level_period_us(level0_period_us, k),
        )
        for k in range(level_count(num_samples, max_levels, min_bins))
    ]
