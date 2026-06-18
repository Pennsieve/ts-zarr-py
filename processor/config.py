"""Run configuration assembled from the environment and command line."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from processor.types import WriteOpts


@dataclass(frozen=True, slots=True)
class Config:
    """Everything one bundle run needs, resolved from env and argv.

    nwb_path is the input NWB file; final_dir is where the published bundle
    lands; staging_dir is the scratch path the bundle is built in before its
    atomic rename onto final_dir. opts carries the writer settings.
    """

    nwb_path: Path
    staging_dir: Path
    final_dir: Path
    opts: WriteOpts


def load_config(env: Mapping[str, str], argv: Sequence[str]) -> Config:
    """Resolve a Config from environment variables and command-line arguments.

    argv holds the arguments after the program name; the first two positionals
    are the input NWB path and the final output directory. The writer settings
    and the staging directory come from env under the ZARR_WRITER_ prefix
    (ZARR_WRITER_STAGING_DIR, ZARR_WRITER_ZSTD_LEVEL, ZARR_WRITER_MAX_LEVELS,
    ZARR_WRITER_MIN_BINS, ZARR_WRITER_INNER_LEN, ZARR_WRITER_TARGET_SHARD_BYTES).

    Unset settings fall back to the WriteOpts defaults; an unset staging
    directory derives from final_dir. Raises ValueError if fewer than two
    positionals are given or if any env value is not a valid integer.
    """
    positional_count = 2
    if len(argv) < positional_count:
        raise ValueError(
            "load_config needs an input NWB path and an output directory"
        )
    nwb_path = Path(argv[0])
    final_dir = Path(argv[1])

    defaults = WriteOpts()
    opts = WriteOpts(
        zstd_level=_int_env(env, "ZARR_WRITER_ZSTD_LEVEL", defaults.zstd_level),
        max_levels=_int_env(env, "ZARR_WRITER_MAX_LEVELS", defaults.max_levels),
        min_bins=_int_env(env, "ZARR_WRITER_MIN_BINS", defaults.min_bins),
        inner_len=_int_env(env, "ZARR_WRITER_INNER_LEN", defaults.inner_len),
        target_shard_bytes=_int_env(
            env, "ZARR_WRITER_TARGET_SHARD_BYTES", defaults.target_shard_bytes
        ),
    )

    staging_raw = env.get("ZARR_WRITER_STAGING_DIR")
    staging_dir = (
        Path(staging_raw)
        if staging_raw is not None
        else final_dir.with_name(final_dir.name + ".staging")
    )

    return Config(
        nwb_path=nwb_path,
        staging_dir=staging_dir,
        final_dir=final_dir,
        opts=opts,
    )


def _int_env(env: Mapping[str, str], key: str, default: int) -> int:
    """Return the integer env value for key, or default if it is unset.

    Raises ValueError (from int) if the value is present but not a valid integer.
    """
    raw = env.get(key)
    return default if raw is None else int(raw)
