"""Run configuration assembled from the environment and command line."""

import os
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

    argv holds the arguments after the program name. When it carries two
    positionals they are the input NWB path and the final output directory.
    When argv is empty the input and output come instead from the INPUT_DIR and
    OUTPUT_DIR environment variables (the directory convention used under
    Pennsieve): INPUT_DIR is scanned for exactly one *.nwb file and OUTPUT_DIR
    is the final output directory. The writer settings and the staging
    directory come from env under the ZARR_WRITER_ prefix
    (ZARR_WRITER_STAGING_DIR, ZARR_WRITER_ZSTD_LEVEL, ZARR_WRITER_MAX_LEVELS,
    ZARR_WRITER_MIN_BINS, ZARR_WRITER_INNER_LEN, ZARR_WRITER_TARGET_SHARD_BYTES).

    Unset settings fall back to the WriteOpts defaults; an unset staging
    directory derives from final_dir. Raises ValueError on a bad invocation:
    exactly one positional, no positionals with INPUT_DIR/OUTPUT_DIR unset, an
    INPUT_DIR holding zero or several *.nwb files, or any non-integer env value.
    """
    nwb_path, final_dir = _resolve_paths(env, argv)

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


def _resolve_paths(
    env: Mapping[str, str], argv: Sequence[str]
) -> tuple[Path, Path]:
    """Return the (input NWB, final output dir) paths from argv or env.

    Two positionals win outright. With no positionals the paths come from the
    INPUT_DIR/OUTPUT_DIR directory convention, INPUT_DIR being scanned for the
    single *.nwb file it must contain; the bundle is published to a directory
    inside OUTPUT_DIR named after the input stem (session.nwb -> session.zarr).
    Raises ValueError on any other shape.
    """
    positional_count = 2
    if len(argv) >= positional_count:
        return Path(argv[0]), Path(argv[1])
    if len(argv) == 1:
        raise ValueError(
            "load_config needs an input NWB path and an output directory"
        )

    input_dir = env.get("INPUT_DIR")
    output_dir = env.get("OUTPUT_DIR")
    if input_dir is None or output_dir is None:
        raise ValueError(
            "load_config needs two positionals or INPUT_DIR and OUTPUT_DIR"
        )
    # The bundle is a named directory inside OUTPUT_DIR rather than OUTPUT_DIR
    # itself: atomic publish renames the final path, which cannot target a
    # mount point such as the bare OUTPUT_DIR volume. It is named after the
    # input stem (session.nwb -> session.zarr).
    nwb_path = _sole_nwb(Path(input_dir))
    return nwb_path, Path(output_dir) / f"{nwb_path.stem}.zarr"


def _sole_nwb(input_dir: Path) -> Path:
    """Return the single *.nwb file in input_dir.

    Raises ValueError unless exactly one *.nwb file is present.
    """
    nwbs = sorted(
        Path(entry.path)
        for entry in os.scandir(input_dir)
        if entry.is_file() and entry.name.lower().endswith(".nwb")
    )
    if len(nwbs) != 1:
        raise ValueError(
            f"expected exactly one .nwb file in {input_dir}, found {len(nwbs)}"
        )
    return nwbs[0]


def _int_env(env: Mapping[str, str], key: str, default: int) -> int:
    """Return the integer env value for key, or default if it is unset.

    Raises ValueError (from int) if the value is present but not a valid integer.
    """
    raw = env.get(key)
    return default if raw is None else int(raw)
