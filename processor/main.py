"""Command-line entry point: NWB file in, published bundle out."""

import logging
import os
import sys
from collections.abc import Sequence

from pynwb import NWBHDF5IO

from processor.bundle import write_bundle
from processor.config import load_config
from processor.nwb_reader import build_sources_from_nwb

logger = logging.getLogger(__name__)


def main(argv: Sequence[str]) -> int:
    """Run one bundle build from the command line and return an exit code.

    Resolves a Config from the process environment and argv, opens the NWB file
    it names, discovers the channel sources, writes and atomically publishes the
    viewer bundle, and logs progress. Returns 0 on success and a nonzero exit
    code on a handled failure (a bad invocation or an unreadable input).
    """
    try:
        cfg = load_config(os.environ, argv)
    except ValueError:
        logger.exception("invalid invocation")
        return 2

    try:
        # The HDF5 file must stay open while the sources stream their data, so
        # discovery and the whole write happen inside the context manager.
        with NWBHDF5IO(str(cfg.nwb_path), mode="r") as io:
            nwbfile = io.read()
            continuous, units = build_sources_from_nwb(nwbfile)
            logger.info(
                "writing %d continuous + %d unit channels to %s",
                len(continuous),
                len(units),
                cfg.final_dir,
            )
            write_bundle(
                continuous,
                units,
                staging_dir=cfg.staging_dir,
                final_dir=cfg.final_dir,
                opts=cfg.opts,
            )
    except (OSError, ValueError):
        logger.exception("failed to build bundle from %s", cfg.nwb_path)
        return 1

    logger.info("published bundle to %s", cfg.final_dir)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main(sys.argv[1:]))
