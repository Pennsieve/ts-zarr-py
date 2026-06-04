# ts-zarr-writer

Producer for **pyramid Zarr v3 "viewer bundles"** from neurophysiology time series.

A bundle is a static directory of files that a browser reads via HTTP range
requests to render arbitrarily wide time windows interactively — by range-fetching
precomputed multi-resolution min/max envelopes instead of raw samples. This project
writes that format. The on-disk format is specified in
[`docs/01-bundle-format.md`](./docs/01-bundle-format.md); the browser reader that
consumes it is in [`docs/02-streaming-client.md`](./docs/02-streaming-client.md).

## Setup

```bash
make venv
source venv/bin/activate
```

## Development

```bash
make test        # run tests
make typecheck   # mypy --strict
make lint        # ruff check --fix + ruff format
make check       # lint check + typecheck + tests
make pre-commit  # install git pre-commit hooks
```

## Stack

- Python 3.12, fully typed (`mypy --strict`), strict `ruff` ruleset.
- Minimal runtime dependencies: `zarr>=3`, `numcodecs`, `numpy`, `pynwb`, `h5py`.
