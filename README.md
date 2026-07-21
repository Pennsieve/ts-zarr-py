# ts-zarr-py

Converts a neurophysiology recording (NWB) into a pyramid Zarr v3 "viewer bundle": a
static directory that a browser reads over HTTP range requests to render any time window
interactively.

A raw recording is hundreds of gigabytes, far more than a client can fetch to render an
overview only ~2000 pixels wide. The writer precomputes multi-resolution min/max envelopes
at ingest, and the reader fetches only the bins a viewport needs. This tool writes that
format; it does no rendering and includes no reader.

## The bundle format

A bundle is plain Zarr v3 with a thin convention on top: no sidecar manifest, no schema
language, no namespace prefix. The only custom part is a few attribute keys in standard Zarr
`attributes`. Because the container is off-the-shelf, any Zarr v3 reader can open it
(`zarrita.js` in the browser, `zarr-python` on the server), and consolidated metadata lets a
client read the whole tree's metadata in one request.

Each channel is a group named by its integer index (`0/`, `1/`, …). The upstream identifier
lives in the group's `id` attribute, and the reader maps it to the index after reading
metadata. One group per channel, rather than a single `(channel, time)` array, means channels
can differ in sample rate and be written or fetched independently.

A **continuous channel** is a pyramid of arrays keyed by level. Level 0 is the raw signal,
shape `(N,)`, `float32`. Each level _k_ ≥ 1 holds interleaved `(min, max)` envelopes of shape
`(N/4ᵏ, 2)`, a 4× decimation of the level below it: level 1 is the min/max over each disjoint
block of 4 raw samples, and level _k_+1 takes the min of the 4 mins and the max of the 4
maxes from level _k_. The pyramid has at most 8 levels, stopping once a level would hold fewer
than ~1024 bins. A trailing partial block of 1–3 elements still produces a final bin, so no
samples are dropped. The fold uses plain (not NaN-aware) min/max, so NaN propagates as
"no data"; the reader treats finite values as "has data".

A **unit (spike) channel** is three flat arrays: `events` (`int64` absolute microseconds),
`units` (`uint8` cluster id per event), and `waveforms` (`float32`, shape
`(n_events, points_per_event)`).

Each channel group carries `id`, `rate_hz`, `start_us`, `kind`, a display `name`, and the
samples' physical `unit` (e.g. `uV`). Each pyramid level carries `period_us`, the microseconds
per bin, which the reader uses to choose a level; the waveform array carries its own
`period_us`, the sample period within a spike. Arrays are Zstd-compressed and sharded with the
ZEP2 ShardingCodec (inner chunk ~256K samples, outer shard ~16 MiB), so the reader fetches
byte ranges within large shard files instead of opening one file per chunk. The whole bundle
is staged in a scratch directory and `os.replace`-renamed onto its final path, so a reader
never sees a half-written bundle.

## Usage

The entry point reads one NWB file and writes one published bundle:

```bash
python -m ts_zarr.main <input.nwb> <output-bundle-dir>
```

Writer settings come from the environment under the `ZARR_WRITER_` prefix, all optional
(they fall back to the format defaults): `ZARR_WRITER_STAGING_DIR` (scratch path for the
atomic publish; defaults next to the output), `ZARR_WRITER_ZSTD_LEVEL`,
`ZARR_WRITER_MAX_LEVELS`, `ZARR_WRITER_MIN_BINS`, `ZARR_WRITER_INNER_LEN`, and
`ZARR_WRITER_TARGET_SHARD_BYTES`.

To run in the container against the mounted `data/input` and `data/output` directories:

```bash
make run        # docker-compose build + up
```

## Architecture

Data flows one way: read → plan → stream/fold → write → publish.

The reader side is isolated behind two `Protocol`s in `protocols.py`,
`ContinuousChannelSource` and `UnitChannelSource`, which expose a channel's metadata and
windowed reads (`read_samples`, `read_events`, and so on). `nwb_reader.py` is the only
concrete adapter. Everything downstream depends on the protocols, not NWB, so the core is
testable with trivial in-memory sources.

`planning.py` and `sizing.py` are the pure decision layer: how many pyramid levels a channel
has, each level's shape and `period_us`, and the chunk and shard shapes for an array.
`fold.py` is the numeric core (the block-of-4 min/max reduction); `streaming.py` drives it
over a source one chunk at a time so memory stays bounded regardless of recording length.
`attrs.py` builds the attribute dicts.

`write_continuous.py` and `write_unit.py` compose those pieces to write one channel, holding
the per-channel logic but no Zarr specifics. All Zarr v3 calls live in `zarr_io.py`, the
single module that imports `zarr`; keeping them there leaves the rest of the package
zarr-agnostic. `bundle.py` orchestrates the whole run (index assignment, per-channel writes,
consolidation, atomic publish), and `main.py` and `config.py` are the thin CLI and
environment-config shell around it.

## Development

The codebase is Python 3.12, fully typed under `mypy --strict`, with a strict `ruff` ruleset
(including pydocstyle). Tests in `tests/` mirror `ts_zarr/` one-to-one.

```bash
make venv        # create the virtualenv and install deps
source venv/bin/activate

make test        # pytest
make typecheck   # mypy --strict
make lint        # ruff check --fix + ruff format (mutating)
make check       # the gate: ruff check + format check + mypy + pytest
make pre-commit  # install the git pre-commit hook
```

Conventions: imports are absolute only (relative imports are banned by ruff); `zarr` is
imported exclusively in `zarr_io.py`; and docstrings follow PEP 257 in plain prose,
documenting units, invariants, and gotchas rather than restating the code. `make check` must
stay green.

## Dependencies

Runtime: `zarr>=3`, `numcodecs`, `numpy`, `pynwb`, `h5py`. Dev/test: `pytest`, `pytest-cov`,
`pytest-mock`, `mypy`, `ruff`, `pre-commit`.
