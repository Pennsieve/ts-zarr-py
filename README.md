# ts-zarr-writer

Turns a neurophysiology recording (NWB) into a **pyramid Zarr v3 "viewer bundle"** —
a static directory of files that a browser can read over HTTP range requests to render
arbitrarily wide time windows interactively.

The problem this solves: a raw recording at 32 kHz × 24 channels × 24 hours is hundreds
of gigabytes. Drawing a full-recording overview into ~2000 screen pixels can't fetch raw
samples — the math doesn't work. So at ingest we precompute multi-resolution min/max
envelopes; the reader then range-fetches only the bins a viewport actually needs. This
tool writes that precomputed format. It does no rendering and ships no reader.

## The bundle format

A bundle is plain Zarr v3 with a thin convention layered on top — no sidecar manifest, no
schema language, no namespace prefix. The only custom surface is a handful of attribute
keys stored in standard Zarr `attributes`. The container being off-the-shelf is the point:
any Zarr v3 reader (`zarrita.js` in the browser, `zarr-python` on the server) can open it,
and consolidated metadata lets a client pull the whole tree's metadata in one request.

Channels are stored as digit-named groups (`0/`, `1/`, …). The digit is opaque — the real
upstream identifier lives in the group's `id` attribute, and the reader builds an
`id → index` map after reading metadata. Per-channel groups (rather than one big
`(channel, time)` array) let channels carry different sample rates and let a recording be
written or fetched one channel at a time.

A **continuous channel** is a pyramid of arrays keyed by level. Level 0 is the raw signal,
shape `(N,)`, `float32`. Level _k_ ≥ 1 holds interleaved `(min, max)` envelopes, shape
`(N/4ᵏ, 2)` — each level is a 4× decimation of the one below: level 1 is the min/max over
each disjoint block of 4 raw samples; level _k_+1 takes the min of the 4 mins and the max
of the 4 maxes from level _k_. Up to 8 levels, stopping once a level would hold fewer than
~1024 bins. A trailing partial block of 1–3 elements still produces a final bin, so no
samples are dropped, and the fold uses plain (not NaN-aware) min/max so that NaN propagates
as "no data" — the reader treats finite values as "has data".

A **unit (spike) channel** is three flat arrays: `events` (`int64` absolute microseconds),
`units` (`uint8` cluster id per event), and `waveforms` (`float32`, shape
`(n_events, points_per_event)`).

Each channel group carries `{id, rate_hz, start_us, kind}`; each pyramid level and the
waveform array carry `{period_us}` (microseconds per bin), which drives the reader's level
selection. Arrays are Zstd-compressed and sharded via the ZEP2 ShardingCodec (inner chunk
~256K samples, outer shard ~16 MiB) so each channel-level is roughly one file. The whole
bundle is staged in a scratch directory and `os.replace`-renamed onto its final path, so a
reader never observes a half-written bundle.

## Usage

The entry point reads one NWB file and writes one published bundle:

```bash
python -m processor.main <input.nwb> <output-bundle-dir>
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

Data flows in one direction: **read → plan → stream/fold → write → publish.**

The reader side is isolated behind two `Protocol`s in `protocols.py` —
`ContinuousChannelSource` and `UnitChannelSource` — that expose a channel's metadata and
windowed reads (`read_samples`, `read_events`, …). `nwb_reader.py` is the only concrete
adapter; everything downstream depends on the protocols, not on NWB, so the core is
testable with trivial in-memory sources.

`planning.py` and `sizing.py` are the pure decision layer: how many pyramid levels a
channel gets, each level's shape and `period_us`, and the chunk/shard shapes for an array.
`fold.py` is the numeric core — the block-of-4 min/max reduction — and `streaming.py`
drives it over a source one chunk at a time so memory stays bounded regardless of recording
length. `attrs.py` builds the attribute dicts (the custom format surface).

`write_continuous.py` and `write_unit.py` compose those pieces to write one channel; they
hold the per-channel logic but no Zarr specifics. **All Zarr v3 calls live in `zarr_io.py`**
— it is the single module that imports `zarr`, which keeps the rest of the package
zarr-agnostic and the API surface in one place. `bundle.py` orchestrates the whole run
(index assignment, per-channel writes, consolidation, atomic publish), and `main.py` /
`config.py` are the thin CLI and environment-config shell around it.

## Development

The codebase is Python 3.12, fully typed under `mypy --strict`, with a strict `ruff`
ruleset (including pydocstyle). Tests in `tests/` mirror `processor/` one-to-one.

```bash
make venv        # create the virtualenv and install deps
source venv/bin/activate

make test        # pytest
make typecheck   # mypy --strict
make lint        # ruff check --fix + ruff format (mutating)
make check       # the gate: ruff check + format check + mypy + pytest
make pre-commit  # install the git pre-commit hook
```

A few conventions worth knowing before editing: imports are absolute only (relative imports
are banned by ruff); `zarr` is imported exclusively in `zarr_io.py`; and docstrings follow
PEP 257 in plain prose — they document units, invariants, and gotchas, never restating the
code. `make check` must stay green.

## Dependencies

Runtime: `zarr>=3`, `numcodecs`, `numpy`, `pynwb`, `h5py`. Dev/test: `pytest`,
`pytest-cov`, `pytest-mock`, `mypy`, `ruff`, `pre-commit`.
