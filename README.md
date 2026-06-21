# streamtools

A Python toolkit for analyzing and visualizing the tidal stream left behind
by a star cluster dissolving as it orbits a host galaxy. It turns raw
N-body snapshots into the standard diagnostics used to study escaping
stars: binding-energy evolution, escape and "potential escaper" detection,
prograde/retrograde classification, and the angular structure of the
forming stream.

## Contents

- [Physical picture](#physical-picture)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Module reference](#module-reference)
- [Testing](#testing)
- [Changelog and design notes](#changelog-and-design-notes)

## Physical picture

A star cluster moves through a galactic potential that is evolving (NFW halo + Miyamoto-Nagai
disk + a bulge component, see `physics/potentials.py`) and is gradually
stripped of stars as they cross its tidal boundary. Each star carries a
binding energy `E` relative to the cluster; once `E` crosses zero it is
energetically unbound, and once it stays positive for the rest of the
simulation it has *permanently* escaped (some stars cross back and forth
several times before settling one way or the other -- these "potential
escapers" get their own detector, see `energy_oscillation_flips` below).

Most of the geometry is expressed in a frame that rotates with the
cluster's orbit: an orthonormal basis `(e_r, e_y, e_L)` built from the
cluster's instantaneous position and velocity, where `e_r` points away
from the galactic center, `e_L` is along the orbital angular momentum, and
`e_y` completes the right-handed triad. A star's position in this frame
(`rotx`, `roty`, `rotz`) separates "where it sits relative to the orbit"
from "where the orbit happens to be pointing in the galaxy," which is what
makes it possible to stack stars from many snapshots into one coherent
picture of the stream's shape. Escaped stars are further classified as
prograde or retrograde depending on whether their z-angular-momentum has
the same sign as the cluster's orbital angular momentum.

## Architecture

The package is organized as a pipeline, and every analysis/plotting
function downstream of the second stage takes the same input type
(`SnapshotData`) so new figures never need to touch raw snapshots again.

**1. Raw ingestion** (`io.snapshot_reader`, `io.cache`) -- `readsnap` parses
the custom binary snapshot format (dark matter + cluster + star + bulge
blocks) into raw position/velocity arrays for the cluster and its stars at
one point in time. `load_or_read_snapshots` calls this for every snapshot
in a run, wraps each result in a `core.snapshot.Snapshot`, and pickles the
list so re-running ingestion is a cache hit instead of a re-parse.

**2. Per-snapshot physics** (`core.snapshot.Snapshot`) -- a `Snapshot`
knows how to transform itself into the cluster's rest frame and rotating
orbital basis, and how to compute binding energy via
`physics.dynamics.calculate_energy`. This is the only place per-star
raw position/velocity arrays are touched.

**3. Precomputed cache** (`io.precomputed_cache`) -- `save_snapshot_data`
runs the stage-2 physics once for every snapshot and writes the *results*
(`theta`, `phi`, `E`, `v_rad`, `rotx`/`roty`/`rotz`, `Lz`, `L_cluster_z`,
`times`, `rc3`, `vc3`) to a single versioned `.npz` file.
`load_snapshot_data` reads it back into a `core.snapshot_data.SnapshotData`
-- a plain dataclass, no raw per-star vectors, just the derived arrays
everything else needs. This is the expensive step you run once per
simulation; everything after it is array indexing.

**4. Analysis** (`analysis.selection`, `analysis.series`) -- vectorized
helpers that operate on whole `SnapshotData` arrays at once (escape-time
detection, oscillation/"potential escaper" detection, per-property time
series, orbit-frame decomposition) instead of looping over stars in
Python.

**5. Visualization** (`viz.plotting.*`) -- every figure- and
animation-producing function takes a `SnapshotData` (plus, for a few
functions, precomputed `escape_idx`/`tracked_idx` arrays from stage 4) and
calls `matplotlib` directly. Nothing in this layer reads a raw `Snapshot`.

```
snapshot files on disk
        |  io.snapshot_reader.readsnap
        v
io.cache.load_or_read_snapshots  -->  list[core.snapshot.Snapshot]
        |  io.precomputed_cache.save_snapshot_data  (run once)
        v
   <dataset>_data.npz
        |  io.precomputed_cache.load_snapshot_data
        v
   core.snapshot_data.SnapshotData
        |
        +--> analysis.selection / analysis.series  (escape detection, time series)
        |
        v
   viz.plotting.{energy,orbits,angularmomentum,histograms,animation}
        |
        v
   .png / .pdf / .mp4 / .gif figures
```

## Installation

```bash
git clone <this repo>
cd stellar-stream-dynamics
pip install -e .
```

This installs `numpy`, `matplotlib`, `scipy`, and `pillow` (see
`requirements.txt` / `pyproject.toml`). For development, install the
`dev` extra to get `pytest`:

```bash
pip install -e ".[dev]"
```

**ffmpeg.** Most animation functions save `.mp4` via matplotlib's ffmpeg
writer, which requires the `ffmpeg` binary on `PATH` -- it is not a pip
package and is not installed by the steps above. Install it separately
(e.g. `apt install ffmpeg`, `brew install ffmpeg`, or via conda). Functions
that default to a `.gif` outfile, or are called with one, use Pillow
instead and have no such dependency.

## Quickstart

```python
from streamtools.io.cache import load_or_read_snapshots
from streamtools.io.precomputed_cache import save_snapshot_data, load_snapshot_data
from streamtools.analysis import selection, series
from streamtools.viz.plotting import energy, orbits, histograms, animation

# 1. Ingest raw snapshots once (reads from STREAM_DATA_DIR/<run_name>, or
#    ./data/<run_name> if that env var isn't set), caching to disk.
snapshots = load_or_read_snapshots("g656", ifilend=1112, cache_file="g656.pkl")

# 2. Precompute the derived-quantity cache once per dataset.
save_snapshot_data(snapshots, filename="g656_data.npz")

# 3. Everything after this only needs the cache.
data = load_snapshot_data("g656_data.npz")

escape_idx = selection.escape_time_index(data)

orbits.plot_cluster_orbit(data, outfile="cluster_orbit.png")
energy.plot_energyvstime(data, snapshots, i_start=0, i_end=200, E_threshold=0.0,
                          outfile="energyvstime.png")
histograms.histogram_at_escape(data, "theta", escape_idx=escape_idx,
                                outfile="theta_at_escape.png")
histograms.plot_elevation_heatmap(data, outfile="elevation_heatmap.png")

# Animations require ffmpeg on PATH (or a .gif outfile, which uses Pillow).
animation.animate_escape_state(data, rotz_range=0.05, outfile="escape_state.mp4")
```

## Module reference

### `streamtools.core`

- **`Snapshot`** -- one simulation snapshot's raw state (`rs3`, `vs3`,
  `rc3`, `vc3`, `time`). `cluster_frame()` shifts to the cluster's rest
  frame; `rotating_basis()` builds the `(e_r, e_y, e_L)` orbital basis;
  `project_cluster_frame()` projects star positions onto it; `radial_distance()`
  and `radial_velocity()` compute cluster-centric radius/radial velocity;
  `compute_energy()` calls `physics.dynamics.calculate_energy` and caches
  the result on `self.E_cluster`.
- **`SnapshotData`** -- the dataclass everything downstream consumes.
  Fields: `theta`, `phi`, `E`, `v_rad`, `rotx`, `roty`, `rotz`, `Lz`
  (all shape `(n_snaps, n_stars)`), `L_cluster_z`, `times` (shape
  `(n_snaps,)`), `rc3`, `vc3` (shape `(n_snaps, 3)`). Methods: `radius()`,
  `angular_momentum_z()`, `prograde_mask()`, `islice(i_start, i_end)`,
  and the `n_snaps`/`n_stars` properties.

### `streamtools.io`

- **`snapshot_reader.readsnap(dir, ifile, data_dir=...)`** -- parses one
  binary snapshot file, returns `(rs3, vs3, rc3, vc3, time)`.
- **`snapshot_reader.readtout(dir, ifile, data_dir=...)`** -- parses a
  `TOUT*` file recording per-star output times; returns `(ctout, outid)`.
- **`cache.load_or_read_snapshots(snap_dir, ifilend=1111, step=1, cache_file=..., data_dir=...)`**
  -- reads (or loads a pickle of) every snapshot in a run, returns
  `list[Snapshot]`.
- **`precomputed_cache.save_snapshot_data(snapshots, filename=...)`** --
  runs the per-snapshot physics once and writes the versioned `.npz`
  cache (`CACHE_VERSION = 3`).
- **`precomputed_cache.load_snapshot_data(filename=...)`** -- loads that
  cache into a `SnapshotData`; raises `ValueError` if the file predates
  the current schema version.

### `streamtools.physics`

- **`potentials.plummer_potential`**, **`nfw_potential`**,
  **`miyamoto_nagai_potential`** -- host-galaxy potential components
  (bulge, halo, disk).
- **`potentials.cluster_potential`** -- self-gravity of the cluster,
  approximated as a Plummer sphere whose mass/scale are estimated from
  the instantaneous half-mass radius of stars within the cluster core.
- **`dynamics.calculate_energy(rs3, vs3, rc3, vc3, time, frame)`** --
  total per-star energy. `frame="binding"` returns kinetic energy plus
  only the cluster's self-potential (i.e. binding energy *to the
  cluster*, the quantity used everywhere else in the package); any other
  value returns the full energy including the host-galaxy potentials.

### `streamtools.analysis.selection`

- **`select_below_threshold` / `select_above_threshold(data, i_snap, E_threshold)`**
  -- star indices on either side of an energy cut at one snapshot.
- **`sample_indices(candidate_idx, n, rng=None)`** -- random subsample
  without replacement.
- **`tracked_stars_below_threshold(data, i_snap, E_threshold, nstars, rng=None)`**
  -- the two above, composed (used by the `energy.py` plotting functions).
- **`classify_escape_state(E_final, Lz_final, L_cluster_z_final)`** --
  per-star "bound" / "prograde escape" / "retrograde escape" label.
- **`escape_time_index(data)`** -- snapshot index of *permanent* escape
  per star (`-1` if it never permanently escapes within the data).
  Vectorized via a reverse running minimum of `E`, no per-star loop.
- **`unbinding_time_index(data)`** -- snapshot index of the *first*
  `E > 0` crossing per star, regardless of whether it later rebinds.
- **`energy_oscillation_flips(data, min_flips=4)`** -- counts
  bound/unbound sign changes in each star's energy history and flags
  stars with at least `min_flips` ("potential escapers" -- stars that
  flirt with the escape threshold repeatedly before settling down).

### `streamtools.analysis.series`

- **`energy_series` / `radius_series` / `v_rad_series` / `position_series` / `angular_momentum_series(data, tracked_idx, i_start=0, i_end=None)`**
  -- per-tracked-star time series of energy, radius, radial velocity,
  `(rotx, roty, rotz)`, and `(Lz, L_cluster_z)` respectively.
- **`escape_property_values(data, property="energy", escape_idx=None)`**
  -- for every permanently-escaping star, the value of `property`
  (`energy`, `v_rad`, `r`, `theta`, `phi`, or `Lz`) at its escape
  snapshot, plus escape time and star index.
- **`escape_property_by_time_bin(data, property="theta", bin_width=1.0, escape_idx=None)`**
  -- buckets the above into `bin_width`-Gyr escape-time windows.
- **`along_orbit_position(data, i_start=0, i_end=None, nskip=1)`** --
  decomposes each star's position into distance along the cluster's
  velocity direction ("leading/trailing") and perpendicular spread,
  reconstructed from `rotx`/`roty`/`rotz` plus an orbit basis rebuilt
  from `rc3`/`vc3` -- no raw position arrays needed.

### `streamtools.viz.plotting`

`style.savefig(fig, outfile, dpi=200, show=False)` is the shared save
helper used by every function below.

**`energy.py`**

- `plot_energyvstime` -- E(t) and R(t) for a random sample of stars
  starting below an energy threshold.
- `plot_energyvstime_with_orbits` -- adds Lz(t) and v_rad(t) panels, and
  classifies each tracked star's final escape state.
- `energy_vs_radius_grid` -- one E-colored `(rotx, roty)` scatter panel
  per snapshot.
- `energy_vs_radius_tracks` -- R(t) for a fixed set of bound/unbound
  stars.
- `energyxy` -- single-snapshot E-colored `(rotx, roty)` scatter.

**`orbits.py`**

- `plot_cluster_orbit` -- cluster galactocentric radius vs time and its
  xy track; prints orbital eccentricity.
- `plot_cluster_trajectory` -- cluster orbit in three projections (xy, R/r
  vs t, R vs z).
- `plot_star_orbits` -- xy orbits and r(t) for a star subsample, in the
  cluster-rotating frame.
- `plot_cluster_orbits_filtered` -- star orbits masked to a radial/vertical
  selection window.
- `plot_star_orbits_energy_snapshots` -- orbits (xy, r vs t, E vs t) for
  stars selected by an initial energy + radius cut.

**`angularmomentum.py`**

- `angular_momentum_stars` -- Lz(t) for tracked stars, cluster `|L|(t)`,
  and E(t) for the same stars, three panels.
- `plot_Lz_nstars` -- Lz(t) for the first N stars by raw index.

**`histograms.py`**

- `histogram_at_escape(data, property="energy", ...)` -- distribution of
  a property at each star's escape snapshot.
- `histogram_delay_time` -- distribution of time between first unbinding
  and permanent escape.
- `histogram_escape_time` -- escape-time distribution plus cumulative
  escape count.
- `histogram_energy_single_snapshot` -- bound or unbound energy
  distribution at one snapshot.
- `histogram_by_time_bin_grid(data, property="theta", ...)` -- grid of
  per-escape-time-window histograms, with optional peak annotation.
- `animate_histogram_by_time_bin` -- animated version of the grid above,
  one frame per time window.
- `histogram_evolution_peaks` -- tracks the dominant escape-angle peak(s)
  over time.
- `histogram_by_radius_cut` -- overlaid histograms split by escape-radius
  band.
- `plot_elevation_heatmap` -- time vs out-of-orbital-plane-angle density
  heatmap.
- `animate_property_histograms(data, mode="newly_escaped"|"unbound", ...)`
  -- 4-panel (theta/phi/E/v_rad) histogram animation.
- `leading_trailing_evolution` -- animated along-orbit position
  histogram + scatter, frame per snapshot.

**`animation.py`**

- `animate_energy_scatter` -- E-colored spatial scatter over time, two
  zoom levels, symmetric-log or linear color scale.
- `animate_energy_groups` -- spatial evolution of bound/transition/unbound
  groups fixed at t=0.
- `animate_energy_histogram_grid` -- combined spatial scatter + E/Lz/r
  histograms for the same three groups.
- `animate_escape_state` -- spatial scatter colored by bound / potential
  escaper / permanently escaped.
- `animate_escape_state_3d` -- 3-D rotating version of the above, plus an
  edge-on 2-D slice.
- `animate_potential_escapers` -- highlights oscillating ("potential
  escaper") stars against the full field, colored by instantaneous
  bound/unbound state.

## Testing

`tests/test_smoke.py` builds one deterministic synthetic `SnapshotData`
(seeded `np.random.default_rng(42)`, with always-escaped, always-bound,
and oscillating stars built in as edge cases) and exercises every public
analysis and plotting function against it, asserting that figures/
animations are actually written and non-empty. It is integration/
regression coverage for the pipeline wiring, not a check of simulation
physics.

```bash
pip install -e ".[dev]"
pytest -q
```

Tests that save `.mp4` are skipped automatically if `ffmpeg` isn't on
`PATH`; the GIF-export test always runs.

