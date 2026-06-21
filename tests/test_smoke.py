"""
End-to-end smoke tests for streamtools.

These build one synthetic SnapshotData fixture (deterministic, seeded RNG)
and exercise every analysis helper plus every plotting/animation function
against it. The goal is regression coverage for the cache -> SnapshotData ->
analysis -> viz pipeline, not numerical correctness of the underlying
physics -- the fixture is a stand-in cluster orbit + escaping star
population, not a real simulation.

Animation tests that call FuncAnimation.save(writer="ffmpeg") are skipped
automatically if the ffmpeg binary isn't on PATH. The GIF-export path
(Pillow) has no such dependency and always runs.

Run with: pytest -q
"""
import shutil

import numpy as np
import pytest

from streamtools.core.snapshot_data import SnapshotData
from streamtools.analysis import selection, series
from streamtools.viz.plotting import energy, orbits, angularmomentum, animation, histograms

HAS_FFMPEG = shutil.which("ffmpeg") is not None
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")


# --------------------------------------------------------------------------
# Fixture: synthetic SnapshotData standing in for a real precomputed cache.
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(42)
    n_snaps, n_stars = 20, 150
    times = np.linspace(0.0, 10.0, n_snaps)

    # Cluster on a mildly precessing, mildly eccentric orbit.
    omega = 0.6
    radius_c = 8 + 2 * np.sin(0.3 * times)
    rc3 = np.stack(
        [radius_c * np.cos(omega * times), radius_c * np.sin(omega * times), 0.1 * np.sin(0.2 * times)],
        axis=1,
    )
    vc3 = np.gradient(rc3, times, axis=0)

    # Stars drift from a tight clump (small rotx/roty/rotz) into a spread
    # stream as time advances.
    rotx = rng.normal(0, 0.01, size=(n_snaps, n_stars))
    roty = rng.normal(0, 0.01, size=(n_snaps, n_stars))
    rotz = rng.normal(0, 0.01, size=(n_snaps, n_stars))
    spread = np.linspace(0.01, 1.0, n_snaps)[:, None]
    rotx = rotx * (1 + spread * 5)
    roty = roty * (1 + spread * 5)
    rotz = rotz * (1 + spread * 2)

    v_rad = rng.normal(0, 1.0, size=(n_snaps, n_stars)) * (1 + spread)

    r_perp = np.sqrt(roty**2 + rotz**2)
    theta = np.degrees(np.arctan2(rotx, r_perp))
    phi = np.degrees(np.arctan2(roty, rotx))

    # Most stars flip from bound (E<0) to escaped (E>0) at a random time;
    # a subset oscillates repeatedly across E=0; two are pinned as the
    # always-escaped / always-bound edge cases.
    escape_time_frac = rng.uniform(0.1, 0.95, size=n_stars)
    base_E = -2.0 + 4.0 * (np.linspace(0, 1, n_snaps)[:, None] > escape_time_frac[None, :])
    E = base_E + rng.normal(0, 0.3, size=(n_snaps, n_stars))
    osc_stars = rng.choice(n_stars, size=10, replace=False)
    E[:, osc_stars] = np.sin(np.linspace(0, 8 * np.pi, n_snaps))[:, None] * 0.5 + rng.normal(
        0, 0.1, size=(n_snaps, 10)
    )
    E[:, 0] = 1.0  # always escaped
    E[:, 1] = -1.0  # always bound

    Lz = rng.normal(0, 1.0, size=(n_snaps, n_stars)) + 0.3
    L_cluster_z = np.cross(rc3, vc3)[:, 2]

    return SnapshotData(
        theta=theta, phi=phi, E=E, v_rad=v_rad,
        rotx=rotx, roty=roty, rotz=rotz,
        Lz=Lz, L_cluster_z=L_cluster_z,
        times=times, rc3=rc3, vc3=vc3,
    )


@pytest.fixture(scope="module")
def escape_idx(data):
    return selection.escape_time_index(data)


def _assert_nonempty(path):
    assert path.exists(), f"{path} was not created"
    assert path.stat().st_size > 0, f"{path} is empty"


# --------------------------------------------------------------------------
# core.SnapshotData
# --------------------------------------------------------------------------

def test_snapshot_data_basics(data):
    assert data.radius().shape == data.E.shape
    assert data.angular_momentum_z().shape == data.E.shape
    assert data.prograde_mask().dtype == bool
    sub = data.islice(0, 10)
    assert sub.n_snaps == 10


# --------------------------------------------------------------------------
# analysis.selection
# --------------------------------------------------------------------------

def test_escape_and_unbinding_indices(data):
    escape_idx = selection.escape_time_index(data)
    unbind_idx = selection.unbinding_time_index(data)
    assert escape_idx.shape == (data.n_stars,)
    assert unbind_idx.shape == (data.n_stars,)
    assert (escape_idx >= 0).any(), "expected at least one permanently escaped star"
    assert (escape_idx == -1).any(), "expected at least one never-escaped star"


def test_energy_oscillation_flips(data):
    osc_idx, flip_counts = selection.energy_oscillation_flips(data, min_flips=4)
    assert len(osc_idx) > 0, "expected at least one oscillating star in synthetic fixture"
    assert flip_counts.shape == (data.n_stars,)


def test_tracked_stars_below_threshold(data):
    rng = np.random.default_rng(0)
    idx = selection.tracked_stars_below_threshold(data, 0, 0.0, 10, rng=rng)
    assert len(idx) <= 10


def test_classify_escape_state(data):
    state = selection.classify_escape_state(data.E[-1, :5], data.Lz[-1, :5], data.L_cluster_z[-1])
    assert len(state) == 5


# --------------------------------------------------------------------------
# analysis.series
# --------------------------------------------------------------------------

def test_basic_series(data):
    idx = np.arange(10)
    _, E = series.energy_series(data, idx)
    assert E.shape[1] == 10
    _, r = series.radius_series(data, idx)
    assert r.shape[1] == 10
    _, v_rad = series.v_rad_series(data, idx)
    assert v_rad.shape[1] == 10
    _, Lz, L_cluster_z = series.angular_momentum_series(data, idx)
    assert Lz.shape[1] == 10 and len(L_cluster_z) == len(Lz)
    _, rotx, roty, rotz = series.position_series(data, idx)
    assert rotx.shape[1] == roty.shape[1] == rotz.shape[1] == 10


@pytest.mark.parametrize("prop", ["energy", "v_rad", "r", "theta", "phi", "Lz"])
def test_escape_property_values(data, escape_idx, prop):
    values, times, star_idx = series.escape_property_values(data, prop, escape_idx)
    assert len(values) == len(times) == len(star_idx)


def test_escape_property_by_time_bin(data, escape_idx):
    bin_edges, values_per_bin, labels = series.escape_property_by_time_bin(data, "theta", 1.0, escape_idx)
    assert len(values_per_bin) == len(labels) == len(bin_edges) - 1


def test_along_orbit_position(data):
    times, s_orbit, r_perp = series.along_orbit_position(data, 0, data.n_snaps - 1, nskip=2)
    assert s_orbit.shape == r_perp.shape
    assert s_orbit.shape[0] == len(times)


# --------------------------------------------------------------------------
# viz.plotting.energy / orbits / angularmomentum (pre-existing modules,
# exercised here as integration regression coverage)
# --------------------------------------------------------------------------

def test_energy_plots(data, tmp_path):
    snaps_stub = list(range(data.n_snaps))
    i_mid = data.n_snaps - 5
    out1 = tmp_path / "e1.png"
    energy.plot_energyvstime(data, snaps_stub, 0, i_mid, 0.0, nstars=5, outfile=str(out1))
    _assert_nonempty(out1)

    out2 = tmp_path / "e2.png"
    _, escape_type, tracked_idx = energy.plot_energyvstime_with_orbits(
        data, snaps_stub, 0, i_mid, 0.0, nstars=5, outfile=str(out2)
    )
    _assert_nonempty(out2)
    assert len(escape_type) == len(tracked_idx)


def test_orbit_plots(data, tmp_path):
    out = tmp_path / "orbit.png"
    _, e = orbits.plot_cluster_orbit(data, outfile=str(out))
    _assert_nonempty(out)
    assert 0 <= e <= 1

    out2 = tmp_path / "star_orbits.pdf"
    orbits.plot_star_orbits(data, data.n_snaps - 1, nskip=25, outfile=str(out2))
    _assert_nonempty(out2)


def test_angular_momentum_plots(data, tmp_path):
    rng = np.random.default_rng(1)
    out = tmp_path / "angmom.png"
    angularmomentum.angular_momentum_stars(data, n_track=8, outfile=str(out), rng=rng)
    _assert_nonempty(out)

    out2 = tmp_path / "lz_nstars.png"
    angularmomentum.plot_Lz_nstars(data, nstars=5, outfile=str(out2))
    _assert_nonempty(out2)


# --------------------------------------------------------------------------
# viz.plotting.histograms (new module)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("prop", ["energy", "theta", "Lz"])
def test_histogram_at_escape(data, escape_idx, tmp_path, prop):
    out = tmp_path / f"hist_{prop}.png"
    histograms.histogram_at_escape(data, prop, escape_idx=escape_idx, outfile=str(out))
    _assert_nonempty(out)


def test_histogram_delay_and_escape_time(data, escape_idx, tmp_path):
    out1 = tmp_path / "delay.png"
    histograms.histogram_delay_time(data, bins=8, outfile=str(out1))
    _assert_nonempty(out1)

    out2 = tmp_path / "esc_time.png"
    histograms.histogram_escape_time(data, bins=10, escape_idx=escape_idx, outfile=str(out2))
    _assert_nonempty(out2)


def test_histogram_energy_single_snapshot(data, tmp_path):
    out = tmp_path / "energy_single_snap.png"
    histograms.histogram_energy_single_snapshot(data, data.n_snaps - 5, bins=20, bound=False, outfile=str(out))
    _assert_nonempty(out)


def test_histogram_by_time_bin_grid(data, escape_idx, tmp_path):
    out = tmp_path / "grid.png"
    histograms.histogram_by_time_bin_grid(data, "theta", bin_width=2.0, escape_idx=escape_idx, outfile=str(out))
    _assert_nonempty(out)


def test_histogram_evolution_peaks(data, escape_idx, tmp_path):
    out = tmp_path / "peaks.png"
    fig, t_centers, peak_positions_all = histograms.histogram_evolution_peaks(
        data, n_time_bins=8, bins=30, min_stars=3, escape_idx=escape_idx, outfile=str(out)
    )
    _assert_nonempty(out)
    assert len(t_centers) == len(peak_positions_all)


def test_histogram_by_radius_cut(data, escape_idx, tmp_path):
    out = tmp_path / "by_radius.png"
    histograms.histogram_by_radius_cut(data, [0.01, 0.05, 0.2], "theta", escape_idx=escape_idx, outfile=str(out))
    _assert_nonempty(out)


def test_plot_elevation_heatmap(data, tmp_path):
    out = tmp_path / "elevation.png"
    histograms.plot_elevation_heatmap(data, n_bins=30, outfile=str(out))
    _assert_nonempty(out)


def test_animate_histogram_by_time_bin_gif(data, escape_idx, tmp_path):
    # GIF export goes through Pillow, not ffmpeg, so this always runs.
    out = tmp_path / "serial.gif"
    histograms.animate_histogram_by_time_bin(
        data, "theta", bin_width=2.0, fps=2, escape_idx=escape_idx, outfile=str(out)
    )
    _assert_nonempty(out)


@needs_ffmpeg
@pytest.mark.parametrize("mode", ["newly_escaped", "unbound"])
def test_animate_property_histograms(data, tmp_path, mode):
    out = tmp_path / f"prop_{mode}.mp4"
    histograms.animate_property_histograms(
        data, mode=mode, i_start=0, i_end=data.n_snaps - 1, fps=5, outfile=str(out)
    )
    _assert_nonempty(out)


@needs_ffmpeg
def test_leading_trailing_evolution(data, tmp_path):
    out = tmp_path / "leading_trailing.mp4"
    histograms.leading_trailing_evolution(data, 0, data.n_snaps - 1, nskip=3, fps=5, outfile=str(out))
    _assert_nonempty(out)


# --------------------------------------------------------------------------
# viz.plotting.animation (new module)
# --------------------------------------------------------------------------

@needs_ffmpeg
def test_animate_energy_scatter(data, tmp_path):
    out = tmp_path / "energy_scatter.mp4"
    animation.animate_energy_scatter(data, (-3, 3), 0.05, fps=5, frame_step=3, outfile=str(out))
    _assert_nonempty(out)


@needs_ffmpeg
def test_animate_energy_groups(data, tmp_path):
    out = tmp_path / "energy_groups.mp4"
    animation.animate_energy_groups(data, fps=5, outfile=str(out))
    _assert_nonempty(out)


@needs_ffmpeg
def test_animate_energy_histogram_grid(data, tmp_path):
    out = tmp_path / "energy_hist_grid.mp4"
    animation.animate_energy_histogram_grid(data, bound_nskip=20, fps=5, outfile=str(out))
    _assert_nonempty(out)


@needs_ffmpeg
def test_animate_escape_state(data, tmp_path):
    out = tmp_path / "escape_state.mp4"
    animation.animate_escape_state(data, 0.05, fps=5, frame_step=3, outfile=str(out))
    _assert_nonempty(out)


@needs_ffmpeg
def test_animate_escape_state_3d(data, tmp_path):
    out = tmp_path / "escape_state_3d.mp4"
    animation.animate_escape_state_3d(data, 0.05, fps=5, frame_step=5, outfile=str(out))
    _assert_nonempty(out)


@needs_ffmpeg
def test_animate_potential_escapers(data, tmp_path):
    out = tmp_path / "potential_escapers.mp4"
    animation.animate_potential_escapers(data, min_flips=4, fps=5, outfile=str(out))
    _assert_nonempty(out)
