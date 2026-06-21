"""Precompute and cache per-snapshot derived quantities.

Workflow
--------
    snapshots = load_or_read_snapshots("g656")
    save_snapshot_data(snapshots, filename="g656_data.npz")   # run once per dataset
    data = load_snapshot_data("g656_data.npz")                # SnapshotData, used by everything after
"""
import numpy as np

from ..physics import dynamics
from ..core.snapshot_data import SnapshotData

# Bump this if the cache schema changes shape/fields, so stale .npz files
# fail loudly instead of silently missing fields.
CACHE_VERSION = 3


def save_snapshot_data(snapshots, filename="snapshots_data.npz"):
    """
    Precompute, for every snapshot:
      - E          : binding energy per star
      - theta, phi : angular coordinates (deg) in the cluster-rotating frame
      - rotx/roty/rotz : cartesian projection onto the cluster-rotating frame
      - v_rad      : radial velocity per star (cluster frame)
      - Lz         : per-star z-angular-momentum (cluster frame)
      - L_cluster_z : cluster orbital angular momentum z-component (scalar per snapshot)
      - time       : snapshot time [Gyr]
      - rc3, vc3   : cluster center position/velocity

    and save to disk, so downstream code loads instead of recomputing from
    raw positions/velocities.
    """
    theta_list, phi_list, E_list, v_rad_list = [], [], [], []
    rotx_list, roty_list, rotz_list = [], [], []
    Lz_list, L_cluster_z_list = [], []
    times, rc3_list, vc3_list = [], [], []

    for snap in snapshots:
        E = dynamics.calculate_energy(
            snap.rs3, snap.vs3, snap.rc3, snap.vc3, snap.time, "binding"
        )
        rotx, roty, rotz = snap.project_cluster_frame()
        r_perp = np.sqrt(roty**2 + rotz**2)
        theta = np.degrees(np.arctan2(rotx, r_perp))
        phi = np.degrees(np.arctan2(roty, rotx))
        v_rad = snap.radial_velocity(snap.rs3, snap.vs3)

        rs_rel = snap.rs3 - snap.rc3
        vs_rel = snap.vs3 - snap.vc3
        Lz = np.cross(rs_rel, vs_rel)[:, 2]
        L_cluster_z = np.cross(snap.rc3, snap.vc3)[2]

        theta_list.append(theta)
        phi_list.append(phi)
        E_list.append(E)
        v_rad_list.append(v_rad)
        rotx_list.append(rotx)
        roty_list.append(roty)
        rotz_list.append(rotz)
        Lz_list.append(Lz)
        L_cluster_z_list.append(L_cluster_z)
        times.append(snap.time)
        rc3_list.append(snap.rc3)
        vc3_list.append(snap.vc3)

    np.savez(
        filename,
        version=CACHE_VERSION,
        theta=theta_list,
        phi=phi_list,
        E=E_list,
        v_rad=v_rad_list,
        rotx=rotx_list,
        roty=roty_list,
        rotz=rotz_list,
        Lz=Lz_list,
        L_cluster_z=L_cluster_z_list,
        times=times,
        rc3=rc3_list,
        vc3=vc3_list,
    )
    print(f"Saved snapshot data ({len(snapshots)} snapshots) to {filename}")


def load_snapshot_data(filename="snapshots_data.npz"):
    """
    Load precomputed per-snapshot arrays and wrap them in a SnapshotData
    container.

    Returns
    -------
    SnapshotData
    """
    data = np.load(filename, allow_pickle=True)

    version = int(data["version"]) if "version" in data else 1
    if version < CACHE_VERSION:
        raise ValueError(
            f"{filename} was built with cache schema v{version}, but this "
            f"code expects v{CACHE_VERSION} (missing rotx/roty/rotz/times/rc3"
            "/Lz/L_cluster_z). Re-run save_snapshot_data() to regenerate it."
        )

    return SnapshotData(
        theta=np.asarray(data["theta"]),
        phi=np.asarray(data["phi"]),
        E=np.asarray(data["E"]),
        v_rad=np.asarray(data["v_rad"]),
        rotx=np.asarray(data["rotx"]),
        roty=np.asarray(data["roty"]),
        rotz=np.asarray(data["rotz"]),
        Lz=np.asarray(data["Lz"]),
        L_cluster_z=np.asarray(data["L_cluster_z"]),
        times=np.asarray(data["times"]),
        rc3=np.asarray(data["rc3"]),
        vc3=np.asarray(data["vc3"]),
    )
