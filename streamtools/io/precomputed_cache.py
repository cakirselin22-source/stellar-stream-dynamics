import numpy as np
from ..physics import dynamics


def save_snapshot_data(snapshots, filename="snapshots_data.npz"):
    """
    Precompute theta, phi, energy, and radial velocity for every snapshot
    and save to disk, so downstream functions can load instead of
    recomputing from raw positions/velocities.
    """
    theta_list, phi_list, E_list, v_rad_list = [], [], [], []

    for snap in snapshots:
        E = dynamics.calculate_energy(
            snap.rs3, snap.vs3, snap.rc3, snap.vc3, snap.time, "binding"
        )
        rotx, roty, rotz = snap.project_cluster_frame()
        r_perp = np.sqrt(roty**2 + rotz**2)
        theta = np.degrees(np.arctan2(rotx, r_perp))
        phi = np.degrees(np.arctan2(roty, rotx))
        v_rad = snap.radial_velocity(snap.rs3, snap.vs3)

        theta_list.append(theta)
        phi_list.append(phi)
        E_list.append(E)
        v_rad_list.append(v_rad)

    np.savez(filename, theta=theta_list, phi=phi_list, E=E_list, v_rad=v_rad_list)
    print(f"Saved snapshot data to {filename}")


def load_snapshot_data(filename="snapshots_data.npz"):
    """
    Load precomputed theta, phi, energy, and radial velocity arrays.

    Returns
    -------
    theta_list, phi_list, E_list, v_rad_list : arrays, one entry per snapshot
    """
    data = np.load(filename, allow_pickle=True)
    return data["theta"], data["phi"], data["E"], data["v_rad"]
