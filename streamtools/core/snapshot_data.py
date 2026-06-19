"""
Typed container for precomputed per-snapshot star-cluster data.

All star-indexed fields have shape (n_snaps, n_stars).
All snapshot-indexed-only fields (times, rc3, vc3) have shape (n_snaps,) or
(n_snaps, 3).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class SnapshotData:
    theta: np.ndarray   # (n_snaps, n_stars) angular coord, deg
    phi: np.ndarray     # (n_snaps, n_stars) angular coord, deg
    E: np.ndarray        # (n_snaps, n_stars) binding energy
    v_rad: np.ndarray    # (n_snaps, n_stars) radial velocity, cluster frame
    rotx: np.ndarray     # (n_snaps, n_stars) cartesian, cluster-rotating frame
    roty: np.ndarray     # (n_snaps, n_stars)
    rotz: np.ndarray     # (n_snaps, n_stars)
    Lz: np.ndarray        # (n_snaps, n_stars) per-star z-angular-momentum, cluster frame
    L_cluster_z: np.ndarray  # (n_snaps,) cluster orbital angular momentum z-component
    times: np.ndarray    # (n_snaps,)
    rc3: np.ndarray       # (n_snaps, 3) cluster center position
    vc3: np.ndarray       # (n_snaps, 3) cluster center velocity

    def __post_init__(self):
        n_snaps = len(self.times)
        star_fields = ("theta", "phi", "E", "v_rad", "rotx", "roty", "rotz", "Lz")
        for name in star_fields:
            arr = getattr(self, name)
            if arr.shape[0] != n_snaps:
                raise ValueError(
                    f"SnapshotData.{name} has {arr.shape[0]} snapshots, "
                    f"expected {n_snaps} (from times)"
                )
        if self.L_cluster_z.shape[0] != n_snaps:
            raise ValueError(
                f"SnapshotData.L_cluster_z has {self.L_cluster_z.shape[0]} "
                f"snapshots, expected {n_snaps} (from times)"
            )

    # -----------------------------
    # Convenience accessors
    # -----------------------------
    @property
    def n_snaps(self):
        return len(self.times)

    @property
    def n_stars(self):
        return self.E.shape[1]

    def radius(self):
        """Cartesian radial distance in the cluster-rotating frame, (n_snaps, n_stars)."""
        return np.sqrt(self.rotx**2 + self.roty**2 + self.rotz**2)

    def angular_momentum_z(self):
        """Per-star z-angular-momentum (cluster frame), (n_snaps, n_stars)."""
        return self.Lz

    def prograde_mask(self):
        """
        Bool array (n_snaps, n_stars): True where a star's Lz has the same
        sign as the cluster's orbital Lz at that snapshot (prograde).
        """
        return self.Lz * self.L_cluster_z[:, None] > 0

    def islice(self, i_start, i_end=None):
        """Return a new SnapshotData restricted to snapshots [i_start:i_end]."""
        i_end = self.n_snaps if i_end is None else i_end
        sl = slice(i_start, i_end)
        return SnapshotData(
            theta=self.theta[sl], phi=self.phi[sl], E=self.E[sl],
            v_rad=self.v_rad[sl], rotx=self.rotx[sl], roty=self.roty[sl],
            rotz=self.rotz[sl], Lz=self.Lz[sl], L_cluster_z=self.L_cluster_z[sl],
            times=self.times[sl], rc3=self.rc3[sl], vc3=self.vc3[sl],
        )

    def __len__(self):
        return self.n_snaps
