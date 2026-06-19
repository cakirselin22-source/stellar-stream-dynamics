import numpy as np
from .potentials import (
    plummer_potential, nfw_potential, miyamoto_nagai_potential, cluster_potential
)


def calculate_energy(rs3, vs3, rc3, vc3, time, frame):
    """
    Compute total energy of stars in a galactic frame.
    """
    rs3, vs3 = np.asarray(rs3), np.asarray(vs3)
    rc3, vc3 = np.asarray(rc3), np.asarray(vc3)

    rs_rel = rs3 - rc3
    vs_rel = vs3 - vc3

    KE = 0.5 * np.sum(vs_rel**2, axis=1)

    P_plum = plummer_potential(rs3)
    P_NFW  = nfw_potential(rs3, time)
    P_MN   = miyamoto_nagai_potential(rs3)
    P_cluster = cluster_potential(rs_rel)

    E_stars = P_cluster + KE + P_plum + P_NFW + P_MN

    if frame == "binding":
        return P_cluster + KE
    else:
        return E_stars
