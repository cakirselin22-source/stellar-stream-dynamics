import numpy as np
from ..physics import dynamics

class Snapshot:
    """
    Physical state of a star cluster at one snapshot in time. Provides methods for cluster-frame      transformations, projections,radial velocities, and energy calculations.
    """

    def __init__(self, rs3, vs3, rc3, vc3, time):
        self.rs3 = np.asarray(rs3)
        self.vs3 = np.asarray(vs3)
        self.rc3 = np.asarray(rc3).flatten()
        self.vc3 = np.asarray(vc3).flatten()
        self.time = float(time)
        
    # -----------------------------
    # Frame transformations
    # -----------------------------
    def cluster_frame(self):
        """
        Return positions and velocities in the cluster COM frame.
        """
        rs = self.rs3 - self.rc3
        vs = self.vs3 - self.vc3
        return rs, vs

    def rotating_basis(self):
        """
        Orthonormal basis (e_r, e_y, e_L) defined by cluster orbit.
        """
        r = self.rc3
        v = self.vc3

        rnorm = np.linalg.norm(r)
        L = np.cross(r, v)
        Lnorm = np.linalg.norm(L)

        e_r = r / rnorm
        e_L = L / Lnorm
        e_y = np.cross(e_L, e_r)

        return e_r, e_y, e_L

    def project_cluster_frame(self, rs=None):
        """
        Project star positions onto rotating cluster basis.
        """
        if rs is None:
            rs, _ = self.cluster_frame()

        e_r, e_y, e_L = self.rotating_basis()

        rotx = rs @ e_r
        roty = rs @ e_y
        rotz = rs @ e_L

        return rotx, roty, rotz
    
    def radial_distance(self, rs=None):
        if rs is None:
            rs, _ = self.cluster_frame()
        return np.linalg.norm(rs, axis=1)
    

    def radial_velocity(self, rs=None, vs=None):
          
        if rs is None or vs is None:
            rs, vs = self.cluster_frame()
    
        r_rel = rs - self.rc3
        v_rel = vs - self.vc3

        # radial unit vectors from cluster center
        rnorm = np.linalg.norm(r_rel, axis=1)[:, None]
        e_r = r_rel / rnorm

        # radial velocity along cluster-centered radial direction
        v_rad = np.sum(v_rel * e_r, axis=1)
        return v_rad
    
    def angular_momentum(self,r, v):
        L = np.cross(r,v)
        return L
    
    def compute_energy(self, frame="binding"):
        """
        Precompute energy of stars in this snapshot.
        Stores in self.E_cluster for fast access.
        """
 
        self.E_cluster = dynamics.calculate_energy(
            self.rs3, self.vs3, self.rc3, self.vc3, self.time, frame
        )
