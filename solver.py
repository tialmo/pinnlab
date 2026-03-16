import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from config import TRAINING
from lapinn_base import LAPINN
from weighted_base import WeightedPINN

from modules.geometry import (
    is_inside_cylinder as _is_inside_cylinder,
    build_interface_pairs as _build_interface_pairs,
)
from modules.training_points import generate_training_points as _generate_training_points
from modules.loss_functions import (
    compute_residual as _compute_residual,
    compute_interface_flux_residual as _compute_interface_flux_residual,
)
from modules.data_io import interpolate_sigma as _interpolate_sigma

_BASE = LAPINN if TRAINING['loss_weighting'] == 'lapinn' else WeightedPINN

class LaplaceCylinderLAPINNSolver(_BASE):
    """
    LA-PINN solver for the 3D Laplace equation with a bipolar DBS electrode:
      - Two cylindrical contacts with Dirichlet BCs (V=1, V=0)
      - Insulating shaft cylinder with Neumann BC (zero normal flux)
      - Optional hemispherical cap at shaft bottom
      - Heterogeneous conductivity σ(x) with interface flux continuity
      - Cubic domain outer boundary also treated as Neumann (insulating)
    """

    def __init__(
        self,
        layers_main=[3, 50, 50, 50, 50, 1],
        layers_lan=[1, 20, 20, 20, 1],
        domain_bounds={'x': [-1, 1], 'y': [-1, 1], 'z': [-1, 1]},
        pde_params=None,
        architecture='encoder',
        use_fourier=False,
        fourier_features=256,
        fourier_sigma=1.0,
        loss_weighting='lapinn',
        classic_weights=None,
    ):
        if pde_params is None:
            pde_params = {
                'contact1_cylinder_radius': 0.3,
                'contact1_cylinder_height': 0.2,
                'contact1_cylinder_center': [0, 0, -0.3],
                'contact1_value': 1.0,
                'contact2_cylinder_radius': 0.3,
                'contact2_cylinder_height': 0.2,
                'contact2_cylinder_center': [0, 0, 0.3],
                'contact2_value': 0.0,
                'neumann_cylinder_radius': 0.7,
                'neumann_cylinder_height': 1.4,
                'neumann_cylinder_center': [0, 0, 0],
                'neumann_value': 0.0,
                'with_half_sphere': True,
                'cube_neumann_value': 0.0,
            }
        super().__init__(
            layers_main, layers_lan, domain_bounds, pde_params,
            architecture=architecture,
            use_fourier=use_fourier,
            fourier_features=fourier_features,
            fourier_sigma=fourier_sigma,
        )

    # ------------------------------------------------------------------
    # Geometry helpers (thin wrappers around module-level functions)
    # ------------------------------------------------------------------

    def is_inside_cylinder(self, x, y, z,
                            radius=None, height=None, center=None,
                            with_half_sphere=False, with_flat_bottom=False):
        """Check whether point (x, y, z) is inside the given cylinder."""
        if radius is None:
            radius = self.pde_params['contact1_cylinder_radius']
        if height is None:
            height = self.pde_params['contact1_cylinder_height']
        if center is None:
            center = self.pde_params['contact1_cylinder_center']
        return _is_inside_cylinder(x, y, z, radius, height, center,
                                   with_half_sphere, with_flat_bottom)

    def build_interface_pairs(self, tol_sigma=1e-12):
        """Build σ-discontinuity pairs and store as self.iface_i/j/n."""
        iface_i, iface_j, iface_n = _build_interface_pairs(
            self.x_r_all, self.sigma_r_all, tol_sigma=tol_sigma
        )
        self.iface_i = iface_i
        self.iface_j = iface_j
        self.iface_n = iface_n

    # ------------------------------------------------------------------
    # Training point generation
    # ------------------------------------------------------------------

    def generate_training_points(self, n_r=10000, n_b_cube=600,
                                  n_b_contact1=400, n_b_contact2=400,
                                  n_b_neumann=400):
        """
        Delegate to modules/training_points and store all results as
        instance attributes.

        Returns the same tuple as the original implementation for API parity.
        """
        data = _generate_training_points(
            pde_params=self.pde_params,
            domain_bounds=self.domain_bounds,
            n_r=n_r,
            n_b_cube=n_b_cube,
            n_b_contact1=n_b_contact1,
            n_b_contact2=n_b_contact2,
            n_b_neumann=n_b_neumann,
        )

        self.x_r       = data['x_r']
        self.x_r_all   = data['x_r_all']
        self.sigma_r   = data['sigma_r']
        self.sigma_r_all = data['sigma_r_all']
        self.iface_i   = data['iface_i']
        self.iface_j   = data['iface_j']
        self.iface_n   = data['iface_n']
        self.x_n       = data['x_n']
        self.g_n       = data['g_n']
        self.x_b_c1    = data['x_b_c1']
        self.u_b_c1    = data['u_b_c1']
        self.x_b_c2    = data['x_b_c2']
        self.u_b_c2    = data['u_b_c2']

        return (self.x_r, self.x_n, self.g_n,
                self.x_b_c1, self.u_b_c1,
                self.x_b_c2, self.u_b_c2)

    # ------------------------------------------------------------------
    # PDE physics (delegate to module-level functions)
    # ------------------------------------------------------------------

    def compute_residual(self, x, sigma=None):
        """Laplacian residual u_xx + u_yy + u_zz at collocation points."""
        return _compute_residual(x, self.main_net)

    def compute_interface_flux_residual(self, xi, xj, si, sj, nij):
        """σ-weighted normal-flux continuity at material interfaces."""
        return _compute_interface_flux_residual(xi, xj, si, sj, nij, self.main_net)

    def interpolate_sigma(self, points):
        """Nearest-neighbour σ lookup at arbitrary query points."""
        return _interpolate_sigma(points, self.x_r_all, self.sigma_r_all)

    # ------------------------------------------------------------------
    # Accuracy evaluation
    # ------------------------------------------------------------------

    def compute_prediction_accuracy(self, n_points=5000):
        """
        Estimate prediction quality by measuring the mean squared Laplacian
        residual over randomly drawn interior points.

        Args:
            n_points: Number of test points

        Returns:
            float: mean squared residual
        """
        x_min, x_max = self.domain_bounds['x']
        y_min, y_max = self.domain_bounds['y']
        z_min, z_max = self.domain_bounds['z']

        n_extra = int(2.0 * n_points)
        x_all = np.random.uniform(x_min, x_max, n_extra)
        y_all = np.random.uniform(y_min, y_max, n_extra)
        z_all = np.random.uniform(z_min, z_max, n_extra)

        with_half_sphere = self.pde_params.get('with_half_sphere', True)

        test_points = []
        for i in range(n_extra):
            in_c1 = self.is_inside_cylinder(
                x_all[i], y_all[i], z_all[i],
                radius=self.pde_params['contact1_cylinder_radius'],
                height=self.pde_params['contact1_cylinder_height'],
                center=self.pde_params['contact1_cylinder_center'],
                with_half_sphere=False)
            in_c2 = self.is_inside_cylinder(
                x_all[i], y_all[i], z_all[i],
                radius=self.pde_params['contact2_cylinder_radius'],
                height=self.pde_params['contact2_cylinder_height'],
                center=self.pde_params['contact2_cylinder_center'],
                with_half_sphere=False)
            in_n = self.is_inside_cylinder(
                x_all[i], y_all[i], z_all[i],
                radius=self.pde_params['neumann_cylinder_radius'],
                height=self.pde_params['neumann_cylinder_height'],
                center=self.pde_params['neumann_cylinder_center'],
                with_half_sphere=with_half_sphere)
            if not (in_c1 or in_c2 or in_n):
                test_points.append([x_all[i], y_all[i], z_all[i]])
                if len(test_points) >= n_points:
                    break

        if len(test_points) < n_points:
            print(f"Warning: Only collected {len(test_points)} test points "
                  f"instead of {n_points}. Consider increasing n_extra.")

        test_points = np.array(test_points)
        test_pts_tf = tf.convert_to_tensor(test_points, dtype=tf.float32)

        sigma_test = tf.convert_to_tensor(
            self.interpolate_sigma(test_points), dtype=tf.float32)

        residual = self.compute_residual(test_pts_tf, sigma=sigma_test)
        return tf.reduce_mean(tf.square(residual)).numpy()
