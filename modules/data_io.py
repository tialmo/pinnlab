import numpy as np
import meshio
import tensorflow as tf
from scipy.spatial import KDTree


def load_fem_data(mesh_file_path, pde_params, is_inside_cylinder_fn,
                  n_sample=None, scale_factor=1.0):
    """
    Load FEM solution from a VTU file, normalize coordinates to [-1, 1],
    filter out points inside the electrode/shaft cylinders, and return
    both numpy arrays and TF tensors.

    Args:
        mesh_file_path:        Path to the .vtu mesh file
        pde_params:            Dict of geometry parameters (cylinder specs)
        is_inside_cylinder_fn: Callable with signature
                               fn(x, y, z, radius, height, center,
                                  with_half_sphere, with_flat_bottom) -> bool
        n_sample:              Max number of points to keep (None = keep all)
        scale_factor:          Multiplicative scale applied to potential values

    Returns:
        scaled_points:  (N, 3) numpy array of filtered, normalized coordinates
        potential_values: (N,) numpy array of filtered potential values
        x_d:            TF constant of scaled_points
        u_d:            TF constant of potential_values reshaped to (N, 1)
    """
    mesh = meshio.read(mesh_file_path)

    # Normalize spatial coordinates to [-1, 1]
    points = mesh.points
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    normalized_points = (points - min_coords) / (max_coords - min_coords)
    scaled_points = normalized_points * 2 - 1

    potential_values = mesh.point_data['potential_real'] * scale_factor

    # Filter out points that lie inside electrode or shaft cylinders
    valid_points, valid_values = [], []
    for i in range(len(scaled_points)):
        x, y, z = scaled_points[i]

        in_contact1 = is_inside_cylinder_fn(x, y, z,
            radius=pde_params['contact1_cylinder_radius'],
            height=pde_params['contact1_cylinder_height'],
            center=pde_params['contact1_cylinder_center'],
            with_half_sphere=False, with_flat_bottom=False)

        in_contact2 = is_inside_cylinder_fn(x, y, z,
            radius=pde_params['contact2_cylinder_radius'],
            height=pde_params['contact2_cylinder_height'],
            center=pde_params['contact2_cylinder_center'],
            with_half_sphere=False, with_flat_bottom=False)

        in_neumann = is_inside_cylinder_fn(x, y, z,
            radius=pde_params['neumann_cylinder_radius'],
            height=pde_params['neumann_cylinder_height'],
            center=pde_params['neumann_cylinder_center'],
            with_half_sphere=pde_params.get('with_half_sphere', False),
            with_flat_bottom=False)

        if not (in_contact1 or in_contact2 or in_neumann):
            valid_points.append(scaled_points[i])
            valid_values.append(potential_values[i])

    scaled_points = np.array(valid_points)
    potential_values = np.array(valid_values)

    print(f"Filtered from {len(points)} to {len(scaled_points)} FEM data points")

    if n_sample is not None and n_sample < len(scaled_points):
        indices = np.random.choice(len(scaled_points), n_sample, replace=False)
        scaled_points = scaled_points[indices]
        potential_values = potential_values[indices]

    x_d = tf.constant(scaled_points, dtype=tf.float32)
    u_d = tf.constant(potential_values.reshape(-1, 1), dtype=tf.float32)

    print(f"Loaded {len(scaled_points)} FEM data points")
    print(f"Coordinate range: {np.min(scaled_points, axis=0)} to {np.max(scaled_points, axis=0)}")
    print(f"Potential range:  {np.min(potential_values):.4f} to {np.max(potential_values):.4f}")

    return scaled_points, potential_values, x_d, u_d


def interpolate_sigma(points, x_r_all, sigma_r_all):
    """
    Nearest-neighbor interpolation of sigma values at arbitrary query points.

    Args:
        points:      (N, 3) numpy array of query coordinates
        x_r_all:     (M, 3) numpy array of collocation point coordinates
        sigma_r_all: (M,) or (M, 1) numpy array of conductivity values

    Returns:
        (N, 1) numpy array of interpolated sigma values (scaled by 1e-3)
    """
    tree = KDTree(x_r_all)
    _, indices = tree.query(points, k=1)
    interpolated_sigma = sigma_r_all[indices]
    return interpolated_sigma.reshape(-1, 1) * 1e-3
