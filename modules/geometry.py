import numpy as np
import tensorflow as tf


def is_inside_cylinder(x, y, z, radius, height, center,
                        with_half_sphere=False, with_flat_bottom=False):
    """
    Check if a point (x, y, z) is inside a cylinder, optionally with
    a half-sphere at the bottom.

    Args:
        x, y, z:          Coordinates of the point
        radius:           Cylinder radius
        height:           Cylinder height (total)
        center:           [cx, cy, cz] center of the cylinder
        with_half_sphere: If True, include a hemispherical cap at the bottom
        with_flat_bottom: (reserved, unused)

    Returns:
        bool
    """
    cx, cy, cz = center
    dist_squared = (x - cx) ** 2 + (y - cy) ** 2
    in_radius = dist_squared <= radius ** 2
    half_height = height / 2
    in_cylinder = in_radius and (cz - half_height <= z <= cz + half_height)

    in_half_sphere = False
    if with_half_sphere:
        sphere_center_z = cz - half_height
        sphere_dist_squared = dist_squared + (z - sphere_center_z) ** 2
        in_half_sphere = (sphere_dist_squared <= radius ** 2) and (z < cz - half_height)

    return in_cylinder or in_half_sphere


def compute_normal_vectors(x_n, domain_bounds, pde_params):
    """
    Compute outward unit normal vectors for Neumann boundary points.
    Handles both the cubic domain faces and the shaft cylinder surface.

    Args:
        x_n:           (N, 3) tensor of boundary points
        domain_bounds: dict with keys 'x', 'y', 'z' and [min, max] values
        pde_params:    dict containing 'neumann_cylinder_center'

    Returns:
        (N, 3) tensor of unit normal vectors
    """
    x, y, z = x_n[:, 0], x_n[:, 1], x_n[:, 2]

    x_min, x_max = domain_bounds['x']
    y_min, y_max = domain_bounds['y']
    z_min, z_max = domain_bounds['z']

    cx, cy, cz = pde_params.get('neumann_cylinder_center', [0, 0, 0])

    normals = tf.zeros_like(x_n)
    tol = 1e-3

    # Cube faces — outward normals
    mask_bottom = tf.abs(z - z_min) < tol
    normals = tf.where(tf.expand_dims(mask_bottom, 1),
                       tf.constant([0.0, 0.0, -1.0]), normals)

    mask_top = tf.abs(z - z_max) < tol
    normals = tf.where(tf.expand_dims(mask_top, 1),
                       tf.constant([0.0, 0.0, 1.0]), normals)

    mask_left = tf.abs(x - x_min) < tol
    normals = tf.where(tf.expand_dims(mask_left, 1),
                       tf.constant([-1.0, 0.0, 0.0]), normals)

    mask_right = tf.abs(x - x_max) < tol
    normals = tf.where(tf.expand_dims(mask_right, 1),
                       tf.constant([1.0, 0.0, 0.0]), normals)

    mask_front = tf.abs(y - y_min) < tol
    normals = tf.where(tf.expand_dims(mask_front, 1),
                       tf.constant([0.0, -1.0, 0.0]), normals)

    mask_back = tf.abs(y - y_max) < tol
    normals = tf.where(tf.expand_dims(mask_back, 1),
                       tf.constant([0.0, 1.0, 0.0]), normals)

    # Shaft cylinder surface — radial outward normals
    on_cube = mask_bottom | mask_top | mask_left | mask_right | mask_front | mask_back
    on_cylinder = tf.logical_not(on_cube)

    normals_x = tf.where(on_cylinder, x - cx, normals[:, 0])
    normals_y = tf.where(on_cylinder, y - cy, normals[:, 1])
    normals_z = tf.where(on_cylinder, tf.zeros_like(z), normals[:, 2])

    cylinder_normals = tf.stack([normals_x, normals_y, normals_z], axis=1)
    norm = tf.norm(cylinder_normals, axis=1, keepdims=True)
    normalized = cylinder_normals / (norm + 1e-8)

    return normalized


def build_interface_pairs(x_r_all, sigma_r_all, tol_sigma=1e-12):
    """
    Build pairs of neighboring structured-grid points (i, j) where sigma differs.
    Only +x, +y, +z neighbors are checked to avoid duplicate pairs.

    Args:
        x_r_all:    (N, 3) numpy array of collocation point coordinates
        sigma_r_all:(N,) or (N,1) numpy array of conductivity values
        tol_sigma:  Minimum sigma difference to be considered an interface

    Returns:
        Tuple of three tf.constant tensors: (iface_i, iface_j, iface_n)
        where iface_n contains the unit normals pointing from i to j.
    """
    X = x_r_all
    S = sigma_r_all.reshape(-1)

    xs = np.unique(X[:, 0]); ys = np.unique(X[:, 1]); zs = np.unique(X[:, 2])
    xs.sort(); ys.sort(); zs.sort()
    dx = np.min(np.diff(xs))
    dy = np.min(np.diff(ys))
    dz = np.min(np.diff(zs))

    x0, y0, z0 = xs[0], ys[0], zs[0]
    ix = np.rint((X[:, 0] - x0) / dx).astype(int)
    iy = np.rint((X[:, 1] - y0) / dy).astype(int)
    iz = np.rint((X[:, 2] - z0) / dz).astype(int)

    coord2idx = {(ix[k], iy[k], iz[k]): k for k in range(len(X))}

    iface_i, iface_j, iface_n = [], [], []

    for k in range(len(X)):
        a = (ix[k], iy[k], iz[k])
        for step, nvec in [((1, 0, 0), (1.0, 0.0, 0.0)),
                            ((0, 1, 0), (0.0, 1.0, 0.0)),
                            ((0, 0, 1), (0.0, 0.0, 1.0))]:
            b = (a[0] + step[0], a[1] + step[1], a[2] + step[2])
            j = coord2idx.get(b, None)
            if j is None:
                continue
            if abs(S[k] - S[j]) > tol_sigma:
                iface_i.append(k)
                iface_j.append(j)
                iface_n.append(nvec)

    iface_i = np.array(iface_i, dtype=np.int32)
    iface_j = np.array(iface_j, dtype=np.int32)
    iface_n = np.array(iface_n, dtype=np.float32)

    print(f"Built {len(iface_i)} interface neighbor pairs.")

    return tf.constant(iface_i), tf.constant(iface_j), tf.constant(iface_n)
