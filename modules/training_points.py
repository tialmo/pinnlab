import numpy as np
import tensorflow as tf
from modules.geometry import is_inside_cylinder, build_interface_pairs


def generate_training_points(pde_params, domain_bounds,
                              n_r=10000, n_b_cube=600,
                              n_b_contact1=400, n_b_contact2=400,
                              n_b_neumann=400):
    """
    Load collocation points from the pre-built NPZ file, filter out cylinder
    interiors, generate all boundary point sets, and return everything as a dict.

    Note: n_r is currently unused — all filtered points from the NPZ are kept.

    Args:
        pde_params:    Dict of geometry/BC parameters
        domain_bounds: Dict with 'x', 'y', 'z' keys and [min, max] values
        n_r:           (ignored) intended number of collocation points
        n_b_cube:      Number of Neumann boundary points on the cube faces
        n_b_contact1:  Number of Dirichlet boundary points on contact 1
        n_b_contact2:  Number of Dirichlet boundary points on contact 2
        n_b_neumann:   Number of Neumann boundary points on the shaft

    Returns:
        dict with keys:
            x_r, x_r_all, sigma_r, sigma_r_all,
            iface_i, iface_j, iface_n,
            x_n, g_n,
            x_b_c1, u_b_c1,
            x_b_c2, u_b_c2
    """
    x_min, x_max = domain_bounds['x']
    y_min, y_max = domain_bounds['y']
    z_min, z_max = domain_bounds['z']

    # -------------------------------------------------------------------------
    # Load and filter collocation points
    # -------------------------------------------------------------------------
    loaded_data = np.load("remapped_scaled_data.npz")
    X_f_sigma = loaded_data['flat_coordinates']    # [x, y, z, sigma]
    X_f_coordinates = X_f_sigma[:, :3]
    sigma_values = X_f_sigma[:, 3:]

    valid_points, valid_sigmas = [], []
    for i in range(len(X_f_coordinates)):
        x, y, z = X_f_coordinates[i]

        in_contact1 = is_inside_cylinder(x, y, z,
            radius=pde_params['contact1_cylinder_radius'],
            height=pde_params['contact1_cylinder_height'],
            center=pde_params['contact1_cylinder_center'],
            with_half_sphere=False, with_flat_bottom=False)

        in_contact2 = is_inside_cylinder(x, y, z,
            radius=pde_params['contact2_cylinder_radius'],
            height=pde_params['contact2_cylinder_height'],
            center=pde_params['contact2_cylinder_center'],
            with_half_sphere=False, with_flat_bottom=False)

        in_neumann = is_inside_cylinder(x, y, z,
            radius=pde_params['neumann_cylinder_radius'],
            height=pde_params['neumann_cylinder_height'],
            center=pde_params['neumann_cylinder_center'],
            with_half_sphere=pde_params.get('with_half_sphere', False),
            with_flat_bottom=False)

        if not (in_contact1 or in_contact2 or in_neumann):
            valid_points.append(X_f_coordinates[i])
            valid_sigmas.append(sigma_values[i])

    x_r_all = np.array(valid_points)
    sigma_r_all = np.array(valid_sigmas)
    x_r = tf.constant(x_r_all, dtype=tf.float32)
    sigma_r = tf.constant(sigma_r_all, dtype=tf.float32) * 1e-3

    print(f"Loaded {len(x_r)} collocation points (filtered from {len(X_f_coordinates)})")

    iface_i, iface_j, iface_n = build_interface_pairs(x_r_all, sigma_r_all)

    # -------------------------------------------------------------------------
    # Cylinder geometry shorthand
    # -------------------------------------------------------------------------
    contact1_cx, contact1_cy, contact1_cz = pde_params['contact1_cylinder_center']
    contact1_r = pde_params['contact1_cylinder_radius']
    contact1_hh = pde_params['contact1_cylinder_height'] / 2

    contact2_cx, contact2_cy, contact2_cz = pde_params['contact2_cylinder_center']
    contact2_r = pde_params['contact2_cylinder_radius']
    contact2_hh = pde_params['contact2_cylinder_height'] / 2

    neumann_cx, neumann_cy, neumann_cz = pde_params['neumann_cylinder_center']
    neumann_r = pde_params['neumann_cylinder_radius']
    neumann_hh = pde_params['neumann_cylinder_height'] / 2
    with_half_sphere = pde_params.get('with_half_sphere', False)

    # -------------------------------------------------------------------------
    # Cube boundary points (Neumann)
    # -------------------------------------------------------------------------
    n_per_face = n_b_cube // 6
    cube_pts = []

    # Bottom face: z = z_min
    x_f = tf.random.uniform((n_per_face, 1), x_min, x_max)
    y_f = tf.random.uniform((n_per_face, 1), y_min, y_max)
    cube_pts.append(tf.concat([x_f, y_f, tf.ones((n_per_face, 1)) * z_min], axis=1))

    # Top face: z = z_max
    x_f = tf.random.uniform((n_per_face, 1), x_min, x_max)
    y_f = tf.random.uniform((n_per_face, 1), y_min, y_max)
    cube_pts.append(tf.concat([x_f, y_f, tf.ones((n_per_face, 1)) * z_max], axis=1))

    # Left face: x = x_min
    y_f = tf.random.uniform((n_per_face, 1), y_min, y_max)
    z_f = tf.random.uniform((n_per_face, 1), z_min, z_max)
    cube_pts.append(tf.concat([tf.ones((n_per_face, 1)) * x_min, y_f, z_f], axis=1))

    # Right face: x = x_max
    y_f = tf.random.uniform((n_per_face, 1), y_min, y_max)
    z_f = tf.random.uniform((n_per_face, 1), z_min, z_max)
    cube_pts.append(tf.concat([tf.ones((n_per_face, 1)) * x_max, y_f, z_f], axis=1))

    # Front face: y = y_min
    x_f = tf.random.uniform((n_per_face, 1), x_min, x_max)
    z_f = tf.random.uniform((n_per_face, 1), z_min, z_max)
    cube_pts.append(tf.concat([x_f, tf.ones((n_per_face, 1)) * y_min, z_f], axis=1))

    # Back face: y = y_max
    x_f = tf.random.uniform((n_per_face, 1), x_min, x_max)
    z_f = tf.random.uniform((n_per_face, 1), z_min, z_max)
    cube_pts.append(tf.concat([x_f, tf.ones((n_per_face, 1)) * y_max, z_f], axis=1))

    x_b_cube = tf.concat(cube_pts, axis=0)

    # -------------------------------------------------------------------------
    # Contact cylinder 1 boundary points (Dirichlet)
    # -------------------------------------------------------------------------
    if n_b_contact1 > 0:
        theta = tf.random.uniform((n_b_contact1, 1), 0, 2 * np.pi)
        z_c = tf.random.uniform((n_b_contact1, 1),
                                 contact1_cz - contact1_hh,
                                 contact1_cz + contact1_hh)
        x_b_contact1 = tf.concat([
            contact1_cx + contact1_r * tf.cos(theta),
            contact1_cy + contact1_r * tf.sin(theta),
            z_c
        ], axis=1)
    else:
        x_b_contact1 = tf.zeros((0, 3), dtype=tf.float32)

    # -------------------------------------------------------------------------
    # Contact cylinder 2 boundary points (Dirichlet)
    # -------------------------------------------------------------------------
    if n_b_contact2 > 0:
        theta = tf.random.uniform((n_b_contact2, 1), 0, 2 * np.pi)
        z_c = tf.random.uniform((n_b_contact2, 1),
                                 contact2_cz - contact2_hh,
                                 contact2_cz + contact2_hh)
        x_b_contact2 = tf.concat([
            contact2_cx + contact2_r * tf.cos(theta),
            contact2_cy + contact2_r * tf.sin(theta),
            z_c
        ], axis=1)
    else:
        x_b_contact2 = tf.zeros((0, 3), dtype=tf.float32)

    # -------------------------------------------------------------------------
    # Shaft cylinder + optional half-sphere boundary points (Neumann)
    # -------------------------------------------------------------------------
    neumann_pts = []
    if n_b_neumann > 0:
        n_cylinder = int(0.7 * n_b_neumann)
        n_sphere = n_b_neumann - n_cylinder

        if n_cylinder > 0:
            theta = tf.random.uniform((n_cylinder, 1), 0, 2 * np.pi)
            z_n = tf.random.uniform((n_cylinder, 1),
                                     neumann_cz - neumann_hh,
                                     neumann_cz + neumann_hh)
            neumann_pts.append(tf.concat([
                neumann_cx + neumann_r * tf.cos(theta),
                neumann_cy + neumann_r * tf.sin(theta),
                z_n
            ], axis=1))

        if with_half_sphere and n_sphere > 0:
            theta_s = tf.random.uniform((n_sphere, 1), 0, 2 * np.pi)
            phi_s = tf.random.uniform((n_sphere, 1), np.pi / 2, np.pi)
            neumann_pts.append(tf.concat([
                neumann_cx + neumann_r * tf.sin(phi_s) * tf.cos(theta_s),
                neumann_cy + neumann_r * tf.sin(phi_s) * tf.sin(theta_s),
                (neumann_cz - neumann_hh) + neumann_r * tf.cos(phi_s)
            ], axis=1))

    x_b_neumann_all = tf.concat([x_b_cube] + neumann_pts, axis=0)

    # -------------------------------------------------------------------------
    # Boundary condition values
    # -------------------------------------------------------------------------
    val_c1 = tf.constant(pde_params.get('contact1_value', 1.0), dtype=tf.float32)
    val_c2 = tf.constant(pde_params.get('contact2_value', 0.0), dtype=tf.float32)
    neumann_value = pde_params.get('neumann_value', 0.0)

    n_c1 = x_b_contact1.shape[0]
    n_c2 = x_b_contact2.shape[0]
    n_neumann_total = x_b_neumann_all.shape[0]

    u_b_c1 = tf.ones((n_c1, 1)) * val_c1
    u_b_c2 = tf.ones((n_c2, 1)) * val_c2
    g_n = tf.ones((n_neumann_total, 1)) * neumann_value

    print(f"Generated {len(x_r)} collocation points")
    print(f"Generated {n_neumann_total} Neumann boundary points (cube + shaft)")
    print(f"Generated {n_c1} contact 1 Dirichlet boundary points")
    print(f"Generated {n_c2} contact 2 Dirichlet boundary points")

    return dict(
        x_r=x_r,
        x_r_all=x_r_all,
        sigma_r=sigma_r,
        sigma_r_all=sigma_r_all,
        iface_i=iface_i,
        iface_j=iface_j,
        iface_n=iface_n,
        x_n=x_b_neumann_all,
        g_n=g_n,
        x_b_c1=x_b_contact1,
        u_b_c1=u_b_c1,
        x_b_c2=x_b_contact2,
        u_b_c2=u_b_c2,
    )
