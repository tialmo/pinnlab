import tensorflow as tf
from modules.geometry import compute_normal_vectors


def compute_residual(x, main_net):
    """
    Compute the Laplace equation residual: u_xx + u_yy + u_zz.
    Uses component-wise differentiation to avoid issues with joint tapes.

    Args:
        x:        (N, 3) tensor of collocation points [x, y, z]
        main_net: Keras model mapping (N, 3) -> (N, 1)

    Returns:
        (N, 1) tensor of Laplacian residuals
    """
    x_coord = x[:, 0:1]
    y_coord = x[:, 1:2]
    z_coord = x[:, 2:3]

    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([x_coord, y_coord, z_coord])
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([x_coord, y_coord, z_coord])
            x_in = tf.concat([x_coord, y_coord, z_coord], axis=1)
            u = main_net(x_in)
        u_x = tape1.gradient(u, x_coord)
        u_y = tape1.gradient(u, y_coord)
        u_z = tape1.gradient(u, z_coord)

    u_xx = tape2.gradient(u_x, x_coord)
    u_yy = tape2.gradient(u_y, y_coord)
    u_zz = tape2.gradient(u_z, z_coord)

    return u_xx + u_yy + u_zz


def compute_neumann_squared_errors(x_n, main_net, domain_bounds, pde_params):
    """
    Compute squared errors for the Neumann (zero normal-flux) boundary condition
    on both the cubic domain faces and the shaft cylinder surface.

    Args:
        x_n:           (N, 3) tensor of Neumann boundary points
        main_net:      Keras model mapping (N, 3) -> (N, 1)
        domain_bounds: dict with 'x', 'y', 'z' and [min, max] values
        pde_params:    dict containing 'neumann_value' and cylinder center

    Returns:
        (N,) tensor of squared Neumann residuals
    """
    with tf.GradientTape() as tape:
        tape.watch(x_n)
        u_n = main_net(x_n)
    grad_u_n = tape.gradient(u_n, x_n)

    normal_vectors = compute_normal_vectors(x_n, domain_bounds, pde_params)
    normal_derivative = tf.reduce_sum(grad_u_n * normal_vectors, axis=1)
    neumann_value = pde_params.get('neumann_value', 0.0)

    return tf.square(normal_derivative - neumann_value)


def compute_interface_flux_residual(xi, xj, si, sj, nij, main_net):
    """
    Enforce the interface flux-continuity condition:
        sigma_i * grad(u)(xi) · n  =  sigma_j * grad(u)(xj) · n

    Args:
        xi:       (B, 3) coordinates on side i of the interface
        xj:       (B, 3) coordinates on side j of the interface
        si:       (B, 1) conductivity values on side i
        sj:       (B, 1) conductivity values on side j
        nij:      (B, 3) unit normal vectors from i to j
        main_net: Keras model mapping (N, 3) -> (N, 1)

    Returns:
        (B, 1) tensor of flux-continuity residuals
    """
    pts = tf.concat([xi, xj], axis=0)

    with tf.GradientTape() as tape:
        tape.watch(pts)
        u = main_net(pts)
    gu = tape.gradient(u, pts)

    B = tf.shape(xi)[0]
    gu_i, gu_j = gu[:B, :], gu[B:, :]

    flux_i = si * tf.reduce_sum(gu_i * nij, axis=1, keepdims=True)
    flux_j = sj * tf.reduce_sum(gu_j * nij, axis=1, keepdims=True)

    return flux_i - flux_j
