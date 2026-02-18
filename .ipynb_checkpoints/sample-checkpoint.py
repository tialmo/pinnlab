#sample.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from netgen.meshing import ImportMesh
import meshio
from scipy.interpolate import griddata
from matplotlib.ticker import MultipleLocator

def _is_inside_shaft_with_half_sphere(point, radius, height, center):
    """
    Shaft geometry where `height` is TOTAL length including the bottom half-sphere.
    Bottom tip is at z_bot = cz - height/2.
    The straight cylinder part begins at z_bot + radius.
    The half-sphere center is at z_bot + radius (so the tip is exactly at z_bot).
    """
    x, y, z = point
    cx, cy, cz = center
    half_h = height / 2.0

    z_bot = cz - half_h
    z_cyl_min = z_bot + radius
    z_cyl_max = cz + half_h  # top stays flat

    # --- cylinder straight part (no bottom cap) ---
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    in_rad = r2 <= radius ** 2
    in_cyl = in_rad and (z_cyl_min <= z <= z_cyl_max)

    # --- bottom half-sphere ---
    sphere_cz = z_bot + radius
    d2 = r2 + (z - sphere_cz) ** 2
    in_sphere = (d2 <= radius ** 2) and (z <= sphere_cz)  # lower half including equator

    return in_cyl or in_sphere



def _sample_cube_faces(bounds, n_total):
    """Uniform sampling on 6 faces; returns points (N,3) and outward normals (N,3)."""
    x0, x1 = bounds["x"]
    y0, y1 = bounds["y"]
    z0, z1 = bounds["z"]

    n_per = n_total // 6
    pts = []
    nrm = []

    # z=z0
    x = np.random.uniform(x0, x1, (n_per, 1))
    y = np.random.uniform(y0, y1, (n_per, 1))
    z = np.full((n_per, 1), z0)
    pts.append(np.hstack([x, y, z]))
    nrm.append(np.tile([0, 0, -1], (n_per, 1)))

    # z=z1
    x = np.random.uniform(x0, x1, (n_per, 1))
    y = np.random.uniform(y0, y1, (n_per, 1))
    z = np.full((n_per, 1), z1)
    pts.append(np.hstack([x, y, z]))
    nrm.append(np.tile([0, 0, 1], (n_per, 1)))

    # x=x0
    x = np.full((n_per, 1), x0)
    y = np.random.uniform(y0, y1, (n_per, 1))
    z = np.random.uniform(z0, z1, (n_per, 1))
    pts.append(np.hstack([x, y, z]))
    nrm.append(np.tile([-1, 0, 0], (n_per, 1)))

    # x=x1
    x = np.full((n_per, 1), x1)
    y = np.random.uniform(y0, y1, (n_per, 1))
    z = np.random.uniform(z0, z1, (n_per, 1))
    pts.append(np.hstack([x, y, z]))
    nrm.append(np.tile([1, 0, 0], (n_per, 1)))

    # y=y0
    x = np.random.uniform(x0, x1, (n_per, 1))
    y = np.full((n_per, 1), y0)
    z = np.random.uniform(z0, z1, (n_per, 1))
    pts.append(np.hstack([x, y, z]))
    nrm.append(np.tile([0, -1, 0], (n_per, 1)))

    # y=y1
    x = np.random.uniform(x0, x1, (n_per, 1))
    y = np.full((n_per, 1), y1)
    z = np.random.uniform(z0, z1, (n_per, 1))
    pts.append(np.hstack([x, y, z]))
    nrm.append(np.tile([0, 1, 0], (n_per, 1)))

    return np.vstack(pts).astype(np.float32), np.vstack(nrm).astype(np.float32)


def _sample_cylinder_lateral(center, radius, height, n):
    """Sample points on lateral surface of z-axis cylinder + outward normals."""
    cx, cy, cz = center
    half_h = height / 2.0

    theta = np.random.uniform(0, 2*np.pi, (n,))
    z = np.random.uniform(cz - half_h, cz + half_h, (n,))

    x = cx + radius * np.cos(theta)
    y = cy + radius * np.sin(theta)

    pts = np.stack([x, y, z], axis=1).astype(np.float32)

    nx = x - cx
    ny = y - cy
    nz = np.zeros_like(nx)
    nrm = np.stack([nx, ny, nz], axis=1).astype(np.float32)
    nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)
    return pts, nrm


def _sample_shaft_half_sphere_bottom(shaft_center, radius, shaft_height, n):
    """
    Sample the BOTTOM half-sphere of the shaft where tip is at z_bot = cz - H/2.
    Sphere center is at z_bot + radius so the tip is exactly at z_bot.
    """
    cx, cy, cz = shaft_center
    half_h = shaft_height / 2.0

    z_bot = cz - half_h
    sphere_cz = z_bot + radius

    theta = np.random.uniform(0, 2*np.pi, (n,))
    phi = np.random.uniform(np.pi/2, np.pi, (n,))  # lower hemisphere

    x = cx + radius * np.sin(phi) * np.cos(theta)
    y = cy + radius * np.sin(phi) * np.sin(theta)
    z = sphere_cz + radius * np.cos(phi)  # ranges [sphere_cz - r (=z_bot), sphere_cz]

    pts = np.stack([x, y, z], axis=1).astype(np.float32)

    # outward normals
    nx = x - cx
    ny = y - cy
    nz = z - sphere_cz
    nrm = np.stack([nx, ny, nz], axis=1).astype(np.float32)
    nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)
    return pts, nrm

def _sample_shaft_cylinder_lateral(shaft_center, radius, shaft_height, n):
    """
    Sample lateral surface of the straight shaft part.
    Straight part spans z in [z_bot + radius, z_top] so it meets the half-sphere at the equator.
    """
    cx, cy, cz = shaft_center
    half_h = shaft_height / 2.0
    z_bot = cz - half_h
    z_top = cz + half_h

    z_min = z_bot + radius
    z_max = z_top

    theta = np.random.uniform(0, 2*np.pi, (n,))
    z = np.random.uniform(z_min, z_max, (n,))

    x = cx + radius * np.cos(theta)
    y = cy + radius * np.sin(theta)

    pts = np.stack([x, y, z], axis=1).astype(np.float32)

    nx = x - cx
    ny = y - cy
    nz = np.zeros_like(nx)
    nrm = np.stack([nx, ny, nz], axis=1).astype(np.float32)
    nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)
    return pts, nrm


def sample_bipolar_geometry_points(
    domain_bounds: dict,
    pde_params: dict,
    n_r: int = 20000,
    n_cube_neumann: int = 2400,
    n_shaft_neumann: int = 600,
    n_contact1_dirichlet: int = 600,
    n_contact2_dirichlet: int = 600,
    sigma_fn=None,
    filter_collocation=True,
    collocation_oversample_factor=2.0,
    seed=None,
):
    """
    Samples:
      - x_r, sigma_r          : collocation interior points & sigma
      - x_n, n_n, g_n         : Neumann points (cube faces + shaft lateral + shaft half-sphere)
      - x_b_c1, u_b_c1        : Dirichlet contact 1 lateral surface
      - x_b_c2, u_b_c2        : Dirichlet contact 2 lateral surface
    """

    if seed is not None:
        np.random.seed(seed)

    # --- unpack params ---
    c1_r = pde_params["contact1_cylinder_radius"]
    c1_h = pde_params["contact1_cylinder_height"]
    c1_c = pde_params["contact1_cylinder_center"]
    V1 = float(pde_params.get("contact1_value", 1.0))

    c2_r = pde_params["contact2_cylinder_radius"]
    c2_h = pde_params["contact2_cylinder_height"]
    c2_c = pde_params["contact2_cylinder_center"]
    V2 = float(pde_params.get("contact2_value", 0.0))

    sh_r = pde_params["neumann_cylinder_radius"]
    sh_h = pde_params["neumann_cylinder_height"]
    sh_c = pde_params["neumann_cylinder_center"]
    neumann_value = float(pde_params.get("neumann_value", 0.0))

    # --- collocation ---
    x0, x1 = domain_bounds["x"]
    y0, y1 = domain_bounds["y"]
    z0, z1 = domain_bounds["z"]

    n_try = int(np.ceil(n_r * collocation_oversample_factor))
    xr_try = np.random.uniform([x0, y0, z0], [x1, y1, z1], size=(n_try, 3)).astype(np.float32)

    if filter_collocation:
        keep = []
        for i in range(n_try):
            p = xr_try[i]
            in_c1 = _is_inside_shaft_with_half_sphere(p, c1_r, c1_h, c1_c)  # contacts also half-sphere? (kept consistent)
            in_c2 = _is_inside_shaft_with_half_sphere(p, c2_r, c2_h, c2_c)
            in_sh = _is_inside_shaft_with_half_sphere(p, sh_r, sh_h, sh_c)  # shaft always has half-sphere
            if not (in_c1 or in_c2 or in_sh):
                keep.append(p)
            if len(keep) >= n_r:
                break
        x_r = np.array(keep, dtype=np.float32)
        if x_r.shape[0] < n_r:
            raise RuntimeError(
                f"Not enough collocation points after filtering: got {x_r.shape[0]}/{n_r}. "
                f"Increase collocation_oversample_factor."
            )
    else:
        x_r = xr_try[:n_r]

    # sigma
    if sigma_fn is None:
        sigma_r = np.ones((x_r.shape[0], 1), dtype=np.float32)
    else:
        sigma_r = np.asarray(sigma_fn(x_r), dtype=np.float32).reshape(-1, 1)

    # --- Neumann boundary points ---
    x_cube, n_cube = _sample_cube_faces(domain_bounds, n_cube_neumann)

    n_cyl = int(0.7 * n_shaft_neumann)
    n_sph = n_shaft_neumann - n_cyl

    x_sh_cyl, n_sh_cyl = _sample_shaft_cylinder_lateral(sh_c, sh_r, sh_h, n_cyl) if n_cyl > 0 else (np.zeros((0,3),np.float32), np.zeros((0,3),np.float32))
    x_sh_sph, n_sh_sph = _sample_shaft_half_sphere_bottom(sh_c, sh_r, sh_h, n_sph) if n_sph > 0 else (np.zeros((0,3),np.float32), np.zeros((0,3),np.float32))

    x_n = np.vstack([x_cube, x_sh_cyl, x_sh_sph]).astype(np.float32)
    n_n = np.vstack([n_cube, n_sh_cyl, n_sh_sph]).astype(np.float32)
    g_n = np.full((x_n.shape[0], 1), neumann_value, dtype=np.float32)

    # --- Dirichlet contacts (lateral surfaces only) ---
    x_b_c1, _ = _sample_cylinder_lateral(c1_c, c1_r, c1_h, n_contact1_dirichlet) if n_contact1_dirichlet > 0 else (np.zeros((0,3),np.float32), None)
    x_b_c2, _ = _sample_cylinder_lateral(c2_c, c2_r, c2_h, n_contact2_dirichlet) if n_contact2_dirichlet > 0 else (np.zeros((0,3),np.float32), None)

    u_b_c1 = np.full((x_b_c1.shape[0], 1), V1, dtype=np.float32)
    u_b_c2 = np.full((x_b_c2.shape[0], 1), V2, dtype=np.float32)

    return {
        "x_r": tf.constant(x_r, dtype=tf.float32),
        "sigma_r": tf.constant(sigma_r, dtype=tf.float32),
        "x_n": tf.constant(x_n, dtype=tf.float32),
        "n_n": tf.constant(n_n, dtype=tf.float32),
        "g_n": tf.constant(g_n, dtype=tf.float32),
        "x_b_c1": tf.constant(x_b_c1, dtype=tf.float32),
        "u_b_c1": tf.constant(u_b_c1, dtype=tf.float32),
        "x_b_c2": tf.constant(x_b_c2, dtype=tf.float32),
        "u_b_c2": tf.constant(u_b_c2, dtype=tf.float32),
    }

def plot_sampled_points(batch, max_points=8000, elev=18, azim=35, title="Sampled PINN Points"):
    """
    Visual sanity-check of all sampled points:
      - 3D scatter (downsampled)
      - XY / XZ / YZ projections (downsampled)
    """
    def to_np(x):
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

    x_r = to_np(batch["x_r"])
    x_n = to_np(batch["x_n"])
    x_c1 = to_np(batch["x_b_c1"])
    x_c2 = to_np(batch["x_b_c2"])

    # Downsample each set proportionally for plotting
    def ds(X, n):
        if X.shape[0] <= n:
            return X
        idx = np.random.choice(X.shape[0], n, replace=False)
        return X[idx]

    # allocate plot budget roughly by sizes
    total = x_r.shape[0] + x_n.shape[0] + x_c1.shape[0] + x_c2.shape[0]
    def budget(k, N):
        return max(50, int(max_points * (N / max(total, 1))))

    xr_p = ds(x_r, budget("r", x_r.shape[0]))
    xn_p = ds(x_n, budget("n", x_n.shape[0]))
    xc1_p = ds(x_c1, budget("c1", x_c1.shape[0]))
    xc2_p = ds(x_c2, budget("c2", x_c2.shape[0]))

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(title, fontsize=14)

    # --- 3D view ---
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.scatter(xr_p[:, 0], xr_p[:, 1], xr_p[:, 2], s=2, alpha=0.35, label=f"Collocation x_r ({x_r.shape[0]})")
    ax.scatter(xn_p[:, 0], xn_p[:, 1], xn_p[:, 2], s=4, alpha=0.6, label=f"Neumann x_n ({x_n.shape[0]})")
    ax.scatter(xc1_p[:, 0], xc1_p[:, 1], xc1_p[:, 2], s=8, alpha=0.9, label=f"Dirichlet c1 ({x_c1.shape[0]})")
    ax.scatter(xc2_p[:, 0], xc2_p[:, 1], xc2_p[:, 2], s=8, alpha=0.9, label=f"Dirichlet c2 ({x_c2.shape[0]})")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=elev, azim=azim)
    ax.legend(loc="upper left")

    # --- XY projection ---
    ax = fig.add_subplot(2, 2, 2)
    ax.scatter(xr_p[:, 0], xr_p[:, 1], s=2, alpha=0.35)
    ax.scatter(xn_p[:, 0], xn_p[:, 1], s=4, alpha=0.6)
    ax.scatter(xc1_p[:, 0], xc1_p[:, 1], s=8, alpha=0.9)
    ax.scatter(xc2_p[:, 0], xc2_p[:, 1], s=8, alpha=0.9)
    ax.set_title("XY projection")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.grid(True, ls="--", alpha=0.4)

    # --- XZ projection ---
    ax = fig.add_subplot(2, 2, 3)
    ax.scatter(xr_p[:, 0], xr_p[:, 2], s=2, alpha=0.35)
    ax.scatter(xn_p[:, 0], xn_p[:, 2], s=4, alpha=0.6)
    ax.scatter(xc1_p[:, 0], xc1_p[:, 2], s=8, alpha=0.9)
    ax.scatter(xc2_p[:, 0], xc2_p[:, 2], s=8, alpha=0.9)
    ax.set_title("XZ projection")
    ax.set_xlabel("x"); ax.set_ylabel("z")
    ax.grid(True, ls="--", alpha=0.4)

    # --- YZ projection ---
    ax = fig.add_subplot(2, 2, 4)
    ax.scatter(xr_p[:, 1], xr_p[:, 2], s=2, alpha=0.35)
    ax.scatter(xn_p[:, 1], xn_p[:, 2], s=4, alpha=0.6)
    ax.scatter(xc1_p[:, 1], xc1_p[:, 2], s=8, alpha=0.9)
    ax.scatter(xc2_p[:, 1], xc2_p[:, 2], s=8, alpha=0.9)
    ax.set_title("YZ projection")
    ax.set_xlabel("y"); ax.set_ylabel("z")
    ax.grid(True, ls="--", alpha=0.4)

    plt.tight_layout()
    #plt.show()
    return fig
