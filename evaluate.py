import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import meshio
from scipy.interpolate import griddata
from matplotlib.ticker import MultipleLocator

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ----------------------------------------------------------------
# 1. HELPER FUNCTIONS
# ----------------------------------------------------------------

def remap_conductivity_to_int(conductivity_array):
    """Map conductivity float values to integers 0-3 by ranking unique values."""
    unique_vals = np.sort(np.unique(np.round(conductivity_array, 6)))
    mapping = {v: i for i, v in enumerate(unique_vals)}
    remapped = np.vectorize(lambda x: mapping[min(mapping.keys(), key=lambda k: abs(k - x))])(conductivity_array)
    return remapped, unique_vals

def apply_mask(grid, mask):
    """Return a copy of grid with masked cells set to NaN (background color)."""
    out = np.array(grid, dtype=float)
    out[mask] = np.nan
    return out

def contact_distance_weights(pts, contact_centers, sigma=0.1):
    diffs = pts[:, None, :] - contact_centers[None, :, :]   # (N, K, 3)
    dists = np.linalg.norm(diffs, axis=-1)                   # (N, K)
    min_dist = dists.min(axis=1)                             # (N,) nearest contact
    return np.exp(-(min_dist**2) / (2 * sigma**2))
    
# ----------------------------------------------------------------
# 2. MAIN COMPARISON & PLOTTING ENGINE
# ----------------------------------------------------------------

def compare_with_fem(solver, n_grid=400, x_bounds=(-1, 1), y_bounds=(-1, 1), z_bounds=(-1, 1), epoch=None, save_dir=None):
    # Load Data
    mesh = meshio.read("remapped_binary_potential.vtu")
    loaded_data = np.load("remapped_scaled_data.npz")
    X_f = tf.constant(loaded_data['flat_coordinates'], dtype=tf.float32)

    # Geometry Parameters
    shaft_cx, shaft_cy, shaft_cz = solver.pde_params['neumann_cylinder_center']
    shaft_r = solver.pde_params['neumann_cylinder_radius']
    shaft_hh = solver.pde_params['neumann_cylinder_height'] / 2
    shaft_bottom = shaft_cz - shaft_hh
    shaft_top = shaft_cz + shaft_hh

    # Masking Logic
    def _mask_xy(Xg, Yg):
        return (Xg - shaft_cx)**2 + (Yg - shaft_cy)**2 < shaft_r**2

    def _mask_xz(Hg, Zg, h_center=shaft_cx):
        h_rel = Hg - h_center
        in_body = (np.abs(h_rel) < shaft_r) & (Zg >= shaft_bottom) & (Zg <= shaft_top)
        in_cap = (h_rel**2 + (Zg - shaft_bottom)**2 < shaft_r**2) & (Zg < shaft_bottom)
        return in_body | in_cap

    def _mask_yz(Hg, Zg, h_center=shaft_cy):
        return _mask_xz(Hg, Zg, h_center=h_center)

    # Slice coordinates
    x_slice, y_slice, z_slice = 0.0, 0.0, 0.1
    tolerance = 0.01  # 1e-5

    # --- FEM Data Processing ---
    fem_pts, fem_pot = mesh.points, mesh.point_data['potential_real']

    # Use the user-provided plotting bounds for all slice grids
    x_f_lim = np.linspace(x_bounds[0], x_bounds[1], n_grid)
    y_f_lim = np.linspace(y_bounds[0], y_bounds[1], n_grid)
    z_f_lim = np.linspace(z_bounds[0], z_bounds[1], n_grid)

    X_xy, Y_xy = np.meshgrid(x_f_lim, y_f_lim)
    X_xz, Z_xz = np.meshgrid(x_f_lim, z_f_lim)
    Y_yz, Z_yz = np.meshgrid(y_f_lim, z_f_lim)

    fem_xy = apply_mask(
        griddata(
            fem_pts,
            fem_pot,
            np.column_stack([X_xy.ravel(), Y_xy.ravel(), np.full(X_xy.size, z_slice)]),
            method='linear'
        ).reshape(n_grid, n_grid),
        _mask_xy(X_xy, Y_xy)
    )
    fem_xz = apply_mask(
        griddata(
            fem_pts,
            fem_pot,
            np.column_stack([X_xz.ravel(), np.full(X_xz.size, y_slice), Z_xz.ravel()]),
            method='linear'
        ).reshape(n_grid, n_grid),
        _mask_xz(X_xz, Z_xz)
    )
    fem_yz = apply_mask(
        griddata(
            fem_pts,
            fem_pot,
            np.column_stack([np.full(Y_yz.size, x_slice), Y_yz.ravel(), Z_yz.ravel()]),
            method='linear'
        ).reshape(n_grid, n_grid),
        _mask_yz(Y_yz, Z_yz)
    )

    # --- PINN Data Processing ---
    X_f_np = X_f.numpy()
    cs_z = X_f_np[np.isclose(X_f_np[:, 2], z_slice, atol=tolerance)]
    cs_y = X_f_np[np.isclose(X_f_np[:, 1], y_slice, atol=tolerance)]
    cs_x = X_f_np[np.isclose(X_f_np[:, 0], x_slice, atol=tolerance)]

    pred_z, cond_z = solver.predict(cs_z[:, :3]), cs_z[:, 3]
    pred_y, cond_y = solver.predict(cs_y[:, :3]), cs_y[:, 3]
    pred_x, cond_x = solver.predict(cs_x[:, :3]), cs_x[:, 3]

    # Conductivity Mapping
    _, unique_sigma = remap_conductivity_to_int(np.concatenate([cond_z, cond_y, cond_x]))
    n_tissues = len(unique_sigma)
    unique_sigma[[0, 3]] = unique_sigma[[3, 0]]
    tissue_labels = {i: ['CSF', 'WM', 'GM', 'nan'][i] for i in range(n_tissues)}
    remap_fn = np.vectorize(lambda x: min(range(n_tissues), key=lambda i: abs(unique_sigma[i] - x)))

    def get_plane_bounds(col_a, col_b):
        bounds_map = {0: x_bounds, 1: y_bounds, 2: z_bounds}
        return bounds_map[col_a], bounds_map[col_b]

    # Gridding PINN Results
    def grid_pinn(cs, vals, col_a, col_b, mask_fn):
        a_bounds, b_bounds = get_plane_bounds(col_a, col_b)
        ga = np.linspace(a_bounds[0], a_bounds[1], n_grid)
        gb = np.linspace(b_bounds[0], b_bounds[1], n_grid)
        Ga, Gb = np.meshgrid(ga, gb)
        grid = griddata((cs[:, col_a], cs[:, col_b]), vals, (Ga, Gb), method='cubic')
        return ga, gb, apply_mask(grid, mask_fn(Ga, Gb))

    def grid_raw(cs, vals, col_a, col_b, mask_fn):
        a_bounds, b_bounds = get_plane_bounds(col_a, col_b)

        # Keep only points inside requested plotting bounds
        in_bounds = (
            (cs[:, col_a] >= a_bounds[0]) & (cs[:, col_a] <= a_bounds[1]) &
            (cs[:, col_b] >= b_bounds[0]) & (cs[:, col_b] <= b_bounds[1])
        )
        cs = cs[in_bounds]
        vals = vals[in_bounds]

        # Get unique coordinate values as they actually exist
        ga = np.sort(np.unique(cs[:, col_a]))
        gb = np.sort(np.unique(cs[:, col_b]))

        # Build lookup and fill grid
        grid = np.full((len(gb), len(ga)), np.nan)
        a_idx = {v: i for i, v in enumerate(ga)}
        b_idx = {v: i for i, v in enumerate(gb)}
        for pt, val in zip(cs, vals):
            i = a_idx.get(pt[col_a])
            j = b_idx.get(pt[col_b])
            if i is not None and j is not None:
                grid[j, i] = val

        Ga, Gb = np.meshgrid(ga, gb)
        return ga, gb, apply_mask(grid, mask_fn(Ga, Gb))

    xg_z, yg_z, pot_z = grid_pinn(cs_z, pred_z, 0, 1, _mask_xy)
    xg_y, zg_y, pot_y = grid_pinn(cs_y, pred_y, 0, 2, _mask_xz)
    yg_x, zg_x, pot_x = grid_pinn(cs_x, pred_x, 1, 2, _mask_yz)

    _, _, cnd_z = grid_raw(cs_z, remap_fn(cond_z), 0, 1, _mask_xy)
    _, _, cnd_y = grid_raw(cs_y, remap_fn(cond_y), 0, 2, _mask_xz)
    _, _, cnd_x = grid_raw(cs_x, remap_fn(cond_x), 1, 2, _mask_yz)

    in_bounds = (
        (X_f_np[:, 0] >= x_bounds[0]) & (X_f_np[:, 0] <= x_bounds[1]) &
        (X_f_np[:, 1] >= y_bounds[0]) & (X_f_np[:, 1] <= y_bounds[1]) &
        (X_f_np[:, 2] >= z_bounds[0]) & (X_f_np[:, 2] <= z_bounds[1])
    )
    X_f_np = X_f_np[in_bounds]

    # --- Error Metrics ---
    interp_fem = griddata(fem_pts, fem_pot, X_f_np[:, :3], method='linear').flatten()
    abs_error = np.abs(interp_fem - solver.predict(X_f_np[:, :3]).numpy().flatten())
    mae, rmse, max_ae = np.mean(abs_error), np.sqrt(np.mean(abs_error**2)), np.max(abs_error)
    nonzero_mask = np.abs(interp_fem) > 1e-10
    #rel_error = np.zeros_like(interp_fem)
    #rel_error[nonzero_mask] = abs_error[nonzero_mask] / np.abs(interp_fem[nonzero_mask])
    rel_error = np.zeros(len(interp_fem))
    rel_error[nonzero_mask] = abs_error[nonzero_mask] / np.abs(interp_fem[nonzero_mask])

    '''
    # Run PINN on the exact FEM points
    pinn_pot = solver.predict(fem_pts)

    # Direct comparison — no interpolation
    abs_error = np.abs(fem_pot - pinn_pot)
    mae, rmse, max_ae = np.mean(abs_error), np.sqrt(np.mean(abs_error**2)), np.max(abs_error)
    rel_error = np.zeros_like(fem_pot)
    nonzero_mask = np.abs(fem_pot) > 1e-10
    rel_error[nonzero_mask] = abs_error[nonzero_mask] / np.abs(fem_pot[nonzero_mask])
    X_f_np = fem_pts
    '''
    # --- Weighted Error Metrics ---
    contact_centers = np.array([
        solver.pde_params['contact1_cylinder_center'],
        solver.pde_params['contact2_cylinder_center'],
    ])
    weights_raw = contact_distance_weights(X_f_np[:, :3], contact_centers, sigma=0.1)
    weights = weights_raw / weights_raw.sum()

    wmae = np.sum(weights * abs_error)
    wrel_error = np.zeros_like(interp_fem)
    wrel_error[nonzero_mask] = abs_error[nonzero_mask] / np.abs(interp_fem[nonzero_mask])
    wmre = np.sum(weights * wrel_error) * 100

    
    def grid_err(cs_indices, col_a, col_b, mask_fn, error_array=abs_error):
        pts, err = X_f_np[cs_indices], error_array[cs_indices]
        a_bounds, b_bounds = get_plane_bounds(col_a, col_b)
        ga = np.linspace(a_bounds[0], a_bounds[1], n_grid)
        gb = np.linspace(b_bounds[0], b_bounds[1], n_grid)
        Ga, Gb = np.meshgrid(ga, gb)
        grid = griddata((pts[:, col_a], pts[:, col_b]), err, (Ga, Gb), method='linear')
        return ga, gb, apply_mask(grid, mask_fn(Ga, Gb))

    eg_z_x, eg_z_y, err_z = grid_err(np.isclose(X_f_np[:, 2], z_slice, atol=tolerance), 0, 1, _mask_xy)
    eg_y_x, eg_y_z, err_y = grid_err(np.isclose(X_f_np[:, 1], y_slice, atol=tolerance), 0, 2, _mask_xz)
    eg_x_y, eg_x_z, err_x = grid_err(np.isclose(X_f_np[:, 0], x_slice, atol=tolerance), 1, 2, _mask_yz)

    _, _, rel_err_z = grid_err(np.isclose(X_f_np[:, 2], z_slice, atol=tolerance), 0, 1, _mask_xy, rel_error)
    _, _, rel_err_y = grid_err(np.isclose(X_f_np[:, 1], y_slice, atol=tolerance), 0, 2, _mask_xz, rel_error)
    _, _, rel_err_x = grid_err(np.isclose(X_f_np[:, 0], x_slice, atol=tolerance), 1, 2, _mask_yz, rel_error)

    _, _, w_z = grid_err(np.isclose(X_f_np[:, 2], z_slice, atol=tolerance), 0, 1, _mask_xy, weights_raw)
    _, _, w_y = grid_err(np.isclose(X_f_np[:, 1], y_slice, atol=tolerance), 0, 2, _mask_xz, weights_raw)
    _, _, w_x = grid_err(np.isclose(X_f_np[:, 0], x_slice, atol=tolerance), 1, 2, _mask_yz, weights_raw)

    # ----------------------------------------------------------------
    # 3. PLOT CONSTRUCTION
    # ----------------------------------------------------------------

    plt.rcParams.update({'font.family': 'serif', 'font.serif': 'DejaVu Serif'})
    fig = plt.figure(figsize=(22, 35), constrained_layout=True)
    gs = gridspec.GridSpec(5, 3, figure=fig)

    titlefont, axislabel, ticksize, cbar_tick = 24, 20, 18, 18
    cmap_pot, cmap_err = plt.get_cmap('coolwarm'), plt.get_cmap('plasma', 10)
    disc_cmap = plt.get_cmap('gray', n_tissues)
    for cm in [cmap_pot, cmap_err, disc_cmap]:
        cm.set_bad('#ffffff')

    def setup_ax(ax, grid, extent, title, cmap, is_disc=False, norm=None, cbar_label='',
                 xlabel='x', ylabel='y', vmin=None, vmax=None, cbar_ticks=None,
                 xlim=None, ylim=None):
        x_min, x_max, y_min, y_max = extent
        pad = 0.05
        padded_extent = [x_min - pad, x_max + pad, y_min - pad, y_max + pad]

        im = ax.imshow(
            grid,
            extent=padded_extent,
            origin='lower',
            cmap=cmap,
            norm=norm,
            aspect='equal',
            vmin=vmin,
            vmax=vmax
        )

        ax.set_title(title, fontsize=titlefont, pad=20)
        ax.tick_params(labelsize=ticksize)

        scale_mm = 20  # 1 unit = 20 mm

        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])

        #x_ticks = np.arange(int(np.ceil(x_min)), int(np.ceil(x_max)) + 1)
        #y_ticks = np.arange(int(np.ceil(y_min)), int(np.ceil(y_max)) + 1)
        x_ticks = np.linspace(x_min, x_max, 5)
        y_ticks = np.linspace(y_min, y_max, 5)

        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels([f'{int(t * scale_mm)}' for t in x_ticks])
        ax.set_yticklabels([f'{int(t * scale_mm)}' for t in y_ticks])

        ax.set_xlabel(f'{xlabel} / mm', fontsize=axislabel)
        ax.set_ylabel(f'{ylabel} / mm', fontsize=axislabel)

        if is_disc:
            cbar = fig.colorbar(im, ax=ax, ticks=np.arange(n_tissues), shrink=0.7, pad=0.02)
            cbar.ax.set_yticklabels([tissue_labels[i] for i in range(n_tissues)], fontsize=cbar_tick)
        else:
            cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
            cbar.ax.tick_params(labelsize=cbar_tick)
            if cbar_ticks is not None:
                cbar.set_ticks(cbar_ticks)

        cbar.set_label(cbar_label, fontsize=axislabel, labelpad=10)
        return ax

    # Row 0: Conductivity
    norm_disc = mcolors.BoundaryNorm(np.arange(-0.5, n_tissues, 1), disc_cmap.N)
    setup_ax(plt.subplot(gs[0, 0]), cnd_z, [x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]],
             f'coronal slice (z={z_slice*20:.0f}mm)', disc_cmap, True, norm_disc,
             cbar_label='Brain tissue', xlabel='x', ylabel='y', xlim=x_bounds, ylim=y_bounds)
    setup_ax(plt.subplot(gs[0, 1]), cnd_y, [x_bounds[0], x_bounds[1], z_bounds[0], z_bounds[1]],
             f'axial slice (y={y_slice*20:.0f}mm)', disc_cmap, True, norm_disc,
             cbar_label='Brain tissue', xlabel='x', ylabel='z', xlim=x_bounds, ylim=z_bounds)
    setup_ax(plt.subplot(gs[0, 2]), cnd_x, [y_bounds[0], y_bounds[1], z_bounds[0], z_bounds[1]],
             f'sagittal slice (x={x_slice*20:.0f}mm)', disc_cmap, True, norm_disc,
             cbar_label='Brain tissue', xlabel='y', ylabel='z', xlim=y_bounds, ylim=z_bounds)

    pot_ticks = np.arange(0, 1.1, 0.2)

    # Row 1: PINN
    setup_ax(plt.subplot(gs[1, 0]), pot_z, [x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]],
             f'PINN prediction on xy-plane (z={z_slice*20:.0f}mm)', cmap_pot,
             cbar_label='Electric potential / V', xlabel='x', ylabel='y',
             vmin=0, vmax=1, cbar_ticks=pot_ticks, xlim=x_bounds, ylim=y_bounds)
    setup_ax(plt.subplot(gs[1, 1]), pot_y, [x_bounds[0], x_bounds[1], z_bounds[0], z_bounds[1]],
             f'PINN prediction on xz-plane (y={y_slice*20:.0f}mm)', cmap_pot,
             cbar_label='Electric potential / V', xlabel='x', ylabel='z',
             vmin=0, vmax=1, cbar_ticks=pot_ticks, xlim=x_bounds, ylim=z_bounds)
    setup_ax(plt.subplot(gs[1, 2]), pot_x, [y_bounds[0], y_bounds[1], z_bounds[0], z_bounds[1]],
             f'PINN prediction on yz-plane (x={x_slice*20:.0f}mm)', cmap_pot,
             cbar_label='Electric potential / V', xlabel='y', ylabel='z',
             vmin=0, vmax=1, cbar_ticks=pot_ticks, xlim=y_bounds, ylim=z_bounds)

    # Row 2: FEM
    setup_ax(plt.subplot(gs[2, 0]), fem_xy, [x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]],
             f'FEM solution on xy-plane (z={z_slice*20:.0f}mm)', cmap_pot,
             cbar_label='Electric potential / V', xlabel='x', ylabel='y',
             vmin=0, vmax=1, cbar_ticks=pot_ticks, xlim=x_bounds, ylim=y_bounds)
    setup_ax(plt.subplot(gs[2, 1]), fem_xz, [x_bounds[0], x_bounds[1], z_bounds[0], z_bounds[1]],
             f'FEM solution on xz-plane (y={y_slice*20:.0f}mm)', cmap_pot,
             cbar_label='Electric potential / V', xlabel='x', ylabel='z',
             vmin=0, vmax=1, cbar_ticks=pot_ticks, xlim=x_bounds, ylim=z_bounds)
    setup_ax(plt.subplot(gs[2, 2]), fem_yz, [y_bounds[0], y_bounds[1], z_bounds[0], z_bounds[1]],
             f'FEM solution on yz-plane (x={x_slice*20:.0f}mm)', cmap_pot,
             cbar_label='Electric potential / V', xlabel='y', ylabel='z',
             vmin=0, vmax=1, cbar_ticks=pot_ticks, xlim=y_bounds, ylim=z_bounds)

    # Row 3: Error
    setup_ax(plt.subplot(gs[3, 0]), err_z, [x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]],
             f'Absolute Error on xy-plane (z={z_slice*20:.0f}mm)', cmap_err,
             cbar_label='Absolute error / V', xlabel='x', ylabel='y',
             xlim=x_bounds, ylim=y_bounds)
    setup_ax(plt.subplot(gs[3, 1]), err_y, [x_bounds[0], x_bounds[1], z_bounds[0], z_bounds[1]],
             f'Absolute Error on xz-plane (y={y_slice})', cmap_err,
             cbar_label='Absolute error / V', xlabel='x', ylabel='z',
             xlim=x_bounds, ylim=z_bounds)
    setup_ax(plt.subplot(gs[3, 2]), err_x, [y_bounds[0], y_bounds[1], z_bounds[0], z_bounds[1]],
             f'Absolute Error on yz-plane (x={x_slice*20:.0f}mm)', cmap_err,
             cbar_label='Absolute error / V', xlabel='y', ylabel='z',
             xlim=y_bounds, ylim=z_bounds)

    # Row 4: Relative Error
    setup_ax(plt.subplot(gs[4, 0]), rel_err_z * 100, [x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]],
             f'Relative Error on xy-plane (z={z_slice*20:.0f}mm)', cmap_err,
             cbar_label='Relative error / %', xlabel='x', ylabel='y',
             xlim=x_bounds, ylim=y_bounds)
    setup_ax(plt.subplot(gs[4, 1]), rel_err_y * 100, [x_bounds[0], x_bounds[1], z_bounds[0], z_bounds[1]],
             f'Relative Error on xz-plane (y={y_slice*20:.0f}mm)', cmap_err,
             cbar_label='Relative error / %', xlabel='x', ylabel='z',
             xlim=x_bounds, ylim=z_bounds)
    setup_ax(plt.subplot(gs[4, 2]), rel_err_x * 100, [y_bounds[0], y_bounds[1], z_bounds[0], z_bounds[1]],
             f'Relative Error on yz-plane (x={x_slice*20:.0f}mm)', cmap_err,
             cbar_label='Relative error / %', xlabel='y', ylabel='z',
             xlim=y_bounds, ylim=z_bounds)

    mre = np.mean(rel_error) * 100
    max_re = np.nanmax(rel_error) * 100
    plt.suptitle(
        f'Accuracy Metrics: MAE={mae:.4f} | MaxAE={max_ae:.4f} | MRE={mre:.4f}% | Max RE={max_re:.2f}%',
        fontsize=20, y=1.02
    )
    suffix = f"_epoch{epoch}" if epoch is not None else ""
    fname = f"pinn_vs_fem{suffix}.svg"
    if save_dir is not None:
        fname = os.path.join(save_dir, fname)
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()

    print("=" * 55)
    print(f"{'ACCURACY METRICS':^55}")
    print("=" * 55)
    print(f"  MAE:      {mae:.6f} V")
    print(f"  RMSE:     {rmse:.6f} V")
    print(f"  MaxAE:    {max_ae:.6f} V")
    print(f"  MRE:      {mre:.4f} %")
    print(f"  Max RE:   {max_re:.4f} %")
    print("-" * 55)
    print(f"  wMAE:     {wmae:.6f} V  (contact-weighted)")
    print(f"  wMRE:     {wmre:.4f} %  (contact-weighted)")
    print("=" * 55)

    fig_w, axes_w = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    for ax, grid, extent, title, xlabel, ylabel in [
        (axes_w[0], w_z, [x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]],
         f'xy-plane (z={z_slice*20:.0f}mm)', 'x', 'y'),
        (axes_w[1], w_y, [x_bounds[0], x_bounds[1], z_bounds[0], z_bounds[1]],
         f'xz-plane (y={y_slice*20:.0f}mm)', 'x', 'z'),
        (axes_w[2], w_x, [y_bounds[0], y_bounds[1], z_bounds[0], z_bounds[1]],
         f'yz-plane (x={x_slice*20:.0f}mm)', 'y', 'z'),
    ]:
        im = ax.imshow(grid, extent=extent, origin='lower', cmap='viridis', vmin=0, vmax=1, aspect='equal')
        ax.set_title(f'Weight map — {title}', fontsize=14)
        ax.set_xlabel(f'{xlabel} / mm', fontsize=12)
        ax.set_ylabel(f'{ylabel} / mm', fontsize=12)
        scale_mm = 20
        x_ticks = np.linspace(extent[0], extent[1], 5)
        y_ticks = np.linspace(extent[2], extent[3], 5)
        ax.set_xticks(x_ticks); ax.set_xticklabels([f'{int(t*scale_mm)}' for t in x_ticks])
        ax.set_yticks(y_ticks); ax.set_yticklabels([f'{int(t*scale_mm)}' for t in y_ticks])
        fig_w.colorbar(im, ax=ax, shrink=0.7, label='Weight')

    fname_w = f"weight_map{suffix}.svg"
    if save_dir is not None:
        fname_w = os.path.join(save_dir, fname_w)
    fig_w.savefig(fname_w, dpi=150, bbox_inches='tight')
    plt.show()