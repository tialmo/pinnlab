import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from netgen.meshing import ImportMesh
import meshio
from scipy.interpolate import griddata
from matplotlib.ticker import MultipleLocator

def plot_fem_solution(mesh):
    
    # Extract points (coordinates)
    points = mesh.points
    
    # Find the min and max of the coordinates
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    plot_radius = 0.0325
    
    
    
    
    #points = scaled_points
    potential = mesh.point_data['potential_real']
    
    # Create regular 3D grid
    n_points = 201
    x = np.linspace(points[:,0].min(), points[:,0].max(), n_points)
    y = np.linspace(points[:,1].min(), points[:,1].max(), n_points)
    z = np.linspace(points[:,2].min(), points[:,2].max(), n_points)
    
    # Get center point for slices
    center = (points.max(axis=0) + points.min(axis=0)) / 2
    
    # Create 2D meshgrids for each plane
    X_xy, Y_xy = np.meshgrid(x, y)
    X_xz, Z_xz = np.meshgrid(x, z)
    Y_yz, Z_yz = np.meshgrid(y, z)
    
    # Create points for interpolation in each plane
    xy_points = np.array([[x, y, center[2]+0.1] for x, y in zip(X_xy.flatten(), Y_xy.flatten())])
    xz_points = np.array([[x, center[1], z] for x, z in zip(X_xz.flatten(), Z_xz.flatten())])
    yz_points = np.array([[center[0], y, z] for y, z in zip(Y_yz.flatten(), Z_yz.flatten())])
    
    # Interpolate values
    xy_vals = griddata(points, potential, xy_points, method='linear')
    xz_vals = griddata(points, potential, xz_points, method='linear')
    yz_vals = griddata(points, potential, yz_points, method='linear')
    
    # Reshape back to 2D
    xy_slice = xy_vals.reshape(n_points, n_points)
    xz_slice = xz_vals.reshape(n_points, n_points)
    yz_slice = yz_vals.reshape(n_points, n_points)
    
    
    
    # Set global font size
    #plt.rcParams.update({'font.size': 14})
    
    # Create figure with adjusted spacing
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)  # spacing between subplots
    
    # Choose a colormap: 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 
    #                    'coolwarm', 'RdBu', 'jet', 'hot', etc.
    cmap = 'coolwarm'
    titlefont=20
    
    im0 = axes[0].imshow(xy_slice, extent=[x.min(), x.max(), y.min(), y.max()],
                         origin='lower', aspect='equal', cmap=cmap)
    axes[0].add_patch(plt.Circle((0.0, 0.0), plot_radius, color='white'))
    axes[0].set_title('Potential [V] at XY plane (z=0.1)'.format(center[2]), fontsize=titlefont, pad=15)
    
    im1 = axes[1].imshow(xz_slice, extent=[x.min(), x.max(), z.min(), z.max()],
                         origin='lower', aspect='equal', cmap=cmap)
    axes[1].set_title('Potential [V] at XZ plane (y={:.0f})'.format(center[1]), fontsize=titlefont, pad=15)
    axes[1].add_patch(plt.Circle((0.0, plot_radius), plot_radius, color='white'))
    axes[1].add_patch(plt.Rectangle((0 - plot_radius, plot_radius), 2 * plot_radius, 1, color='white'))
    
    im2 = axes[2].imshow(yz_slice, extent=[y.min(), y.max(), z.min(), z.max()],
                         origin='lower', aspect='equal', cmap=cmap)
    axes[2].set_title('Potential [V] at YZ plane  (x={:.0f})'.format(center[0]), fontsize=titlefont, pad=15)
    axes[2].add_patch(plt.Circle((0.0, plot_radius), plot_radius, color='white'))
    axes[2].add_patch(plt.Rectangle((0 - plot_radius, plot_radius), 2 * plot_radius, 1, color='white'))
    
    for ax, im in zip(axes, [im0, im1, im2]):
        ax.tick_params(axis='both', labelsize=20)# tick label size
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=20)  # colorbar tick size
        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.yaxis.set_major_locator(MultipleLocator(1.0))
        cbar.ax.yaxis.set_major_locator(MultipleLocator(0.5))
    
    axislabel=20
    labelpadside=-10
    labelpadbot=-1
    axes[0].set_xlabel('x', fontsize=axislabel, labelpad=labelpadbot)
    axes[0].set_ylabel('y', fontsize=axislabel, labelpad=labelpadside)
    
    axes[1].set_xlabel('x', fontsize=axislabel, labelpad=labelpadbot)
    axes[1].set_ylabel('z', fontsize=axislabel, labelpad=labelpadside)
    
    axes[2].set_xlabel('y', fontsize=axislabel, labelpad=labelpadbot)
    axes[2].set_ylabel('z', fontsize=axislabel, labelpad=labelpadside)
    
    plt.tight_layout()
    plt.show()


def plot_prediction(solver, radius, X_f):

    plot_radius = radius
    
    # Adjustable slice coordinates for x, y, and z axes
    x_slice = 0  # Set the desired x-coordinate for the x-slice
    y_slice = 0  # Set the desired y-coordinate for the y-slice
    z_slice = 0.1  # Set the desired z-coordinate for the z-slice
    
    # Generate cross-sections from collocation points without tolerance
    cross_section_z = X_f[:, :3].numpy()[X_f.numpy()[:, 2] == z_slice]
    cross_section_x = X_f[:, :3].numpy()[X_f.numpy()[:, 0] == x_slice]
    cross_section_y = X_f[:, :3].numpy()[X_f.numpy()[:, 1] == y_slice]
    
    # Predict the potential distribution for each cross-section
    predicted_potential_z = solver.predict(cross_section_z)
    predicted_potential_x = solver.predict(cross_section_x)
    predicted_potential_y = solver.predict(cross_section_y)
    
    # Extract conductivity values
    conductivity_z = X_f[:, 3].numpy()[X_f.numpy()[:, 2] == z_slice]
    conductivity_x = X_f[:, 3].numpy()[X_f.numpy()[:, 0] == x_slice]
    conductivity_y = X_f[:, 3].numpy()[X_f.numpy()[:, 1] == y_slice]
    
    # Create smooth grids for plotting
    resolution = 201
    x_grid_z, y_grid_z = np.linspace(cross_section_z[:, 0].min(), cross_section_z[:, 0].max(), resolution), np.linspace(cross_section_z[:, 1].min(), cross_section_z[:, 1].max(), resolution)
    x_grid_x, z_grid_x = np.linspace(cross_section_x[:, 1].min(), cross_section_x[:, 1].max(), resolution), np.linspace(cross_section_x[:, 2].min(), cross_section_x[:, 2].max(), resolution)
    y_grid_y, z_grid_y = np.linspace(cross_section_y[:, 0].min(), cross_section_y[:, 0].max(), resolution), np.linspace(cross_section_y[:, 2].min(), cross_section_y[:, 2].max(), resolution)
    
    X_grid_z, Y_grid_z = np.meshgrid(x_grid_z, y_grid_z)
    X_grid_x, Z_grid_x = np.meshgrid(x_grid_x, z_grid_x)
    Y_grid_y, Z_grid_y = np.meshgrid(y_grid_y, z_grid_y)
    
    smooth_potential_z = griddata((cross_section_z[:, 0], cross_section_z[:, 1]), predicted_potential_z, (X_grid_z, Y_grid_z), method='cubic')
    smooth_potential_x = griddata((cross_section_x[:, 1], cross_section_x[:, 2]), predicted_potential_x, (X_grid_x, Z_grid_x), method='cubic')
    smooth_potential_y = griddata((cross_section_y[:, 0], cross_section_y[:, 2]), predicted_potential_y, (Y_grid_y, Z_grid_y), method='cubic')
    
    smooth_conductivity_z = griddata((cross_section_z[:, 0], cross_section_z[:, 1]), conductivity_z, (X_grid_z, Y_grid_z), method='cubic')
    smooth_conductivity_x = griddata((cross_section_x[:, 1], cross_section_x[:, 2]), conductivity_x, (X_grid_x, Z_grid_x), method='cubic')
    smooth_conductivity_y = griddata((cross_section_y[:, 0], cross_section_y[:, 2]), conductivity_y, (Y_grid_y, Z_grid_y), method='cubic')
    
    print("done")
    
    from matplotlib import gridspec
    from matplotlib.lines import Line2D
    
    # Plot the results
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3)
    
    # Plot the cross-sectional view at the specified z-slice with predicted potential
    ax0 = plt.subplot(gs[0, 0])
    sc0 = ax0.imshow(smooth_potential_z, extent=[x_grid_z.min(), x_grid_z.max(), y_grid_z.min(), y_grid_z.max()], origin='lower', cmap='viridis', aspect='auto')
    ax0.add_patch(plt.Circle((0.0, 0.0), plot_radius, color='white'))
    #ax0.add_patch(plt.Circle((0.0, 0.0), plot_radius, fill=False, edgecolor='red', linewidth=2, label='Contact'))
    ax0.set_title(f'Potential Distribution at z = {z_slice}')
    ax0.set_xlabel('X')
    ax0.set_ylabel('Y')
    #ax0.legend()
    cbar0 = plt.colorbar(sc0, ax=ax0)
    cbar0.set_label('Predicted Potential')
    
    # Plot the conductivity for the z-slice
    ax3 = plt.subplot(gs[1, 0])
    sc3 = ax3.imshow(smooth_conductivity_z, extent=[x_grid_z.min(), x_grid_z.max(), y_grid_z.min(), y_grid_z.max()], origin='lower', cmap='plasma', aspect='auto')
    ax3.set_title(f'Conductivity at z = {z_slice}')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    cbar3 = plt.colorbar(sc3, ax=ax3)
    cbar3.set_label('Conductivity (Sigma)')
    
    # Plot the cross-sectional view at the specified x-slice with predicted potential
    ax1 = plt.subplot(gs[0, 1])
    sc1 = ax1.imshow(smooth_potential_x, extent=[x_grid_x.min(), x_grid_x.max(), z_grid_x.min(), z_grid_x.max()], origin='lower', cmap='viridis', aspect='auto')
    ax1.add_patch(plt.Circle((0.0, plot_radius), plot_radius, color='white'))
    #ax1.add_patch(plt.Circle((0.0, 0.0), plot_radius, fill=False, edgecolor='red', linewidth=2, label='Contact'))
    ax1.add_patch(plt.Rectangle((0 - 0.0325, plot_radius),
                                     2 * plot_radius, 1, color='white'))
    
    #ax1.add_line(Line2D([0 - plot_radius, 0 - plot_radius], [0, 1], color='blue', linewidth=2, label='Shaft'))
    #ax1.add_line(Line2D([0 + plot_radius, 0 + plot_radius], [0, 1], linewidth=2, color='blue'))
    ax1.set_title(f'Potential Distribution at x = {x_slice}')
    ax1.set_xlabel('Y')
    ax1.set_ylabel('Z')
    #ax1.legend()
    cbar1 = plt.colorbar(sc1, ax=ax1)
    cbar1.set_label('Predicted Potential')
    
    # Plot the conductivity for the x-slice
    ax4 = plt.subplot(gs[1, 1])
    sc4 = ax4.imshow(smooth_conductivity_x, extent=[x_grid_x.min(), x_grid_x.max(), z_grid_x.min(), z_grid_x.max()], origin='lower', cmap='plasma', aspect='auto')
    ax4.set_title(f'Conductivity at x = {x_slice}')
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Z')
    cbar4 = plt.colorbar(sc4, ax=ax4)
    cbar4.set_label('Conductivity (Sigma)')
    
    # Plot the cross-sectional view at the specified y-slice with predicted potential
    ax2 = plt.subplot(gs[0, 2])
    sc2 = ax2.imshow(smooth_potential_y, extent=[y_grid_y.min(), y_grid_y.max(), z_grid_y.min(), z_grid_y.max()], origin='lower', cmap='viridis', aspect='auto')
    ax2.add_patch(plt.Circle((0.0, plot_radius), plot_radius, color='white'))
    #ax2.add_patch(plt.Circle((0.0, 0.0), plot_radius, fill=False, edgecolor='red', linewidth=2, label='Contact'))
    ax2.add_patch(plt.Rectangle((0 - plot_radius, plot_radius),
                                     2 * plot_radius, 1, color='white'))
    
    #ax2.add_line(Line2D([0 - plot_radius, 0 - plot_radius], [0, 1], color='blue', linewidth=2, label='Shaft'))
    #ax2.add_line(Line2D([0 + plot_radius, 0 + plot_radius], [0, 1], linewidth=2, color='blue'))
    ax2.set_title(f'Potential Distribution at y = {y_slice}')
    #ax2.legend()
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    cbar2 = plt.colorbar(sc2, ax=ax2)
    cbar2.set_label('Predicted Potential')
    
    # Plot the conductivity for the y-slice
    ax5 = plt.subplot(gs[1, 2])
    sc5 = ax5.imshow(smooth_conductivity_y, extent=[y_grid_y.min(), y_grid_y.max(), z_grid_y.min(), z_grid_y.max()], origin='lower', cmap='plasma', aspect='auto')
    ax5.set_title(f'Conductivity at y = {y_slice}')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Z')
    cbar5 = plt.colorbar(sc5, ax=ax5)
    cbar5.set_label('Conductivity (Sigma)')
    
    # Show the plot
    plt.tight_layout()
    plt.show()

def calc_accuracy(solver, X_f, mesh, plot_radius):
    
    # Extract the true solution
    true_points = mesh.points                  # shape: (N_true, 3)
    true_values = mesh.point_data['potential_real']  # shape: (N_true,)
    
    # Example: Your PINN input data, shape: (N_pred, 4) -> [x, y, z, conductivity]
    # X_f is presumably a tf.Tensor or a numpy array. If it's a tensor, convert to np:
    # X_f_np = X_f.numpy()
    
    # For illustration, we assume X_f_np is already available:
    # (If your X_f is a numpy array, just rename it accordingly.)
    X_f_np = X_f.numpy()   # shape: (N_pred, 4)
    
    # The first 3 columns are x,y,z
    pred_points = X_f_np[:, :3]   # shape: (N_pred, 3)
    
    # ----------------------------------------------------------------
    # PREDICT the solution from the PINN
    # ----------------------------------------------------------------
    pred_values = solver.predict(pred_points)#.ravel()  # shape: (N_pred,)
    
    # ----------------------------------------------------------------
    # INTERPOLATE the FEM solution onto the same points as the PINN
    # ----------------------------------------------------------------
    interp_true_values = griddata(
        points=true_points,       # shape (N_true, 3)
        values=true_values,       # shape (N_true,)
        xi=pred_points,           # shape (N_pred, 3)
        method='linear'           # or 'nearest'; 'cubic' is only 2D
    )
    
    abs_error = np.abs(interp_true_values - pred_values)
    mse  = np.mean(abs_error**2)
    rmse = np.sqrt(mse)
    mae  = np.mean(abs_error)
    
    print("Mean Absolute Error (MAE) =", mae)
    print("Root Mean Squared Error (RMSE) =", rmse)


    
    # maximum absolute error
    max_ae = np.max(abs_error)
    
    # relative error (with protection against division by zero)
    epsilon = 1e-10  # Small value to avoid division by zero
    nonzero_mask = np.abs(interp_true_values) > epsilon
    rel_error = np.zeros_like(interp_true_values)
    rel_error[nonzero_mask] = abs_error[nonzero_mask] / np.abs(interp_true_values[nonzero_mask])
    
    # mean and maximum relative error
    mre = np.mean(rel_error)
    max_re = np.max(rel_error)

    ## # PLOT --------------------------------------------------------------------
    # Example inputs (already computed):
    # pred_points: shape (N, 3) -> each row is (x, y, z) 
    # abs_error: shape (N,)     -> absolute error at each point
    # --------------------------------------------------------------------
    # For example:
    # abs_error = np.abs(interp_true_values - pred_values)
    
    # Choose the planes you want to slice
    x_slice = 0.0
    y_slice = 0.0
    z_slice = 0.0
    tolerance = 1e-5  # determines how close points must be to the plane
    
    # --------------------------------------------------------------------
    # Prepare figure with 3 subplots
    # --------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    
    
    # ==================== 3) Slice at z = z_slice =====================
    mask_z = np.isclose(pred_points[:, 2], z_slice, atol=tolerance)
    points_z = pred_points[mask_z]
    error_z  = abs_error[mask_z]
    
    
    x_min, x_max = points_z[:,0].min(), points_z[:,0].max()
    y_min, y_max = points_z[:,1].min(), points_z[:,1].max()
    nx, ny = 200, 200
    x_grid = np.linspace(x_min, x_max, nx)
    y_grid = np.linspace(y_min, y_max, ny)
    Xg, Yg = np.meshgrid(x_grid, y_grid)
    
    error_grid_z = griddata(
        (points_z[:,0], points_z[:,1]),  # (x,y)
        error_z,
        (Xg, Yg),
        method='linear'
    )
    
    im0 = axes[0].imshow(
        error_grid_z,
        extent=[x_min, x_max, y_min, y_max],
        origin='lower',
        aspect='auto',
        cmap='viridis'
    )
    axes[0].set_title(f'Error Map at z={z_slice}')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0], label='Absolute Error')
    axes[0].add_patch(plt.Circle((0.0, 0.0), plot_radius, color='white'))
        
    # ==================== 2) Slice at y = y_slice =====================
    mask_y = np.isclose(pred_points[:, 1], y_slice, atol=tolerance)
    points_y = pred_points[mask_y] 
    error_y  = abs_error[mask_y]
    
    
    x_min, x_max = points_y[:,0].min(), points_y[:,0].max()
    z_min, z_max = points_y[:,2].min(), points_y[:,2].max()
    nx, nz = 200, 200
    x_grid = np.linspace(x_min, x_max, nx)
    z_grid = np.linspace(z_min, z_max, nz)
    Xg, Zg = np.meshgrid(x_grid, z_grid)
    
    error_grid_y = griddata(
        (points_y[:,0], points_y[:,2]),  # (x,z)
        error_y,
        (Xg, Zg),
        method='linear'
    )
    
    im1 = axes[1].imshow(
        error_grid_y,
        extent=[x_min, x_max, z_min, z_max],
        origin='lower',
        aspect='auto',
        cmap='viridis'
    )
    axes[1].set_title(f'Error Map at y={y_slice}')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('z')
    plt.colorbar(im1, ax=axes[1], label='Absolute Error')
    axes[1].add_patch(plt.Circle((0.005, 0.0), plot_radius, color='white'))
    axes[1].add_patch(plt.Rectangle((0 - plot_radius, 0.0),
                                     2 * plot_radius, 1, color='white'))
    
    
    # ==================== 1) Slice at x = x_slice =====================
    mask_x = np.isclose(pred_points[:, 0], x_slice, atol=tolerance)
    points_x = pred_points[mask_x]       # shape (Nx, 3)
    error_x  = abs_error[mask_x]         # shape (Nx, )
    
    # We'll plot in (y,z) coordinates
    # Create a grid in y,z for interpolation
    
    y_min, y_max = points_x[:,1].min(), points_x[:,1].max()
    z_min, z_max = points_x[:,2].min(), points_x[:,2].max()
    ny, nz = 200, 200  # resolution of the slice grid
    y_grid = np.linspace(y_min, y_max, ny)
    z_grid = np.linspace(z_min, z_max, nz)
    Yg, Zg = np.meshgrid(y_grid, z_grid)
    
    # Interpolate abs_error onto the (Yg, Zg) grid
    error_grid_x = griddata(
        (points_x[:,1], points_x[:,2]),  # (y,z)
        error_x,
        (Yg, Zg),
        method='linear'
    )
    
    im2 = axes[2].imshow(
        error_grid_x,
        extent=[y_min, y_max, z_min, z_max],
        origin='lower',
        aspect='auto',
        cmap='viridis'
    )
    axes[2].set_title(f'Error Map at x={x_slice}')
    axes[2].set_xlabel('y')
    axes[2].set_ylabel('z')
    plt.colorbar(im2, ax=axes[2], label='Absolute Error')
    axes[2].add_patch(plt.Circle((0.005, 0.0), plot_radius, color='white'))
    axes[2].add_patch(plt.Rectangle((0 - plot_radius, 0.0),
                                     2 * plot_radius, 1, color='white'))
    
    plt.tight_layout()
    plt.show()

    
    # Print all error metrics
    print("Mean Absolute Error (MAE) =", mae)
    print("Root Mean Squared Error (RMSE) =", rmse)
    print("Maximum Absolute Error (Max AE) =", max_ae)
    print("Mean Relative Error (MRE) =", mre)
    print("Maximum Relative Error (Max RE) =", max_re)

def compare_with_fem(solver, radius):

    mesh = meshio.read("remapped_binary_potential.vtu")
    
    # collocation data
    output_npz = "remapped_scaled_data.npz"
    loaded_data = np.load(output_npz)
    
    
    
    #X_f = sampler.collocation_points
    X_f = loaded_data['flat_coordinates']
    X_f = tf.constant(X_f, dtype=tf.float32) ###
    print("Collocation Points Shape:", X_f.shape)

    plot_prediction(solver, radius, X_f)
    plot_fem_solution(mesh)
    calc_accuracy(solver, X_f, mesh, radius)