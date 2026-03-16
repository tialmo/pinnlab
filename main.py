import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   # Disable GPU

from solver import LaplaceCylinderLAPINNSolver
from config import DOMAIN, PDE, MODEL, TRAINING, LAYERS_MAIN, LAYERS_LAN


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_bipolar_laplace_with_fem_data():
    """
    Run the full LA-PINN training experiment:
      1. Build the solver
      2. Load FEM reference data
      3. Generate training points
      4. Train
      5. Plot results
    """
    np.random.seed(42)
    tf.random.set_seed(42)

    # --- Build solver ---
    laplace_solver = LaplaceCylinderLAPINNSolver(
        layers_main=LAYERS_MAIN,
        layers_lan=LAYERS_LAN,
        domain_bounds=DOMAIN['bounds'],
        pde_params=PDE['params'],
        architecture=MODEL['architecture'],
        use_fourier=MODEL['use_fourier'],
        fourier_features=MODEL['fourier_features'],
        fourier_sigma=MODEL['fourier_sigma'],
    )

    ff_str = (f" + Fourier (features={MODEL['fourier_features']}, "
              f"sigma={MODEL['fourier_sigma']})"
              if MODEL['use_fourier'] else "")
    print(f"PDE order     : {PDE['order']}")
    print(f"Architecture  : {MODEL['architecture']}{ff_str}")
    print(f"Loss weighting: {TRAINING['loss_weighting']}")
    print(f"Layers main   : {LAYERS_MAIN}")
    print(f"Layers LAN    : {LAYERS_LAN}")

    # --- Load FEM reference data ---
    try:
        fem_points, fem_values = laplace_solver.load_fem_data(
            mesh_file_path=TRAINING['fem_data_path'],
            n_sample=TRAINING['fem_n_sample'],
            scale_factor=1.0,
        )
        print("FEM data loaded successfully")
    except Exception as e:
        print(f"Could not load FEM data: {e}")
        print("Proceeding without FEM data training...")
        laplace_solver.x_d = tf.zeros((0, 3), dtype=tf.float32)
        laplace_solver.u_d = tf.zeros((0, 1), dtype=tf.float32)

    # --- Generate training points ---
    laplace_solver.generate_training_points(
        n_r=TRAINING['n_collocation'],
        n_b_cube=TRAINING['n_b_cube'],
        n_b_contact1=TRAINING['n_b_contact1'],
        n_b_contact2=TRAINING['n_b_contact2'],
        n_b_neumann=TRAINING['n_b_neumann'],
    )

    # --- Train ---
    history = laplace_solver.train(
        epochs=TRAINING['epochs'],
        display_interval=TRAINING['display_interval'],
        accuracy_interval=TRAINING['accuracy_interval'],
    )

    # --- Plot training history ---
    _plot_training_history(history)

    # --- Plot solution slices ---
    laplace_solver.plot_solution_slices(
        n_points=50,
        save_path='bipolar_laplace_solution_slices.png',
    )

    # --- Final accuracy ---
    final_mse = laplace_solver.compute_prediction_accuracy(n_points=5000)
    print(f"Final prediction MSE: {final_mse:.6e}")

    return laplace_solver, history


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def save_model(solver, save_dir):
    """
    Save all LA-PINN networks to disk.

    For Fourier mode (use_fourier=True) the random frequency matrix B is also
    saved as 'fourier_B.npy' — it must be present when reloading because the
    Keras Lambda layers embed B as a compile-time constant.

    Args:
        solver:   LaplaceCylinderLAPINNSolver instance (after training)
        save_dir: Target directory (created if it does not exist)
    """
    os.makedirs(save_dir, exist_ok=True)

    solver.main_net.save_weights(os.path.join(save_dir, 'main_net.weights.h5'))
    solver.lan_r.save_weights(   os.path.join(save_dir, 'lan_r.weights.h5'))
    solver.lan_n.save_weights(   os.path.join(save_dir, 'lan_n.weights.h5'))
    solver.lan_b_c1.save_weights(os.path.join(save_dir, 'lan_b_c1.weights.h5'))
    solver.lan_b_c2.save_weights(os.path.join(save_dir, 'lan_b_c2.weights.h5'))
    solver.lan_d.save_weights(   os.path.join(save_dir, 'lan_d.weights.h5'))
    solver.lan_if.save_weights(  os.path.join(save_dir, 'lan_if.weights.h5'))

    if solver.fourier_B is not None:
        np.save(os.path.join(save_dir, 'fourier_B.npy'), solver.fourier_B)
        print(f"Fourier B matrix saved  shape={solver.fourier_B.shape}")

    print(f"Model saved to '{save_dir}'")


def load_model(solver, save_dir):
    """
    Restore all network weights into an already-constructed solver.

    Requirements:
      - The solver must be built with the same architecture as the saved one.
      - generate_training_points() must have been called first so that Keras
        has built concrete weight tensors.
      - For Fourier mode: the solver's fourier_B must match the saved one.

    Args:
        solver:   LaplaceCylinderLAPINNSolver instance
        save_dir: Directory written by save_model()
    """
    b_path = os.path.join(save_dir, 'fourier_B.npy')
    if os.path.exists(b_path):
        saved_B = np.load(b_path)
        if solver.fourier_B is None:
            print("WARNING: saved model used Fourier features but current solver "
                  "was not built with use_fourier=True — weights may not load.")
        elif not np.allclose(saved_B, solver.fourier_B):
            print("WARNING: solver fourier_B differs from saved fourier_B.npy. "
                  "Inference will be incorrect — reload the solver with the saved B.")

    solver.main_net.load_weights(os.path.join(save_dir, 'main_net.weights.h5'))
    solver.lan_r.load_weights(   os.path.join(save_dir, 'lan_r.weights.h5'))
    solver.lan_n.load_weights(   os.path.join(save_dir, 'lan_n.weights.h5'))
    solver.lan_b_c1.load_weights(os.path.join(save_dir, 'lan_b_c1.weights.h5'))
    solver.lan_b_c2.load_weights(os.path.join(save_dir, 'lan_b_c2.weights.h5'))
    solver.lan_d.load_weights(   os.path.join(save_dir, 'lan_d.weights.h5'))
    solver.lan_if.load_weights(  os.path.join(save_dir, 'lan_if.weights.h5'))

    print(f"Model loaded from '{save_dir}'")


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def _plot_training_history(history):
    """Save a three-panel training history figure."""
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 3, 1)
    plt.semilogy(history['total_loss'], label='Total Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (log scale)')
    plt.title('Bipolar LA-PINN Training: Total Loss')
    plt.grid(True, which="both", ls="--")

    plt.subplot(1, 3, 2)
    plt.semilogy(history['loss_r'],    label='Residual Loss (interior)')
    plt.semilogy(history['loss_n'],    label='Neumann Loss (Cube+Shaft)')
    plt.semilogy(history['loss_b_c1'], label='Contact 1 Dirichlet Loss')
    plt.semilogy(history['loss_b_c2'], label='Contact 2 Dirichlet Loss')
    plt.semilogy(history['loss_d'],    label='Data Loss')
    plt.semilogy(history['loss_if'],   label='Interface Flux Loss (lan_if)', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Component Loss (log scale)')
    plt.title('Bipolar LA-PINN Training: Component Losses')
    plt.legend()
    plt.grid(True, which="both", ls="--")

    plt.subplot(1, 3, 3)
    plt.semilogy(history['accuracy_epochs'], history['raw_mse'],
                 'r-o', label='Raw Prediction MSE')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (log scale)')
    plt.title('Bipolar LA-PINN Training: Raw Prediction Accuracy')
    plt.legend()
    plt.grid(True, which="both", ls="--")

    plt.tight_layout()
    plt.savefig('bipolar_laplace_with_fem_data_training_history.png',
                dpi=300, bbox_inches='tight')
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

print("start")
if __name__ == "__main__":
    print("Starting Bipolar LA-PINN training with FEM data...")
    solver, history = run_bipolar_laplace_with_fem_data()
    print("Bipolar experiment completed successfully!")