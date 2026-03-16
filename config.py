"""
config.py — central configuration for the LA-PINN DBS electrode solver.

All hyperparameters, geometry, and training settings live here.
Import this in main.py; nothing else should define these values.
"""

# ---------------------------------------------------------------------------
# PDE
# ---------------------------------------------------------------------------
bound = 0.5
PDE = {
    # Formulation order:
    #   2 — second-order: network outputs u (scalar), PDE is Δu=0 per region,
    #       conductivity enters only through the interface flux condition.
    #   1 — first-order (coming): network outputs [u, qx, qy, qz] where
    #       q = σ∇u, avoiding differentiation of σ. Output dim is set
    #       automatically (1 if order=2, 4 if order=1).
    'order': 2,

    # Spatial domain bounds (normalized coordinates).
    # Adjust to zoom into a sub-region without touching the physics.
    'domain_bounds': {
        'x': [-bound, bound],
        'y': [-bound, bound],
        'z': [-bound, bound],
    },

    # Geometry and boundary conditions (normalized coordinates).
    'params': {
        # Active contact 1 (Dirichlet, V = contact1_value)
        'contact1_cylinder_radius': 0.0325,
        'contact1_cylinder_height': 0.075,
        'contact1_cylinder_center': [0.0, 0.0, 0.0925],
        'contact1_value': 1.0,

        # Active contact 2 (Dirichlet, V = contact2_value)
        'contact2_cylinder_radius': 0.0325,
        'contact2_cylinder_height': 0.075,
        'contact2_cylinder_center': [0.0, 0.0, 0.1925],
        'contact2_value': 0.0,

        # Insulating shaft + hemispherical cap (Neumann, ∂_n u = 0)
        'neumann_cylinder_radius': 0.0325,
        'neumann_cylinder_height': 1.0225,
        'neumann_cylinder_center': [0.0, 0.0, 0.5275],
        'neumann_value': 0.0,
        'with_half_sphere': True,

        # Outer cubic domain boundary (Neumann, ∂_n u = 0)
        'cube_neumann_value': 0.0,
    },
}

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL = {
    # Network width (neurons per hidden layer).
    # LAN width automatically mirrors this.
    'width': 120,

    # Number of hidden layers.
    # LAN depth automatically mirrors this.
    'hidden_layers': 5,

    # Main network body:
    #   'encoder' — dual-encoder MLP: two encoder branches (enc1, enc2) computed
    #               once from the input; every hidden layer h combined as
    #               h*enc1 + (1-h)*enc2. Improves gradient flow in deep networks.
    #   'mlp'     — plain tanh MLP, no encoder combination.
    'architecture': 'mlp',

    # Fourier feature embedding (orthogonal to architecture):
    #   False — raw coordinates fed directly into the network.
    #   True  — coordinates first embedded as γ(x) = [cos(2πBx), sin(2πBx)]
    #           via a fixed random B ~ N(0, fourier_sigma²).
    #           Addresses spectral bias for high-frequency solutions.
    'use_fourier': False,
    'fourier_features': 128,   # embedding output dim = 2 * this
    'fourier_sigma': 2.0,      # larger → higher-frequency features in B
}

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
TRAINING = {
    'epochs': 50000,
    'evaluate_interval': 5000,
    'display_interval': 200,    # print loss every N epochs
    'accuracy_interval': 500,   # compute FEM comparison error every N epochs

    # Loss weighting strategy:
    #   'lapinn'   — Loss-Attentional Networks (one LAN per loss term, adversarial
    #                training). LAN architecture mirrors MODEL width/hidden_layers.
    #   'adaptive' — learnable scalar weight per loss term, gradient-based update.
    #   'classic'  — fixed scalar weights defined in classic_weights below.
    'loss_weighting': 'lapinn', # or 'adaptive' or 'classic'
    'classic_weights': {
        'w_r':    1.0,
        'w_n':    1.0,
        'w_b_c1': 1.0,
        'w_b_c2': 1.0,
        'w_if':   1.0,
        'w_d':    1.0,
    },

    # Collocation point counts per epoch (randomly resampled each epoch)
    'n_collocation': 10000,
    'n_b_cube':      2400,
    'n_b_contact1':  600,
    'n_b_contact2':  600,
    'n_b_neumann':   600,
    'n_interface':   5000,

    # FEM reference data (used for optional validation / data loss term)
    'fem_data_path':   'remapped_binary_potential.vtu',
    'sigma_data_path': 'remapped_scaled_data.npz',
    'fem_n_sample':    10000,
}


# ---------------------------------------------------------------------------
# Derived quantities (computed from the above — do not edit directly)
# ---------------------------------------------------------------------------

# Convenience alias so main.py can do: from config import DOMAIN
DOMAIN = PDE['domain_bounds']

# Output dim is determined by PDE order: 1 scalar u (order=2), or [u,qx,qy,qz] (order=1)
_OUTPUT_DIM = 1 if PDE['order'] == 2 else 4
_W = MODEL['width']
_H = MODEL['hidden_layers']

# Layer lists derived from MODEL — LAN mirrors main network width and depth
LAYERS_MAIN = [3]  + [_W] * _H + [_OUTPUT_DIM]
LAYERS_LAN  = [1]  + [_W] * _H + [1]