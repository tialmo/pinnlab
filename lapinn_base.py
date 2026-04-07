import numpy as np
import tensorflow as tf

from modules.networks import build_main_network, build_lan_network
from modules.data_io import load_fem_data as _load_fem_data
from modules.loss_functions import compute_neumann_squared_errors as _compute_neumann_squared_errors

import importlib
import evaluate
import os
import shutil

class LAPINN:
    """
    Base class for Loss-Attentional Physics-Informed Neural Networks (LA-PINN).

    Manages:
      - Main network and six LAN (Loss-Attentional Network) instances
        (lan_r, lan_n, lan_b_c1, lan_b_c2, lan_d, lan_if)
      - Adam optimizers for each network
      - Adversarial training loop (main net minimises, LANs maximise)
      - Generic compute_loss() that calls self.compute_residual() —
        implemented by subclasses.
      - Interface flux residuals always pass through their own dedicated lan_if,
        producing a separate loss_if term (physically correct, always enabled).
      - Main-network architecture selected by two orthogonal settings:
            architecture='encoder' (default) or 'mlp'
            use_fourier=False (default) or True
          These combine freely:
            encoder, no FF  →  original dual-encoder MLP on raw coords
            mlp,     no FF  →  plain tanh MLP on raw coords
            encoder, FF     →  dual-encoder MLP on Fourier embedding
            mlp,     FF     →  plain tanh MLP on Fourier embedding
    """

    def __init__(self, layers_main, layers_lan, domain_bounds, pde_params=None,
                 architecture='encoder',
                 use_fourier=False, fourier_features=256, fourier_sigma=1.0, **kwargs):
        """
        Args:
            layers_main:      List of ints [input, hidden..., output] for main net.
                              layers_main[0] is always the raw coordinate dim (3).
            layers_lan:       List of ints [1, hidden..., 1] for each LAN.
            domain_bounds:    Dict {'x': [min,max], 'y': [min,max], 'z': [min,max]}.
            pde_params:       Dict of PDE / geometry parameters.
            architecture:     'encoder' (default) — dual-encoder MLP.
                              'mlp'               — plain tanh MLP.
            use_fourier:      If True, prepend a fixed random Fourier feature
                              embedding γ(x)=[cos(2πBx), sin(2πBx)] before the
                              chosen architecture.  Orthogonal to architecture.
                              B is stored as self.fourier_B (None when False).
            fourier_features: Number of random frequencies (use_fourier=True only).
                              Embedding output dim = 2 * fourier_features.
            fourier_sigma:    Std-dev of the Gaussian used to sample B
                              (use_fourier=True only).  Larger → higher frequencies.
        """
        self.layers_main     = layers_main
        self.layers_lan      = layers_lan
        self.domain_bounds   = domain_bounds
        self.pde_params      = pde_params if pde_params is not None else {}
        # Interface-flux residuals always routed through their own dedicated LAN.
        # This is the physically correct formulation and is not configurable.
        self.separate_interface_lan = True
        self.architecture    = architecture
        self.use_fourier     = use_fourier
        self.fourier_features = fourier_features
        self.fourier_sigma    = fourier_sigma

        # build_main_network handles all four combinations via architecture + use_fourier
        self.main_net, self.fourier_B = build_main_network(
            layers_main,
            architecture=architecture,
            use_fourier=use_fourier,
            fourier_features=fourier_features,
            fourier_sigma=fourier_sigma,
            input_dim=layers_main[0],
        )
        self.lan_r    = build_lan_network(layers_lan)   # interior PDE residual
        self.lan_n    = build_lan_network(layers_lan)   # Neumann BC
        self.lan_b_c1 = build_lan_network(layers_lan)   # Dirichlet contact 1
        self.lan_b_c2 = build_lan_network(layers_lan)   # Dirichlet contact 2
        self.lan_d    = build_lan_network(layers_lan)   # FEM data loss
        self.lan_if   = build_lan_network(layers_lan)   # interface flux residual

        # Optimizers
        self.main_optimizer    = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.lan_r_optimizer   = tf.keras.optimizers.Adam(learning_rate=2e-4)
        self.lan_n_optimizer   = tf.keras.optimizers.Adam(learning_rate=2e-4)
        self.lan_b_c1_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)
        self.lan_b_c2_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)
        self.lan_d_optimizer   = tf.keras.optimizers.Adam(learning_rate=2e-4)
        self.lan_if_optimizer  = tf.keras.optimizers.Adam(learning_rate=2e-4)

        # Training data placeholders — populated by subclass / load_fem_data
        self.x_r    = None   # Collocation points
        self.x_n    = None   # Neumann boundary points
        self.x_b_c1 = None   # Dirichlet contact 1 points
        self.x_b_c2 = None   # Dirichlet contact 2 points
        self.x_d    = None   # FEM data points
        self.g_n    = None   # Neumann values (normal derivative)
        self.u_b_c1 = None   # Dirichlet contact 1 values
        self.u_b_c2 = None   # Dirichlet contact 2 values
        self.u_d    = None   # FEM data values
        self.x_r_near = tf.zeros((0, 3), dtype=tf.float32)  # overwritten by generate_training_points
        self.n_collocation_sample = kwargs.pop('n_collocation_sample', 10000)
        self.x_r_random = tf.zeros((0, 3), dtype=tf.float32)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_fem_data(self, mesh_file_path, n_sample=None, scale_factor=1.0):
        """
        Delegate to data_io.load_fem_data and store results as self.x_d / self.u_d.

        Args:
            mesh_file_path: Path to the .vtu file
            n_sample:       Max number of FEM points to keep
            scale_factor:   Multiplicative scale on potential values

        Returns:
            (scaled_points, potential_values) numpy arrays
        """
        scaled_points, potential_values, x_d, u_d = _load_fem_data(
            mesh_file_path=mesh_file_path,
            pde_params=self.pde_params,
            is_inside_cylinder_fn=self.is_inside_cylinder,
            n_sample=n_sample,
            scale_factor=scale_factor
        )
        self.x_d = x_d
        self.u_d = u_d
        return scaled_points, potential_values

    # ------------------------------------------------------------------
    # Boundary helpers (delegate to module-level functions)
    # ------------------------------------------------------------------

    def compute_neumann_squared_errors(self, x_n):
        """Squared Neumann residuals — delegates to modules/loss_functions."""
        return _compute_neumann_squared_errors(
            x_n, self.main_net, self.domain_bounds, self.pde_params
        )

    def _contact_pde_weights(self, pts, sigma):
        """Gaussian weights by distance to nearest contact. Mean-normalized."""
        contact_centers = tf.constant([
            self.pde_params['contact1_cylinder_center'],
            self.pde_params['contact2_cylinder_center'],
        ], dtype=tf.float32)
        diffs = pts[:, None, :] - contact_centers[None, :, :]  # (N, K, 3)
        dists = tf.reduce_min(tf.norm(diffs, axis=-1), axis=1)  # (N,)
        w = tf.exp(-(dists**2) / (2 * sigma**2))
        return w / tf.reduce_mean(w)  # mean=1 so overall loss scale unchanged

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_loss(self):
        """
        Compute all weighted loss components for the bipolar electrode system.

        Uses random sub-sampling of collocation and FEM data points each call.
        Calls self.compute_residual() (implemented by subclass) and
        self.compute_interface_flux_residual() if interface pairs are available.

        Interface-flux handling is controlled by self.separate_interface_lan:

          False (original) — r_int and r_if are concatenated into r_all, which
            is passed through lan_r together.  loss_if is always 0.0.

          True  (new)      — r_int goes through lan_r alone; r_if goes through
            the dedicated lan_if.  loss_if is a non-zero separate term and is
            included in total_loss.

        Returns:
            total_loss, loss_r, loss_n, loss_b_c1, loss_b_c2, loss_d, loss_if,
            tape,
            se_r_reshaped, se_n_reshaped, se_b_c1_reshaped, se_b_c2_reshaped,
            se_d_reshaped, se_if_reshaped
        """
        # Random sub-sampling of collocation and data points
        r_indices = tf.random.shuffle(tf.range(tf.shape(self.x_r)[0]))[:10000]
        x_r_sampled = tf.gather(self.x_r, r_indices)

        d_indices = tf.random.shuffle(tf.range(tf.shape(self.x_d)[0]))[:10000]
        x_d_sampled = tf.gather(self.x_d, d_indices)
        u_d_sampled = tf.gather(self.u_d, d_indices)

        # Random sub-sampling of collocation and data points
        #r_indices = tf.random.shuffle(tf.range(tf.shape(self.x_r)[0]))[:7500]
        #d_indices = tf.random.shuffle(tf.range(tf.shape(self.x_d)[0]))[:10000]
        """
        x_r_sampled = tf.concat([
            tf.gather(self.x_r, r_indices),
            tf.gather(self.x_d[:, :3], d_indices[:1500]),  # FEM coords only
        ], axis=0)  # total: 10000 collocation points
        
        #x_d_sampled = tf.gather(self.x_d, d_indices)
        #u_d_sampled = tf.gather(self.u_d, d_indices)
        
        
        n_sample = self.n_collocation_sample
        if n_sample is not None:
            n_near = tf.shape(self.x_r_near)[0]
            n_grid = tf.maximum(n_sample - n_near, 0)
            r_indices = tf.random.shuffle(tf.range(tf.shape(self.x_r)[0]))[:n_grid]
            x_r_sampled = tf.concat([
                tf.gather(self.x_r, r_indices),
                self.x_r_near,
            ], axis=0)
        else:
            x_r_sampled = tf.concat([self.x_r, self.x_r_near], axis=0)

        x_r_sampled = tf.concat([
            tf.gather(self.x_r, r_indices),
            self.x_r_near,
            self.x_r_random,
        ], axis=0)
        
        # FEM data sampling — always needed for loss_d
        d_indices = tf.random.shuffle(tf.range(tf.shape(self.x_d)[0]))[:10000]
        x_d_sampled = tf.gather(self.x_d, d_indices)
        u_d_sampled = tf.gather(self.u_d, d_indices)
        """
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.x_n)

            # --- Interior PDE residual ---
            r_int = self.compute_residual(x_r_sampled)      # (Nr, 1)
            #w = self._contact_pde_weights(x_r_sampled, sigma=0.1) ##########################################################################
            #r_int = tf.reduce_mean(w * r_int) ################################################################ gaussian weighting should be optional

            
            r_if  = tf.zeros((0, 1), dtype=tf.float32)

            # --- Interface flux residual (if pairs were built) ---
            if hasattr(self, "iface_i"):
                n_pairs = tf.shape(self.iface_i)[0]
                if n_pairs > 0:
                    n_if = tf.minimum(tf.constant(5000, dtype=tf.int32), n_pairs)
                    if_idx = tf.random.shuffle(tf.range(n_pairs))[:n_if]

                    ii  = tf.gather(self.iface_i, if_idx)
                    jj  = tf.gather(self.iface_j, if_idx)
                    nij = tf.gather(self.iface_n, if_idx)

                    xi = tf.gather(self.x_r, ii)
                    xj = tf.gather(self.x_r, jj)
                    si = tf.gather(self.sigma_r, ii)
                    sj = tf.gather(self.sigma_r, jj)

                    r_if = self.compute_interface_flux_residual(xi, xj, si, sj, nij)

            # Interior residual → lan_r
            # Interface residual → dedicated lan_if (always separate: physically correct)
            se_if_reshaped = tf.zeros((0, 1), dtype=tf.float32)
            loss_if = tf.constant(0.0, dtype=tf.float32)

            se_r_int = tf.square(r_int)
            se_r_reshaped = tf.reshape(se_r_int, [-1, 1])
            loss_r = tf.reduce_mean(self.lan_r(se_r_reshaped))

            if tf.size(r_if) > 0:
                se_if = tf.square(r_if)
                se_if_reshaped = tf.reshape(se_if, [-1, 1])
                loss_if = tf.reduce_mean(self.lan_if(se_if_reshaped))

            # --- Neumann BC ---
            se_n = self.compute_neumann_squared_errors(self.x_n)

            # --- Dirichlet contact 1 ---
            u_b_c1_pred = self.main_net(self.x_b_c1)
            se_b_c1 = tf.square(u_b_c1_pred - self.u_b_c1)

            # --- Dirichlet contact 2 ---
            u_b_c2_pred = self.main_net(self.x_b_c2)
            se_b_c2 = tf.square(u_b_c2_pred - self.u_b_c2)

            # --- FEM data loss ---
            u_d_pred = self.main_net(x_d_sampled)
            se_d = tf.square(u_d_pred - u_d_sampled)

            # Reshape for LAN inputs
            se_n_reshaped    = tf.reshape(se_n,   [-1, 1])
            se_b_c1_reshaped = tf.reshape(se_b_c1, [-1, 1])
            se_b_c2_reshaped = tf.reshape(se_b_c2, [-1, 1])
            se_d_reshaped    = tf.reshape(se_d,   [-1, 1])

            # LAN-weighted losses
            loss_n    = tf.reduce_mean(self.lan_n(se_n_reshaped))
            loss_b_c1 = tf.reduce_mean(self.lan_b_c1(se_b_c1_reshaped))
            loss_b_c2 = tf.reduce_mean(self.lan_b_c2(se_b_c2_reshaped))
            loss_d    = tf.reduce_mean(self.lan_d(se_d_reshaped))

            
            total_loss = loss_r + 1e1*loss_b_c1 + 1e1*loss_b_c2 + loss_n + loss_if
            # total_loss = loss_d  # (alternative)

        return (total_loss, loss_r, loss_n, loss_b_c1, loss_b_c2, loss_d, loss_if,
                tape,
                se_r_reshaped, se_n_reshaped,
                se_b_c1_reshaped, se_b_c2_reshaped, se_d_reshaped, se_if_reshaped)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    @tf.function
    def train_step(self):
        """
        One adversarial training step:
          - Main network  →  minimise total_loss
          - Each LAN      →  maximise its respective component loss
        """
        (total_loss, loss_r, loss_n, loss_b_c1, loss_b_c2, loss_d, loss_if,
         tape,
         se_r, se_n, se_b_c1, se_b_c2, se_d, se_if) = self.compute_loss()

        # Gradients for main network (minimise)
        main_grads = tape.gradient(total_loss, self.main_net.trainable_variables)

        # Gradients for LANs (maximise → negate)
        def _neg_grads(loss, lan):
            grads = tape.gradient(loss, lan.trainable_variables)
            return [tf.zeros_like(v) if g is None else -g
                    for g, v in zip(grads, lan.trainable_variables)]

        lan_r_grads    = _neg_grads(loss_r,   self.lan_r)
        lan_n_grads    = _neg_grads(loss_n,   self.lan_n)
        lan_b_c1_grads = _neg_grads(loss_b_c1, self.lan_b_c1)
        lan_b_c2_grads = _neg_grads(loss_b_c2, self.lan_b_c2)
        lan_d_grads    = _neg_grads(loss_d,   self.lan_d)
        lan_if_grads   = _neg_grads(loss_if,  self.lan_if)

        # Apply gradients
        self.main_optimizer.apply_gradients(
            zip(main_grads, self.main_net.trainable_variables))
        self.lan_r_optimizer.apply_gradients(
            zip(lan_r_grads, self.lan_r.trainable_variables))
        self.lan_n_optimizer.apply_gradients(
            zip(lan_n_grads, self.lan_n.trainable_variables))
        self.lan_b_c1_optimizer.apply_gradients(
            zip(lan_b_c1_grads, self.lan_b_c1.trainable_variables))
        self.lan_b_c2_optimizer.apply_gradients(
            zip(lan_b_c2_grads, self.lan_b_c2.trainable_variables))
        self.lan_d_optimizer.apply_gradients(
            zip(lan_d_grads, self.lan_d.trainable_variables))
        self.lan_if_optimizer.apply_gradients(
            zip(lan_if_grads, self.lan_if.trainable_variables))

        del tape
        return total_loss, loss_r, loss_n, loss_b_c1, loss_b_c2, loss_d, loss_if

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, epochs, display_interval=100, accuracy_interval=500, evaluate_interval=0, save_dir=None):
        """
        Full training loop.

        Args:
            epochs:            Number of gradient steps
            display_interval:  Print loss every N epochs
            accuracy_interval: Compute prediction MSE every N epochs
                               (requires subclass to implement compute_prediction_accuracy)

        Returns:
            history dict with keys: total_loss, loss_r, loss_n, loss_b_c1,
                                    loss_b_c2, loss_d, raw_mse, accuracy_epochs
        """
        history = {
            'total_loss': [], 'loss_r': [], 'loss_n': [],
            'loss_b_c1': [], 'loss_b_c2': [], 'loss_d': [],
            'loss_if': [],   # interface-flux loss (via dedicated lan_if)
            'raw_mse': [], 'accuracy_epochs': []
        }

        for epoch in range(epochs):
            total_loss, loss_r, loss_n, loss_b_c1, loss_b_c2, loss_d, loss_if = self.train_step()

            history['total_loss'].append(total_loss.numpy())
            history['loss_r'].append(loss_r.numpy())
            history['loss_n'].append(loss_n.numpy())
            history['loss_b_c1'].append(loss_b_c1.numpy())
            history['loss_b_c2'].append(loss_b_c2.numpy())
            history['loss_d'].append(loss_d.numpy())
            history['loss_if'].append(loss_if.numpy())

            if epoch % display_interval == 0:
                print(f"Epoch {epoch}: Loss = {total_loss.numpy():.6f}, "
                      f"Loss_r = {loss_r.numpy():.6f}, "
                      f"Loss_n = {loss_n.numpy():.6f}, "
                      f"Loss_b_c1 = {loss_b_c1.numpy():.6f}, "
                      f"Loss_b_c2 = {loss_b_c2.numpy():.6f}, "
                      f"Loss_d = {loss_d.numpy():.6f}, "
                      f"Loss_if = {loss_if.numpy():.6f}")

            if epoch % accuracy_interval == 0 and hasattr(self, 'compute_prediction_accuracy'):
                mse = self.compute_prediction_accuracy()
                history['raw_mse'].append(mse)
                history['accuracy_epochs'].append(epoch)
                print(f"Epoch {epoch}: Raw Prediction MSE = {mse:.6f}")

            if evaluate_interval > 0 and epoch % evaluate_interval == 0 and epoch > 0:
                print(f"Epoch {epoch}: running evaluation...")
                epoch_dir = self.save(save_dir, epoch=epoch)
                evaluate.compare_with_fem(
                    self,
                    x_bounds=(-0.5, 0.5),
                    y_bounds=(-0.5, 0.5),
                    z_bounds=(-0.5, 0.5),
                    epoch=epoch,
                    save_dir=epoch_dir,   # <-- plots go into epoch subfolder
                )

        return history

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, x):
        """Run the main network on input x (numpy or tensor)."""
        if isinstance(x, np.ndarray):
            x = tf.convert_to_tensor(x, dtype=tf.float32)
        return self.main_net(x)

    def save(self, save_dir, epoch=None):
        # create epoch subfolder
        if epoch is not None:
            epoch_dir = os.path.join(save_dir, f'epoch_{epoch}')
        else:
            epoch_dir = save_dir
        os.makedirs(epoch_dir, exist_ok=True)

        shutil.copy('config.py', os.path.join(epoch_dir, 'config.py'))
        self.main_net.save_weights(os.path.join(epoch_dir, 'main_net.weights.h5'))
        self.lan_r.save_weights(   os.path.join(epoch_dir, 'lan_r.weights.h5'))
        self.lan_n.save_weights(   os.path.join(epoch_dir, 'lan_n.weights.h5'))
        self.lan_b_c1.save_weights(os.path.join(epoch_dir, 'lan_b_c1.weights.h5'))
        self.lan_b_c2.save_weights(os.path.join(epoch_dir, 'lan_b_c2.weights.h5'))
        self.lan_d.save_weights(   os.path.join(epoch_dir, 'lan_d.weights.h5'))
        self.lan_if.save_weights(  os.path.join(epoch_dir, 'lan_if.weights.h5'))
        if self.fourier_B is not None:
            np.save(os.path.join(epoch_dir, 'fourier_B.npy'), self.fourier_B)
        print(f"Epoch {epoch}: all weights saved to '{epoch_dir}'")
        return epoch_dir
    '''
    # in lapinn_base.py
    def save(self, save_dir, epoch=None):
        os.makedirs(save_dir, exist_ok=True)
        shutil.copy('config.py', os.path.join(save_dir, 'config.py'))

        suffix = f"_epoch{epoch}" if epoch is not None else ""
        self.main_net.save_weights(os.path.join(save_dir, f'main_net{suffix}.weights.h5'))
        self.lan_r.save_weights(   os.path.join(save_dir, f'lan_r{suffix}.weights.h5'))
        self.lan_n.save_weights(   os.path.join(save_dir, f'lan_n{suffix}.weights.h5'))
        self.lan_b_c1.save_weights(os.path.join(save_dir, f'lan_b_c1{suffix}.weights.h5'))
        self.lan_b_c2.save_weights(os.path.join(save_dir, f'lan_b_c2{suffix}.weights.h5'))
        self.lan_d.save_weights(   os.path.join(save_dir, f'lan_d{suffix}.weights.h5'))
        self.lan_if.save_weights(  os.path.join(save_dir, f'lan_if{suffix}.weights.h5'))
        if self.fourier_B is not None:
            np.save(os.path.join(save_dir, f'fourier_B{suffix}.npy'), self.fourier_B)
        print(f"Epoch {epoch}: all weights saved to '{save_dir}'")
    '''