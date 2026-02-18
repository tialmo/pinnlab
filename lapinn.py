# lapinn.py
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf


class LaplaceLAPINN:
    def __init__(
        self,
        layers_main,
        layers_lan,
        pde_params=None,
        lr_main=1e-3,
        lr_lan=2e-4,
    ):
        self.layers_main = layers_main
        self.layers_lan = layers_lan
        self.pde_params = pde_params if pde_params is not None else {}

        self.main_net = self._build_main_network()

        # LANs (NO data LAN)
        self.lan_r = self._build_lan_network()
        self.lan_n = self._build_lan_network()
        self.lan_b_c1 = self._build_lan_network()
        self.lan_b_c2 = self._build_lan_network()

        # optimizers
        self.main_optimizer = tf.keras.optimizers.Adam(lr_main)
        self.lan_r_optimizer = tf.keras.optimizers.Adam(lr_lan)
        self.lan_n_optimizer = tf.keras.optimizers.Adam(lr_lan)
        self.lan_b_c1_optimizer = tf.keras.optimizers.Adam(lr_lan)
        self.lan_b_c2_optimizer = tf.keras.optimizers.Adam(lr_lan)

        # interface cache (built OUTSIDE tf.function)
        self.iface_i = tf.constant([], dtype=tf.int32)
        self.iface_j = tf.constant([], dtype=tf.int32)
        self.iface_n = tf.constant([], dtype=tf.float32)  # shape (0,) but we’ll reshape when used


    # ---------- networks ----------
    def _build_main_network(self):
        inputs = tf.keras.Input(shape=(self.layers_main[0],))
        initializer = tf.keras.initializers.GlorotNormal()

        enc1 = tf.keras.layers.Dense(
            self.layers_main[1], activation="tanh",
            kernel_initializer=initializer, bias_initializer="zeros",
            name="encoder1"
        )(inputs)

        enc2 = tf.keras.layers.Dense(
            self.layers_main[1], activation="tanh",
            kernel_initializer=initializer, bias_initializer="zeros",
            name="encoder2"
        )(inputs)

        x = tf.keras.layers.Dense(
            self.layers_main[1], activation="tanh",
            kernel_initializer=initializer, bias_initializer="zeros"
        )(inputs)

        x = tf.keras.layers.Multiply()([x, enc1]) + tf.keras.layers.Multiply()([1 - x, enc2])

        for layer_size in self.layers_main[2:-1]:
            x = tf.keras.layers.Dense(
                layer_size, activation="tanh",
                kernel_initializer=initializer, bias_initializer="zeros"
            )(x)
            x = tf.keras.layers.Multiply()([x, enc1]) + tf.keras.layers.Multiply()([1 - x, enc2])

        outputs = tf.keras.layers.Dense(
            self.layers_main[-1],
            activation=tf.nn.softplus,
            kernel_initializer=initializer,
            bias_initializer="zeros"
        )(x)

        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def _build_lan_network(self):
        inputs = tf.keras.Input(shape=(1,))
        x = inputs
        for layer_size in self.layers_lan[1:-1]:
            x = tf.keras.layers.Dense(
                layer_size, activation=None,
                kernel_initializer=tf.keras.initializers.Constant(1.0),
                bias_initializer="zeros"
            )(x)
        outputs = tf.keras.layers.Dense(
            1, activation=None,
            kernel_initializer=tf.keras.initializers.Constant(1.0),
            bias_initializer="zeros"
        )(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    # ---------- interface (NUMPY ONLY, call outside tf.function) ----------
    def prepare_interface_from_batch(self, batch, tol_sigma=1e-12, max_pairs=200000):
        """
        Call ONCE after your batch is ready (especially after overriding x_r/sigma_r from npz).
        Builds structured-grid interface neighbor pairs using your original method.
        """
        x_r = batch["x_r"]
        sigma_r = batch["sigma_r"]

        X = x_r.numpy()
        S = sigma_r.numpy().reshape(-1)

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
            for step, nvec in [
                ((1, 0, 0), (1.0, 0.0, 0.0)),
                ((0, 1, 0), (0.0, 1.0, 0.0)),
                ((0, 0, 1), (0.0, 0.0, 1.0)),
            ]:
                b = (a[0] + step[0], a[1] + step[1], a[2] + step[2])
                j = coord2idx.get(b, None)
                if j is None:
                    continue
                if abs(S[k] - S[j]) > tol_sigma:
                    iface_i.append(k)
                    iface_j.append(j)
                    iface_n.append(nvec)
                    if len(iface_i) >= max_pairs:
                        break
            if len(iface_i) >= max_pairs:
                break

        print(f"[lapinn] Built {len(iface_i)} grid interface neighbor pairs.")

        self.iface_i = tf.constant(np.asarray(iface_i, np.int32))
        self.iface_j = tf.constant(np.asarray(iface_j, np.int32))
        self.iface_n = tf.constant(np.asarray(iface_n, np.float32))

    # ---------- PDE pieces ----------
    def _compute_residual_laplace(self, x):
        x_coord = x[:, 0:1]
        y_coord = x[:, 1:2]
        z_coord = x[:, 2:3]

        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x_coord, y_coord, z_coord])
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch([x_coord, y_coord, z_coord])
                u = self.main_net(tf.concat([x_coord, y_coord, z_coord], axis=1))

            u_x = tape1.gradient(u, x_coord)
            u_y = tape1.gradient(u, y_coord)
            u_z = tape1.gradient(u, z_coord)

        u_xx = tape2.gradient(u_x, x_coord)
        u_yy = tape2.gradient(u_y, y_coord)
        u_zz = tape2.gradient(u_z, z_coord)

        return u_xx + u_yy + u_zz

    def _neumann_squared_errors(self, x_n, n_n, g_n):
        with tf.GradientTape() as tape:
            tape.watch(x_n)
            u = self.main_net(x_n)
        grad_u = tape.gradient(u, x_n)  # (N,3)
        dudn = tf.reduce_sum(grad_u * n_n, axis=1, keepdims=True)
        return tf.square(dudn - g_n)

    def _interface_flux_residual(self, x_r, sigma_r, n_if_sample=5000):
        n_pairs = tf.shape(self.iface_i)[0]
        
        def compute():
            n_if = tf.minimum(n_if_sample, n_pairs)
            idx = tf.random.shuffle(tf.range(n_pairs))[:n_if]
            
            ii = tf.gather(self.iface_i, idx)
            jj = tf.gather(self.iface_j, idx)
            nij = tf.reshape(tf.gather(self.iface_n, idx), (-1, 3))
            
            xi = tf.gather(x_r, ii)
            xj = tf.gather(x_r, jj)
            si = tf.gather(sigma_r, ii)
            sj = tf.gather(sigma_r, jj)
            
            pts = tf.concat([xi, xj], axis=0)
            with tf.GradientTape() as tape:
                tape.watch(pts)
                u = self.main_net(pts)
            gu = tape.gradient(u, pts)
            
            B = tf.shape(xi)[0]
            flux_i = si * tf.reduce_sum(gu[:B, :] * nij, axis=1, keepdims=True)
            flux_j = sj * tf.reduce_sum(gu[B:, :] * nij, axis=1, keepdims=True)
            
            return flux_i - flux_j  # No padding - return actual residuals
        
        def empty():
            return tf.zeros((0, 1), dtype=tf.float32)  # Empty tensor, not padded
        
        return tf.cond(n_pairs > 0, compute, empty)


    # ---------- loss on batch ----------
    def compute_loss_on_batch(self, batch, n_r_sample=10000, n_if_sample=5000):
        x_r = batch["x_r"]
        sigma_r = batch["sigma_r"]

        x_n = batch["x_n"]
        n_n = batch["n_n"]
        g_n = batch["g_n"]

        x_b_c1 = batch["x_b_c1"]
        u_b_c1 = batch["u_b_c1"]

        x_b_c2 = batch["x_b_c2"]
        u_b_c2 = batch["u_b_c2"]

        # sample collocation like your old code
        Nr = tf.shape(x_r)[0]
        take_r = tf.minimum(tf.constant(n_r_sample, tf.int32), Nr)
        ridx = tf.random.shuffle(tf.range(Nr))[:take_r]
        x_r_s = tf.gather(x_r, ridx)

        with tf.GradientTape(persistent=True) as tape:
            r_int = self._compute_residual_laplace(x_r_s)
            r_if = self._interface_flux_residual(x_r, sigma_r, n_if_sample=n_if_sample)

            r_all = tf.concat([r_int, r_if], axis=0)
            se_r = tf.square(r_all)
            loss_r = tf.reduce_mean(self.lan_r(tf.reshape(se_r, [-1, 1])))

            se_n = self._neumann_squared_errors(x_n, n_n, g_n)
            loss_n = tf.reduce_mean(self.lan_n(tf.reshape(se_n, [-1, 1])))

            se_c1 = tf.square(self.main_net(x_b_c1) - u_b_c1)
            se_c2 = tf.square(self.main_net(x_b_c2) - u_b_c2)
            loss_c1 = tf.reduce_mean(self.lan_b_c1(tf.reshape(se_c1, [-1, 1])))
            loss_c2 = tf.reduce_mean(self.lan_b_c2(tf.reshape(se_c2, [-1, 1])))

            total = loss_r + loss_n + loss_c1 + loss_c2

        return total, loss_r, loss_n, loss_c1, loss_c2, tape

    @tf.function
    def train_step_on_batch(self, batch):
        total, lr, ln, lc1, lc2, tape = self.compute_loss_on_batch(batch)

        g_main = tape.gradient(total, self.main_net.trainable_variables)
        g_r = tape.gradient(lr, self.lan_r.trainable_variables)
        g_n = tape.gradient(ln, self.lan_n.trainable_variables)
        g_c1 = tape.gradient(lc1, self.lan_b_c1.trainable_variables)
        g_c2 = tape.gradient(lc2, self.lan_b_c2.trainable_variables)

        def flip(grads, vars_):
            return [tf.zeros_like(v) if g is None else -g for g, v in zip(grads, vars_)]

        g_r = flip(g_r, self.lan_r.trainable_variables)
        g_n = flip(g_n, self.lan_n.trainable_variables)
        g_c1 = flip(g_c1, self.lan_b_c1.trainable_variables)
        g_c2 = flip(g_c2, self.lan_b_c2.trainable_variables)

        self.main_optimizer.apply_gradients(zip(g_main, self.main_net.trainable_variables))
        self.lan_r_optimizer.apply_gradients(zip(g_r, self.lan_r.trainable_variables))
        self.lan_n_optimizer.apply_gradients(zip(g_n, self.lan_n.trainable_variables))
        self.lan_b_c1_optimizer.apply_gradients(zip(g_c1, self.lan_b_c1.trainable_variables))
        self.lan_b_c2_optimizer.apply_gradients(zip(g_c2, self.lan_b_c2.trainable_variables))

        del tape
        return total, lr, ln, lc1, lc2

    def predict(self, x):
        if isinstance(x, np.ndarray):
            x = tf.convert_to_tensor(x, tf.float32)
        return self.main_net(x)
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf


class LaplaceLAPINN:
    def __init__(
        self,
        layers_main,
        layers_lan,
        pde_params=None,
        lr_main=1e-3,
        lr_lan=2e-4,
    ):
        self.layers_main = layers_main
        self.layers_lan = layers_lan
        self.pde_params = pde_params if pde_params is not None else {}

        self.main_net = self._build_main_network()

        # LANs — now with SEPARATE interface LAN
        self.lan_r = self._build_lan_network()      # Interior Laplace residual
        self.lan_if = self._build_lan_network()     # Interface flux continuity (NEW)
        self.lan_n = self._build_lan_network()      # Neumann BC
        self.lan_b_c1 = self._build_lan_network()   # Dirichlet contact 1
        self.lan_b_c2 = self._build_lan_network()   # Dirichlet contact 2

        # Optimizers — add one for interface LAN
        self.main_optimizer = tf.keras.optimizers.Adam(lr_main)
        self.lan_r_optimizer = tf.keras.optimizers.Adam(lr_lan)
        self.lan_if_optimizer = tf.keras.optimizers.Adam(lr_lan)  # NEW
        self.lan_n_optimizer = tf.keras.optimizers.Adam(lr_lan)
        self.lan_b_c1_optimizer = tf.keras.optimizers.Adam(lr_lan)
        self.lan_b_c2_optimizer = tf.keras.optimizers.Adam(lr_lan)

        # Interface cache (built OUTSIDE tf.function)
        self.iface_i = tf.constant([], dtype=tf.int32)
        self.iface_j = tf.constant([], dtype=tf.int32)
        self.iface_n = tf.constant([], dtype=tf.float32)

    # ---------- networks ----------
    def _build_main_network(self):
        inputs = tf.keras.Input(shape=(self.layers_main[0],))
        initializer = tf.keras.initializers.GlorotNormal()

        enc1 = tf.keras.layers.Dense(
            self.layers_main[1], activation="tanh",
            kernel_initializer=initializer, bias_initializer="zeros",
            name="encoder1"
        )(inputs)

        enc2 = tf.keras.layers.Dense(
            self.layers_main[1], activation="tanh",
            kernel_initializer=initializer, bias_initializer="zeros",
            name="encoder2"
        )(inputs)

        x = tf.keras.layers.Dense(
            self.layers_main[1], activation="tanh",
            kernel_initializer=initializer, bias_initializer="zeros"
        )(inputs)

        x = tf.keras.layers.Multiply()([x, enc1]) + tf.keras.layers.Multiply()([1 - x, enc2])

        for layer_size in self.layers_main[2:-1]:
            x = tf.keras.layers.Dense(
                layer_size, activation="tanh",
                kernel_initializer=initializer, bias_initializer="zeros"
            )(x)
            x = tf.keras.layers.Multiply()([x, enc1]) + tf.keras.layers.Multiply()([1 - x, enc2])

        outputs = tf.keras.layers.Dense(
            self.layers_main[-1],
            activation=tf.nn.softplus,
            kernel_initializer=initializer,
            bias_initializer="zeros"
        )(x)

        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def _build_lan_network(self):
        inputs = tf.keras.Input(shape=(1,))
        x = inputs
        for layer_size in self.layers_lan[1:-1]:
            x = tf.keras.layers.Dense(
                layer_size, activation=None,
                kernel_initializer=tf.keras.initializers.Constant(1.0),
                bias_initializer="zeros"
            )(x)
        outputs = tf.keras.layers.Dense(
            1, activation=None,
            kernel_initializer=tf.keras.initializers.Constant(1.0),
            bias_initializer="zeros"
        )(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    # ---------- interface (NUMPY ONLY, call outside tf.function) ----------
    def prepare_interface_from_batch(self, batch, tol_sigma=1e-12, max_pairs=200000):
        """
        Build structured-grid interface neighbor pairs.
        Call ONCE after batch is ready.
        """
        x_r = batch["x_r"]
        sigma_r = batch["sigma_r"]

        X = x_r.numpy()
        S = sigma_r.numpy().reshape(-1)

        xs = np.unique(X[:, 0]); ys = np.unique(X[:, 1]); zs = np.unique(X[:, 2])
        xs.sort(); ys.sort(); zs.sort()

        dx = np.min(np.diff(xs)) if len(xs) > 1 else 1.0
        dy = np.min(np.diff(ys)) if len(ys) > 1 else 1.0
        dz = np.min(np.diff(zs)) if len(zs) > 1 else 1.0

        x0, y0, z0 = xs[0], ys[0], zs[0]
        ix = np.rint((X[:, 0] - x0) / dx).astype(int)
        iy = np.rint((X[:, 1] - y0) / dy).astype(int)
        iz = np.rint((X[:, 2] - z0) / dz).astype(int)

        coord2idx = {(ix[k], iy[k], iz[k]): k for k in range(len(X))}

        iface_i, iface_j, iface_n = [], [], []

        for k in range(len(X)):
            a = (ix[k], iy[k], iz[k])
            for step, nvec in [
                ((1, 0, 0), (1.0, 0.0, 0.0)),
                ((0, 1, 0), (0.0, 1.0, 0.0)),
                ((0, 0, 1), (0.0, 0.0, 1.0)),
            ]:
                b = (a[0] + step[0], a[1] + step[1], a[2] + step[2])
                j = coord2idx.get(b, None)
                if j is None:
                    continue
                if abs(S[k] - S[j]) > tol_sigma:
                    iface_i.append(k)
                    iface_j.append(j)
                    iface_n.append(nvec)
                    if len(iface_i) >= max_pairs:
                        break
            if len(iface_i) >= max_pairs:
                break

        print(f"[lapinn] Built {len(iface_i)} interface neighbor pairs.")

        self.iface_i = tf.constant(np.asarray(iface_i, np.int32))
        self.iface_j = tf.constant(np.asarray(iface_j, np.int32))
        self.iface_n = tf.constant(np.asarray(iface_n, np.float32))

    # ---------- PDE pieces ----------
    def _compute_residual_laplace(self, x):
        """Compute ∇²u for interior Laplace equation."""
        x_coord = x[:, 0:1]
        y_coord = x[:, 1:2]
        z_coord = x[:, 2:3]

        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x_coord, y_coord, z_coord])
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch([x_coord, y_coord, z_coord])
                u = self.main_net(tf.concat([x_coord, y_coord, z_coord], axis=1))

            u_x = tape1.gradient(u, x_coord)
            u_y = tape1.gradient(u, y_coord)
            u_z = tape1.gradient(u, z_coord)

        u_xx = tape2.gradient(u_x, x_coord)
        u_yy = tape2.gradient(u_y, y_coord)
        u_zz = tape2.gradient(u_z, z_coord)

        return u_xx + u_yy + u_zz

    def _neumann_squared_errors(self, x_n, n_n, g_n):
        """Compute (∂u/∂n - g)² for Neumann BC."""
        with tf.GradientTape() as tape:
            tape.watch(x_n)
            u = self.main_net(x_n)
        grad_u = tape.gradient(u, x_n)
        dudn = tf.reduce_sum(grad_u * n_n, axis=1, keepdims=True)
        return tf.square(dudn - g_n)

    def _interface_flux_residual(self, x_r, sigma_r, n_if_sample=5000):
        """
        Compute interface flux discontinuity: σ_i·∇u(x_i)·n - σ_j·∇u(x_j)·n
        Returns actual residuals without padding.
        """
        n_pairs = tf.shape(self.iface_i)[0]

        def compute():
            n_if = tf.minimum(n_if_sample, n_pairs)
            idx = tf.random.shuffle(tf.range(n_pairs))[:n_if]

            ii = tf.gather(self.iface_i, idx)
            jj = tf.gather(self.iface_j, idx)
            nij = tf.reshape(tf.gather(self.iface_n, idx), (-1, 3))

            xi = tf.gather(x_r, ii)
            xj = tf.gather(x_r, jj)
            si = tf.gather(sigma_r, ii)
            sj = tf.gather(sigma_r, jj)

            pts = tf.concat([xi, xj], axis=0)
            with tf.GradientTape() as tape:
                tape.watch(pts)
                u = self.main_net(pts)
            gu = tape.gradient(u, pts)

            B = tf.shape(xi)[0]
            gu_i = gu[:B, :]
            gu_j = gu[B:, :]

            flux_i = si * tf.reduce_sum(gu_i * nij, axis=1, keepdims=True)
            flux_j = sj * tf.reduce_sum(gu_j * nij, axis=1, keepdims=True)

            return flux_i - flux_j  # No padding — return actual residuals only

        def empty():
            return tf.zeros((0, 1), dtype=tf.float32)

        return tf.cond(n_pairs > 0, compute, empty)

    # ---------- loss on batch ----------
    def compute_loss_on_batch(self, batch, n_r_sample=10000, n_if_sample=5000):
        """Compute all loss components separately."""
        x_r = batch["x_r"]
        sigma_r = batch["sigma_r"]

        x_n = batch["x_n"]
        n_n = batch["n_n"]
        g_n = batch["g_n"]

        x_b_c1 = batch["x_b_c1"]
        u_b_c1 = batch["u_b_c1"]

        x_b_c2 = batch["x_b_c2"]
        u_b_c2 = batch["u_b_c2"]

        # Sample collocation points for interior residual
        Nr = tf.shape(x_r)[0]
        take_r = tf.minimum(tf.constant(n_r_sample, tf.int32), Nr)
        ridx = tf.random.shuffle(tf.range(Nr))[:take_r]
        x_r_s = tf.gather(x_r, ridx)

        with tf.GradientTape(persistent=True) as tape:
            # 1. Interior Laplace residual: ∇²u = 0
            r_int = self._compute_residual_laplace(x_r_s)
            se_r = tf.square(r_int)
            loss_r = tf.reduce_mean(self.lan_r(tf.reshape(se_r, [-1, 1])))

            # 2. Interface flux continuity: σ_i·∇u·n = σ_j·∇u·n (SEPARATE)
            r_if = self._interface_flux_residual(x_r, sigma_r, n_if_sample=n_if_sample)
            # Handle empty interface case
            loss_if = tf.cond(
                tf.shape(r_if)[0] > 0,
                lambda: tf.reduce_mean(self.lan_if(tf.square(r_if))),
                lambda: tf.constant(0.0, dtype=tf.float32)
            )

            # 3. Neumann BC: ∂u/∂n = 0
            se_n = self._neumann_squared_errors(x_n, n_n, g_n)
            loss_n = tf.reduce_mean(self.lan_n(tf.reshape(se_n, [-1, 1])))

            # 4. Dirichlet BC contact 1: u = V1
            se_c1 = tf.square(self.main_net(x_b_c1) - u_b_c1)
            loss_c1 = tf.reduce_mean(self.lan_b_c1(tf.reshape(se_c1, [-1, 1])))

            # 5. Dirichlet BC contact 2: u = V2
            se_c2 = tf.square(self.main_net(x_b_c2) - u_b_c2)
            loss_c2 = tf.reduce_mean(self.lan_b_c2(tf.reshape(se_c2, [-1, 1])))

            # Total loss
            total = loss_r + loss_if + loss_n + loss_c1 + loss_c2

        return total, loss_r, loss_if, loss_n, loss_c1, loss_c2, tape

    @tf.function
    def train_step_on_batch(self, batch):
        """Single training step with adversarial LAN updates."""
        total, lr, lif, ln, lc1, lc2, tape = self.compute_loss_on_batch(batch)

        # Gradients for main network (minimize)
        g_main = tape.gradient(total, self.main_net.trainable_variables)

        # Gradients for LANs (maximize — flip sign)
        g_r = tape.gradient(lr, self.lan_r.trainable_variables)
        g_if = tape.gradient(lif, self.lan_if.trainable_variables)
        g_n = tape.gradient(ln, self.lan_n.trainable_variables)
        g_c1 = tape.gradient(lc1, self.lan_b_c1.trainable_variables)
        g_c2 = tape.gradient(lc2, self.lan_b_c2.trainable_variables)

        def flip(grads, vars_):
            return [tf.zeros_like(v) if g is None else -g for g, v in zip(grads, vars_)]

        g_r = flip(g_r, self.lan_r.trainable_variables)
        g_if = flip(g_if, self.lan_if.trainable_variables)
        g_n = flip(g_n, self.lan_n.trainable_variables)
        g_c1 = flip(g_c1, self.lan_b_c1.trainable_variables)
        g_c2 = flip(g_c2, self.lan_b_c2.trainable_variables)

        # Apply gradients
        self.main_optimizer.apply_gradients(zip(g_main, self.main_net.trainable_variables))
        self.lan_r_optimizer.apply_gradients(zip(g_r, self.lan_r.trainable_variables))
        self.lan_if_optimizer.apply_gradients(zip(g_if, self.lan_if.trainable_variables))
        self.lan_n_optimizer.apply_gradients(zip(g_n, self.lan_n.trainable_variables))
        self.lan_b_c1_optimizer.apply_gradients(zip(g_c1, self.lan_b_c1.trainable_variables))
        self.lan_b_c2_optimizer.apply_gradients(zip(g_c2, self.lan_b_c2.trainable_variables))

        del tape
        return total, lr, lif, ln, lc1, lc2

    def predict(self, x):
        if isinstance(x, np.ndarray):
            x = tf.convert_to_tensor(x, tf.float32)
        return self.main_net(x)
