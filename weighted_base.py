import tensorflow as tf
from lapinn_base import LAPINN


class WeightedPINN(LAPINN):

    def __init__(self, *args, loss_weighting='classic', classic_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if loss_weighting not in ('classic', 'adaptive'):
            raise ValueError(f"loss_weighting must be 'classic' or 'adaptive', got '{loss_weighting}'")
        self.loss_weighting  = loss_weighting
        self.classic_weights = classic_weights or {
            'w_r': 1.0, 'w_n': 1.0, 'w_b_c1': 1.0,
            'w_b_c2': 1.0, 'w_if': 1.0, 'w_d': 1.0,
        }

    def compute_loss(self):
        r_indices   = tf.random.shuffle(tf.range(tf.shape(self.x_r)[0]))[:10000]
        x_r_sampled = tf.gather(self.x_r, r_indices)

        d_indices   = tf.random.shuffle(tf.range(tf.shape(self.x_d)[0]))[:10000]
        x_d_sampled = tf.gather(self.x_d, d_indices)
        u_d_sampled = tf.gather(self.u_d, d_indices)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.x_n)

            # Interior PDE residual
            r_int = self.compute_residual(x_r_sampled)
            mse_r = tf.reduce_mean(tf.square(r_int))

            # Interface flux residual
            mse_if = tf.constant(0.0, dtype=tf.float32)
            if hasattr(self, 'iface_i'):
                n_pairs = tf.shape(self.iface_i)[0]
                if n_pairs > 0:
                    n_if   = tf.minimum(tf.constant(5000, dtype=tf.int32), n_pairs)
                    if_idx = tf.random.shuffle(tf.range(n_pairs))[:n_if]
                    xi  = tf.gather(self.x_r,    tf.gather(self.iface_i, if_idx))
                    xj  = tf.gather(self.x_r,    tf.gather(self.iface_j, if_idx))
                    si  = tf.gather(self.sigma_r, tf.gather(self.iface_i, if_idx))
                    sj  = tf.gather(self.sigma_r, tf.gather(self.iface_j, if_idx))
                    nij = tf.gather(self.iface_n, if_idx)
                    mse_if = tf.reduce_mean(tf.square(
                        self.compute_interface_flux_residual(xi, xj, si, sj, nij)))

            # Neumann BC
            mse_n = tf.reduce_mean(self.compute_neumann_squared_errors(self.x_n))

            # Dirichlet contacts
            mse_b_c1 = tf.reduce_mean(tf.square(self.main_net(self.x_b_c1) - self.u_b_c1))
            mse_b_c2 = tf.reduce_mean(tf.square(self.main_net(self.x_b_c2) - self.u_b_c2))

            # FEM data — tracked only, not in total_loss
            mse_d = tf.reduce_mean(tf.square(self.main_net(x_d_sampled) - u_d_sampled))

            if self.loss_weighting == 'adaptive':
                denom = tf.maximum(
                    tf.reduce_max(tf.stack([mse_r, mse_n, mse_b_c1, mse_b_c2, mse_if])),
                    1e-12)
                loss_r    = (mse_r    / denom) * mse_r
                loss_n    = (mse_n    / denom) * mse_n
                loss_b_c1 = (mse_b_c1 / denom) * mse_b_c1
                loss_b_c2 = (mse_b_c2 / denom) * mse_b_c2
                loss_if   = (mse_if   / denom) * mse_if
                loss_d    = mse_d
            else:  # 'classic'
                w = self.classic_weights
                loss_r    = w['w_r']    * mse_r
                loss_n    = w['w_n']    * mse_n
                loss_b_c1 = w['w_b_c1'] * mse_b_c1
                loss_b_c2 = w['w_b_c2'] * mse_b_c2
                loss_if   = w['w_if']   * mse_if
                loss_d    = mse_d

            total_loss = loss_r + loss_n + loss_b_c1 + loss_b_c2 + loss_if

        return total_loss, loss_r, loss_n, loss_b_c1, loss_b_c2, loss_d, loss_if, tape

    @tf.function
    def train_step(self):
        total_loss, loss_r, loss_n, loss_b_c1, loss_b_c2, loss_d, loss_if, tape \
            = self.compute_loss()

        main_grads = tape.gradient(total_loss, self.main_net.trainable_variables)
        self.main_optimizer.apply_gradients(
            zip(main_grads, self.main_net.trainable_variables))

        del tape
        return total_loss, loss_r, loss_n, loss_b_c1, loss_b_c2, loss_d, loss_if