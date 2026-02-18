# models.py
import tensorflow as tf


def build_mlp(in_dim=3, hidden=(64, 64, 64, 64), out_dim=1, act="tanh"):
    inputs = tf.keras.Input(shape=(in_dim,))
    x = inputs
    for w in hidden:
        x = tf.keras.layers.Dense(w, activation=act, kernel_initializer="glorot_normal")(x)
    outputs = tf.keras.layers.Dense(out_dim, kernel_initializer="glorot_normal")(x)
    return tf.keras.Model(inputs, outputs)


class BasicLaplacePINN(tf.keras.Model):
    """
    Minimal PINN for Laplace / weighted Laplace with Neumann + Dirichlet.
    - PDE residual: div( sigma * grad(u) ) = 0 if sigma provided else Laplace(u)=0
    """
    def __init__(self, net: tf.keras.Model):
        super().__init__()
        self.net = net

    def call(self, x):
        return self.net(x)

    def pde_residual(self, x_r, sigma_r=None):
        """
        r(x) = div( sigma * grad(u) )
        If sigma_r is None -> Laplace(u)
        Robust implementation via batch_jacobian.
        """
        x_r = tf.convert_to_tensor(x_r, tf.float32)
    
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(x_r)
            with tf.GradientTape() as tape1:
                tape1.watch(x_r)
                u = self.net(x_r)  # (N,1)
            grad_u = tape1.gradient(u, x_r)  # (N,3)
    
            if sigma_r is None:
                flux = grad_u
            else:
                sigma_r = tf.convert_to_tensor(sigma_r, tf.float32)
                sigma_r = tf.reshape(sigma_r, (-1, 1))  # (N,1)
                flux = sigma_r * grad_u  # (N,3)
    
        # J = d(flux)/d(x): (N,3,3)
        J = tape2.batch_jacobian(flux, x_r)
        del tape2
    
        div = tf.linalg.trace(J)           # (N,)
        return div[:, None]                # (N,1)


    def neumann_residual(self, x_n, n_n, g_n):
        """
        Neumann: grad(u)·n = g
        """
        x_n = tf.convert_to_tensor(x_n, tf.float32)
        n_n = tf.convert_to_tensor(n_n, tf.float32)
        g_n = tf.convert_to_tensor(g_n, tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(x_n)
            u = self.net(x_n)
        grad_u = tape.gradient(u, x_n)  # (N,3)

        dudn = tf.reduce_sum(grad_u * n_n, axis=1, keepdims=True)  # (N,1)
        return dudn - g_n

    #@tf.function
    def train_step_on_batch(self, batch, optimizer,
                            w_r=1.0, w_n=1.0, w_c1=1.0, w_c2=1.0,
                            use_sigma=True):
        x_r = batch["x_r"]
        sigma_r = batch.get("sigma_r", None)
        x_n = batch["x_n"]
        n_n = batch["n_n"]
        g_n = batch["g_n"]
        x_b_c1 = batch["x_b_c1"]
        u_b_c1 = batch["u_b_c1"]
        x_b_c2 = batch["x_b_c2"]
        u_b_c2 = batch["u_b_c2"]

        with tf.GradientTape() as tape:
            # PDE residual
            r = self.pde_residual(x_r, sigma_r if (use_sigma and sigma_r is not None) else None)
            loss_r = tf.reduce_mean(tf.square(r))

            # Neumann
            rn = self.neumann_residual(x_n, n_n, g_n)
            loss_n = tf.reduce_mean(tf.square(rn))

            # Dirichlet
            u1 = self.net(x_b_c1)
            u2 = self.net(x_b_c2)
            loss_c1 = tf.reduce_mean(tf.square(u1 - u_b_c1))
            loss_c2 = tf.reduce_mean(tf.square(u2 - u_b_c2))

            total = w_r * loss_r + w_n * loss_n + w_c1 * loss_c1 + w_c2 * loss_c2

        grads = tape.gradient(total, self.net.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.net.trainable_variables))

        return total, loss_r, loss_n, loss_c1, loss_c2
