import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------------
# Internal building blocks
# ---------------------------------------------------------------------------

def _fourier_embedding(inputs, fourier_features, fourier_sigma, input_dim):
    """
    Apply a fixed random Fourier feature embedding to a Keras tensor.

    Maps x ∈ R^input_dim  →  γ(x) = [cos(2πBx), sin(2πBx)] ∈ R^(2*fourier_features)
    where B ~ N(0, fourier_sigma²) is sampled once and kept constant.

    Args:
        inputs:           Keras tensor of shape (batch, input_dim)
        fourier_features: Number of random frequencies; output dim = 2*this
        fourier_sigma:    Std-dev of the Gaussian used to sample B
        input_dim:        Dimensionality of the raw input (e.g. 3 for x,y,z)

    Returns:
        (embedded, B) where embedded is the (batch, 2*fourier_features) Keras
        tensor and B is the (fourier_features, input_dim) numpy array, which
        must be saved alongside model weights for reproducible inference.
    """
    B = np.random.normal(0, fourier_sigma,
                         (fourier_features, input_dim)).astype(np.float32)
    B_const = tf.constant(B, dtype=tf.float32)

    projection = tf.keras.layers.Lambda(
        lambda x: 2.0 * np.pi * tf.matmul(x, B_const, transpose_b=True),
        name='fourier_projection'
    )(inputs)

    cos_feats = tf.keras.layers.Lambda(tf.cos, name='fourier_cos')(projection)
    sin_feats = tf.keras.layers.Lambda(tf.sin, name='fourier_sin')(projection)
    embedded  = tf.keras.layers.Concatenate(name='fourier_embedding')(
        [cos_feats, sin_feats]
    )  # shape: (batch, 2*fourier_features)

    return embedded, B


def _build_mlp_body(net_input, layers_main, initializer):
    """
    Plain tanh MLP body (no encoder combination).

    Args:
        net_input:   Keras tensor fed into the first hidden layer
        layers_main: Full layer list [input_dim, hidden..., output_dim];
                     hidden widths taken from [1:-1]
        initializer: Keras weight initializer

    Returns:
        Output Keras tensor (before the final Dense)
    """
    x = net_input
    for i, layer_size in enumerate(layers_main[1:-1]):
        x = tf.keras.layers.Dense(
            layer_size, activation='tanh',
            kernel_initializer=initializer,
            bias_initializer='zeros',
            name=f'hidden_{i}'
        )(x)
    return x


def _build_encoder_body(net_input, layers_main, initializer):
    """
    Dual-encoder MLP body.

    Two encoder networks (enc1, enc2) are computed once from net_input.
    Every hidden layer h is then combined as  h*enc1 + (1-h)*enc2,
    which improves gradient flow through deep networks.

    Args:
        net_input:   Keras tensor fed into encoders and first hidden layer
        layers_main: Full layer list [input_dim, hidden..., output_dim]
        initializer: Keras weight initializer

    Returns:
        Output Keras tensor (before the final Dense)
    """
    enc1 = tf.keras.layers.Dense(
        layers_main[1], activation='tanh',
        kernel_initializer=initializer,
        bias_initializer='zeros', name='encoder1'
    )(net_input)

    enc2 = tf.keras.layers.Dense(
        layers_main[1], activation='tanh',
        kernel_initializer=initializer,
        bias_initializer='zeros', name='encoder2'
    )(net_input)

    x = tf.keras.layers.Dense(
        layers_main[1], activation='tanh',
        kernel_initializer=initializer,
        bias_initializer='zeros', name='hidden_0'
    )(net_input)
    x = (tf.keras.layers.Multiply()([x, enc1]) +
         tf.keras.layers.Multiply()([1 - x, enc2]))

    for i, layer_size in enumerate(layers_main[2:-1], start=1):
        x = tf.keras.layers.Dense(
            layer_size, activation='tanh',
            kernel_initializer=initializer,
            bias_initializer='zeros', name=f'hidden_{i}'
        )(x)
        x = (tf.keras.layers.Multiply()([x, enc1]) +
             tf.keras.layers.Multiply()([1 - x, enc2]))

    return x


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_main_network(layers_main, architecture='encoder',
                       use_fourier=False, fourier_features=256,
                       fourier_sigma=1.0, input_dim=3):
    """
    Build the main network that approximates the PDE solution.

    Architecture and Fourier embedding are orthogonal choices:

        architecture='encoder', use_fourier=False  ->  encoder MLP on raw coords  (original)
        architecture='mlp',     use_fourier=False  ->  plain MLP on raw coords
        architecture='encoder', use_fourier=True   ->  encoder MLP on Fourier embedding
        architecture='mlp',     use_fourier=True   ->  plain MLP on Fourier embedding

    Args:
        layers_main:      List [input_dim, hidden..., output_dim].
                          layers_main[0] must equal input_dim (typically 3).
                          Hidden widths from layers_main[1:-1], output from layers_main[-1].
        architecture:     'encoder' (default) or 'mlp'.
        use_fourier:      If True, prepend the Fourier embedding gamma(x) to the chosen
                          architecture. The random matrix B is returned alongside the
                          model and must be saved for reproducible inference.
        fourier_features: Number of random frequencies (only when use_fourier=True).
                          Embedding output dim = 2 * fourier_features.
        fourier_sigma:    Std-dev of the Gaussian used to sample B (only when use_fourier=True).
                          Larger values -> higher-frequency features.
        input_dim:        Raw coordinate dimension (default 3 for x, y, z).

    Returns:
        model  -- tf.keras.Model with input shape (batch, input_dim)
        B      -- (fourier_features, input_dim) numpy array if use_fourier=True, else None.
                  Must be saved alongside model weights when use_fourier=True.

    Raises:
        ValueError: if architecture is not 'encoder' or 'mlp'.
    """
    if architecture not in ('encoder', 'mlp'):
        raise ValueError(
            f"architecture must be 'encoder' or 'mlp', got '{architecture}'"
        )

    initializer = tf.keras.initializers.GlorotNormal()
    B = None

    # ---- Input ----
    inputs = tf.keras.Input(shape=(input_dim,), name='coords')

    # ---- Optional Fourier embedding ----
    if use_fourier:
        net_input, B = _fourier_embedding(
            inputs, fourier_features, fourier_sigma, input_dim
        )
    else:
        net_input = inputs

    # ---- Architecture body ----
    if architecture == 'encoder':
        x = _build_encoder_body(net_input, layers_main, initializer)
        output_activation = tf.nn.softplus
    else:  # 'mlp'
        x = _build_mlp_body(net_input, layers_main, initializer)
        output_activation = None  # linear output

    # ---- Output layer ----
    outputs = tf.keras.layers.Dense(
        layers_main[-1],
        activation=output_activation,
        kernel_initializer=initializer,
        bias_initializer='zeros',
        name='output'
    )(x)

    arch_label = f'fourier_{architecture}' if use_fourier else architecture
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=arch_label)

    return model, B


def build_lan_network(layers_lan):
    """
    Build a Loss-Attentional Network (LAN) for weighting point errors.
    Uses linear activations with constant initialization as per LA-PINN paper.

    Args:
        layers_lan: List of integers [1, hidden..., 1]

    Returns:
        tf.keras.Model
    """
    inputs = tf.keras.Input(shape=(1,))
    x = inputs

    for layer_size in layers_lan[1:-1]:
        x = tf.keras.layers.Dense(
            layer_size, activation=None,
            kernel_initializer=tf.keras.initializers.Constant(1.0),
            bias_initializer='zeros'
        )(x)

    outputs = tf.keras.layers.Dense(
        1, activation=None,
        kernel_initializer=tf.keras.initializers.Constant(1.0),
        bias_initializer='zeros'
    )(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)