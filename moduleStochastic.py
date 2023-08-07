import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras


# ==============================================================================
# =                                  networks                                  =
# ==============================================================================

def _get_norm_layer(norm):
    if norm == 'none':
        return lambda: lambda x: x
    elif norm == 'batch_norm':
        return keras.layers.BatchNormalization
    elif norm == 'instance_norm':
        return tfa.layers.InstanceNormalization
    elif norm == 'layer_norm':
        return keras.layers.LayerNormalization

def _residual_block(x, norm):
    dim = x.shape[-1]

    h = x

    h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    h = keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h)
    h = norm()(h)
    h = tf.nn.relu(h)

    h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    h = keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h)
    h = norm()(h)

    return keras.layers.add([x, h])

def ForwardGeneratorEncoder(input_shape=(256, 256, 1),
                    output_channels=1,
                    dim=64,
                    n_downsamplings=2,
                    n_blocks=4,
                    norm='instance_norm'):
    Norm = _get_norm_layer(norm)

    # 0
    h = inputs = keras.Input(shape=input_shape)

    # 1
    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = keras.layers.Conv2D(dim, 7, padding='valid', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.relu(h)

    # 2
    for _ in range(n_downsamplings):
        dim *= 2
        h = keras.layers.Conv2D(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

    # 3
    for _ in range(n_blocks):
        h = _residual_block(h, Norm)

    h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    h = keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h)
    latent = h

    return keras.Model(inputs=inputs, outputs=latent)

def ForwardGeneratorDecoder(input_shape=(8, 8, 128),
                    output_channels=1,
                    dim=64,
                    n_downsamplings=2,
                    n_blocks=4,
                    norm='instance_norm'):
    Norm = _get_norm_layer(norm)

    # 0
    latent = keras.Input(shape=input_shape)
    style_code = keras.Input(shape=input_shape)

    # h_latent = keras.layers.Conv2DTranspose(input_shape[-1], 3, strides=1, padding='same', use_bias=False)(latent)
    # h_style_code = keras.layers.Conv2DTranspose(input_shape[-1], 3, strides=1, padding='same', use_bias=False)(style_code)

    h_latent = _residual_block(latent, Norm)
    h_style_code = _residual_block(style_code, Norm)

    h = h_latent + h_style_code
    for _ in range(n_blocks):
        h = _residual_block(h, Norm)

    # 4
    for _ in range(n_downsamplings):
        dim //= 2
        h = keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

    # 5
    h = keras.layers.Conv2D(output_channels, 7, padding='same')(h)
    h = tf.tanh(h)

    return keras.Model(inputs=[latent, style_code], outputs=h)

def BackwardGeneratorEncoder(input_shape=(256, 256, 1),
                    output_channels=3,
                    dim=64,
                    n_downsamplings=2,
                    n_blocks=4,
                    norm='instance_norm'):
    Norm = _get_norm_layer(norm)

    # 0
    h = inputs = keras.Input(shape=input_shape)

    # 1
    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = keras.layers.Conv2D(dim, 7, padding='valid', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.relu(h)

    # 2
    for _ in range(n_downsamplings):
        dim *= 2
        h = keras.layers.Conv2D(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

    # 3
    for _ in range(n_blocks):
        h = _residual_block(h, Norm)

    h_latent = _residual_block(h, Norm)
    h_latent = tf.pad(h_latent, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    h_latent = keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h_latent)
    h_style_code = _residual_block(h, Norm)
    h_style_code = tf.pad(h_style_code, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    h_style_code = keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h_style_code)

    return keras.Model(inputs=inputs, outputs=[h_latent, h_style_code])

def BackwardGeneratorDecoder(input_shape=(8, 8, 64),
                    output_channels=1,
                    dim=64,
                    n_downsamplings=2,
                    n_blocks=4,
                    norm='instance_norm'):
    Norm = _get_norm_layer(norm)

    # 0
    latent = h = keras.Input(shape=input_shape)

    for _ in range(n_blocks):
        h = _residual_block(h, Norm)

    # 4
    for _ in range(n_downsamplings):
        dim //= 2
        h = keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

    # 5
    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = keras.layers.Conv2D(output_channels, 7, padding='valid')(h)
    h = tf.tanh(h)

    return keras.Model(inputs=latent, outputs=h)

# ==============================================================================
# =                          learning rate scheduler                           =
# ==============================================================================

class LinearDecay(keras.optimizers.schedules.LearningRateSchedule):
    # if `step` < `step_decay`: use fixed learning rate
    # else: linearly decay the learning rate to zero

    def __init__(self, initial_learning_rate, total_steps, step_decay):
        super(LinearDecay, self).__init__()
        self._initial_learning_rate = initial_learning_rate
        self._steps = total_steps
        self._step_decay = step_decay
        self.current_learning_rate = tf.Variable(initial_value=initial_learning_rate, trainable=False, dtype=tf.float32)

    def __call__(self, step):
        self.current_learning_rate.assign(tf.cond(
            step >= self._step_decay,
            true_fn=lambda: self._initial_learning_rate * (1 - 1 / (self._steps - self._step_decay) * (step - self._step_decay)),
            false_fn=lambda: self._initial_learning_rate
        ))
        return self.current_learning_rate
