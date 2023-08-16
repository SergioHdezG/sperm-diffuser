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


def Splines2SingleSpermGenerator(input_shape=(10),
                    output_channels=1,
                    dim=16,
                    n_downsamplings=2,
                    n_blocks=9,
                    norm='instance_norm'):
    Norm = _get_norm_layer(norm)

    # 0
    h = inputs = keras.Input(shape=input_shape)

    h1 = keras.layers.Dense(128, activation='relu')(h)
    h = keras.layers.Dropout(0.2)(h1)
    h = keras.layers.Dense(32, activation='relu')(h)
    h = keras.layers.Dense(4, activation='relu')(h)
    h = keras.layers.Dense(32, activation='relu')(h)
    h = keras.layers.Dense(128, activation='relu')(h) + h1
    h = keras.layers.Dense(1225, activation='relu')(h)
    h = keras.layers.Dropout(0.2)(h)
    h = keras.layers.Reshape((35, 35, 1))(h)

    # 4
    for _ in range(n_downsamplings):
        h = keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

    # 5
    h = keras.layers.Conv2D(output_channels, 7, padding='same')(h)
    h = tf.tanh(h)

    return keras.Model(inputs=inputs, outputs=h)

def SingleSperm2SplineGenerator(input_shape=(140, 140, 1),
                    output_channels=10,
                    dim=64,
                    n_downsamplings=2,
                    n_blocks=9,
                    norm='instance_norm'):
    Norm = _get_norm_layer(norm)

    # 0
    h = inputs = keras.Input(shape=input_shape)

    # 1
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
    h = keras.layers.Flatten()(h)
    h = keras.layers.Dropout(0.2)(h)
    h = keras.layers.Dense(1225, activation='relu')(h)
    h1 = keras.layers.Dense(128, activation='relu')(h)
    h = keras.layers.Dense(32, activation='relu')(h)
    h = keras.layers.Dense(4, activation='relu')(h)
    h = keras.layers.Dense(32, activation='relu')(h)
    h = keras.layers.Dense(128, activation='relu')(h) + h1
    h = keras.layers.Dense(output_channels, activation='linear')(h)

    return keras.Model(inputs=inputs, outputs=h)

def LatentSplines2SingleSpermGeneratorEncoder(input_shape=(10)):

    # 0
    h = inputs = keras.Input(shape=input_shape)

    h = keras.layers.Dense(128, activation='relu')(h)
    h = keras.layers.Dropout(0.2)(h)
    h = keras.layers.Dense(32, activation='relu')(h)
    latent = keras.layers.Dense(4, activation='relu')(h)

    return keras.Model(inputs=inputs, outputs=latent)

def LatentSplines2SingleSpermGeneratorDecoder(input_shape=(4),
                    output_channels=1,
                    dim=16,
                    n_downsamplings=2,
                    n_blocks=9,
                    norm='instance_norm'):
    Norm = _get_norm_layer(norm)

    # 0
    latent = inputs = keras.Input(shape=input_shape)
    code = keras.Input(shape=32)

    h = keras.layers.Dense(32, activation='relu')(latent) + keras.layers.Dense(32, activation='relu')(code)
    h = keras.layers.Dense(128, activation='relu')(h)
    h = keras.layers.Dense(1225, activation='relu')(h)
    h = keras.layers.Dropout(0.2)(h)
    h = keras.layers.Reshape((35, 35, 1))(h)

    # 4
    for _ in range(n_downsamplings):
        h = keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

    # 5
    h = keras.layers.Conv2D(output_channels, 7, padding='same')(h)
    h = tf.tanh(h)

    return keras.Model(inputs=[inputs, code], outputs=h)

def LatentSingleSperm2SplineGeneratorEncoder(input_shape=(140, 140, 1),
                    output_channels=10,
                    dim=64,
                    n_downsamplings=2,
                    n_blocks=9,
                    norm='instance_norm'):
    Norm = _get_norm_layer(norm)

    # 0
    h = inputs = keras.Input(shape=input_shape)

    # 1
    h = keras.layers.Conv2D(dim, 7, padding='valid', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.relu(h)

    # 2
    for _ in range(n_downsamplings):
        dim *= 2
        h = keras.layers.Conv2D(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        conv_out = tf.nn.relu(h)

    # 3
    h = keras.layers.Flatten()(conv_out)
    h = keras.layers.Dropout(0.2)(h)
    h = keras.layers.Dense(1225, activation='relu')(h)
    h = keras.layers.Dense(128, activation='relu')(h)
    h = keras.layers.Dense(32, activation='relu')(h)
    latent = keras.layers.Dense(4, activation='relu')(h)

    code = keras.layers.Conv2D(int(dim/2), 3, strides=1, padding='same', use_bias=False)(conv_out)
    code = Norm()(code)
    conv_out = tf.nn.relu(code)
    conv_out = keras.layers.Flatten()(conv_out)
    conv_out = keras.layers.Dropout(0.2)(conv_out)
    code = keras.layers.Dense(64, activation='linear')(conv_out)
    return keras.Model(inputs=inputs, outputs=[latent, code])

def LatentSingleSperm2SplineGeneratorDecoder(input_shape=(4),
                    output_channels=10,
                    dim=64,
                    n_downsamplings=2,
                    n_blocks=9,
                    norm='instance_norm'):

    # 0
    latent = inputs = keras.Input(shape=input_shape)

    h = keras.layers.Dense(32, activation='relu')(latent)
    h = keras.layers.Dense(128, activation='relu')(h)
    h = keras.layers.Dense(output_channels, activation='linear')(h)
    return keras.Model(inputs=inputs, outputs=h)

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
                    dim=32,
                    n_downsamplings=4,
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

    h = keras.layers.Conv2D(16, 5, strides=2, padding='same', use_bias=False)(h)

    latent = h

    return keras.Model(inputs=inputs, outputs=latent)

def ForwardGeneratorDecoder(input_shape=(8, 8, 8),
                    output_channels=1,
                    dim=512,
                    n_downsamplings=4,
                    n_blocks=4,
                    norm='instance_norm'):
    Norm = _get_norm_layer(norm)

    # 0
    latent = keras.Input(shape=input_shape)
    style_code = keras.Input(shape=input_shape)

    h_latent = _residual_block(latent, Norm)
    h_style_code = _residual_block(style_code, Norm)

    h = h_latent + h_style_code

    h = keras.layers.Conv2DTranspose(dim, 5, strides=2, padding='same', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.relu(h)

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

    return keras.Model(inputs=[latent, style_code], outputs=h)

def BackwardGeneratorEncoder(input_shape=(256, 256, 1),
                    output_channels=1,
                    dim=64,
                    n_downsamplings=4,
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
        dim = tf.clip_by_value(dim * 2, 16, 512)
        h = keras.layers.Conv2D(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

    # 3
    for _ in range(n_blocks):
        h = _residual_block(h, Norm)

    h_latent = h
    h_style_code = h

    h_latent = _residual_block(h_latent, Norm)
    h_style_code =_residual_block(h_style_code, Norm)

    h_latent = keras.layers.Conv2D(16, 5, strides=2, padding='same', use_bias=False)(h_latent)
    h_style_code = keras.layers.Conv2D(16, 5, strides=2, padding='same', use_bias=False)(h_style_code)

    return keras.Model(inputs=inputs, outputs=[h_latent, h_style_code])

def BackwardGeneratorDecoder(input_shape=(8, 8, 64),
                    output_channels=1,
                    dim=64,
                    n_downsamplings=4,
                    n_blocks=4,
                    norm='instance_norm'):
    Norm = _get_norm_layer(norm)

    # 0
    latent = keras.Input(shape=input_shape)

    h = keras.layers.Conv2DTranspose(dim, 5, strides=2, padding='same', use_bias=False)(latent)


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


def ForwardResidualGeneratorEncoder(input_shape=(256, 256, 1),
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

    hconv = h
    # 3
    for _ in range(n_blocks//2):
        h = _residual_block(h, Norm)

    hres = h
    for _ in range(n_blocks//2):
        h = _residual_block(h, Norm)

    latent = h

    return keras.Model(inputs=inputs, outputs=[latent, hres, hconv, tf.repeat(inputs,64, axis=-1)])

def ForwardResidualGeneratorDecoder(latent_input_shape=(8, 8, 8),
                                    code_input_shape=(8, 8, 8),
                                    res1_input_shape=(8, 8, 8),
                                    res2_input_shape=(8, 8, 8),
                                    res3_input_shape=(8, 8, 8),
                                    output_channels=1,
                                    dim=256,
                                    n_downsamplings=2,
                                    n_blocks=4,
                                    norm='instance_norm'):
    Norm = _get_norm_layer(norm)

    # 0
    latent = keras.Input(shape=latent_input_shape)
    dim = latent_input_shape[-1]
    style_code = keras.Input(shape=code_input_shape)

    res1 = keras.Input(shape=res1_input_shape)
    res2 = keras.Input(shape=res2_input_shape)
    res3 = keras.Input(shape=res3_input_shape)


    h_style_code = keras.layers.Conv2DTranspose(dim, 7, strides=2, padding='same', use_bias=False)(style_code)

    h = latent + h_style_code


    for _ in range(n_blocks//4):
        h = _residual_block(h, Norm)

    h = h + res1

    for _ in range(n_blocks//2):
        h = _residual_block(h, Norm)

    h = h + res2

    for _ in range(n_downsamplings):
        dim //= 2
        h = keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

    h = h + res3

    # 5
    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = keras.layers.Conv2D(output_channels, 7, padding='valid')(h)
    h = tf.tanh(h)

    return keras.Model(inputs=[latent, style_code, res1, res2, res3], outputs=h)

def BackwardResidualGeneratorEncoder(input_shape=(256, 256, 1),
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

    hinput = h
    # 2
    for _ in range(n_downsamplings):
        dim *= 2
        h = keras.layers.Conv2D(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

    hconv = h

    # 3
    for _ in range(n_blocks//2):
        h = _residual_block(h, Norm)

    hres = h

    for _ in range(n_blocks//2):
        h = _residual_block(h, Norm)

    h_latent = _residual_block(h, Norm)
    h_style_code =_residual_block(h, Norm)

    h_style_code = keras.layers.Conv2D(4, 5, strides=2, padding='same', use_bias=False)(h_style_code)

    return keras.Model(inputs=inputs, outputs=[h_latent, h_style_code, hres, hconv,  hinput])

def BackwardResidualGeneratorDecoder(latent_input_shape=(8, 8, 8),
                                    res1_input_shape=(8, 8, 8),
                                    res2_input_shape=(8, 8, 8),
                                    res3_input_shape=(8, 8, 8),
                    output_channels=1,
                    dim=64,
                    n_downsamplings=2,
                    n_blocks=4,
                    norm='instance_norm'):
    Norm = _get_norm_layer(norm)
    dim = latent_input_shape[-1]

    # 0
    h = latent = keras.Input(shape=latent_input_shape)

    res1 = keras.Input(shape=res1_input_shape)
    res2 = keras.Input(shape=res2_input_shape)
    res3 = keras.Input(shape=res3_input_shape)


    for _ in range(n_blocks//2):
        h = _residual_block(h, Norm)

    h = h + res1

    for _ in range(n_blocks//2):
        h = _residual_block(h, Norm)

    h = h + res2
    # 4
    for _ in range(n_downsamplings):
        dim //= 2
        h = keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

    h = h + res3

    # 5
    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = keras.layers.Conv2D(output_channels, 7, padding='valid')(h)
    h = tf.tanh(h)

    return keras.Model(inputs=[latent, res1, res2, res3], outputs=h)
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
