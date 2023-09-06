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

def ForwardResidualGeneratorEncoder(input_shape=(256, 256, 1),
                    output_channels=1,
                    dim=64,
                    n_downsamplings=4,
                    n_blocks=4,
                    norm='layer_norm'):
    Norm = _get_norm_layer(norm)

    # 0
    h = inputs = keras.Input(shape=input_shape)

    # 1
    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = keras.layers.Conv2D(dim, 7, padding='valid', use_bias=True)(h)
    h = Norm()(h)
    h = tf.nn.relu(h)

    # 2
    for _ in range(n_downsamplings):
        dim *= 2
        h = keras.layers.Conv2D(dim, 3, strides=2, padding='same', use_bias=True)(h)
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
                                    n_downsamplings=4,
                                    n_blocks=4,
                                    norm='layer_norm'):
    Norm = _get_norm_layer(norm)

    # 0
    latent = keras.Input(shape=latent_input_shape)
    dim = latent_input_shape[-1]
    style_code = keras.Input(shape=code_input_shape)

    res1 = keras.Input(shape=res1_input_shape)
    res2 = keras.Input(shape=res2_input_shape)
    res3 = keras.Input(shape=res3_input_shape)


    h_style_code = keras.layers.Conv2DTranspose(dim, 7, strides=2, padding='same', use_bias=True)(style_code)

    h = latent + h_style_code


    for _ in range(n_blocks//4):
        h = _residual_block(h, Norm)

    h = h + res1

    for _ in range(n_blocks//2):
        h = _residual_block(h, Norm)

    h = h + res2

    for _ in range(n_downsamplings):
        dim //= 2
        h = keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=True)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

    h = h + res3

    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = keras.layers.Conv2D(output_channels, 7, padding='valid')(h)
    h = tf.tanh(h)

    return keras.Model(inputs=[latent, style_code, res1, res2, res3], outputs=h)

def BackwardResidualGeneratorEncoder(input_shape=(256, 256, 1),
                    output_channels=1,
                    dim=64,
                    n_downsamplings=4,
                    n_blocks=4,
                    norm='layer_norm'):
    Norm = _get_norm_layer(norm)

    # 0
    h = inputs = keras.Input(shape=input_shape)

    # 1
    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = keras.layers.Conv2D(dim, 7, padding='valid', use_bias=True)(h)
    h = Norm()(h)
    h = tf.nn.relu(h)

    # 2
    for _ in range(n_downsamplings):
        dim *= 2
        h = keras.layers.Conv2D(dim, 3, strides=2, padding='same', use_bias=True)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

    # 3
    for _ in range(n_blocks//2):
        h = _residual_block(h, Norm)

    for _ in range(n_blocks//2):
        h = _residual_block(h, Norm)

    # h_latent = _residual_block(h, Norm)
    # h_style_code =_residual_block(h, Norm)
    h_style_code = h
    h_style_code = keras.layers.Conv2D(4, 5, strides=2, padding='same', use_bias=True)(h_style_code)

    return keras.Model(inputs=inputs, outputs=h_style_code)

def BackwardResidualGeneratorDecoder(latent_input_shape=(8, 8, 8),
                                    res1_input_shape=(8, 8, 8),
                                    res2_input_shape=(8, 8, 8),
                                    res3_input_shape=(8, 8, 8),
                    output_channels=1,
                    dim=64,
                    n_downsamplings=4,
                    n_blocks=4,
                    norm='layer_norm'):
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
        h = keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=True)(h)
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

def ForwardResidualGeneratorEncoder2(input_shape=(256, 256, 1),
                    output_channels=1,
                    dim=64,
                    n_downsamplings=4,
                    n_blocks=4,
                    norm='layer_norm'):
    Norm = _get_norm_layer(norm)

    # 0
    h = inputs = keras.Input(shape=input_shape)

    # 1
    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = keras.layers.Conv2D(dim, 7, padding='valid', use_bias=True)(h)
    h = Norm()(h)
    h = tf.nn.relu(h)

    hconv1 = h

    # 2
    for _ in range(n_downsamplings):
        dim *= 2
        h = keras.layers.Conv2D(dim, 3, strides=2, padding='same', use_bias=True)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

    hconv2 = h
    # 3
    for _ in range(n_blocks//2):
        h = _residual_block(h, Norm)

    hres = h
    for _ in range(n_blocks//2):
        h = _residual_block(h, Norm)

    latent = h

    return keras.Model(inputs=inputs, outputs=[latent, hres, hconv2, hconv1])

