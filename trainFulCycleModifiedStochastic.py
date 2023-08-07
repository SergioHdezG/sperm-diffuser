import functools

import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tensorflow.keras as keras
import tf2lib as tl
import tf2gan as gan
import tqdm

import data
import module
import moduleStochastic

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--output_dir', default='BezierImage2SpermImageConditionedStochastic')
py.arg('--dataset', default='FulBezierSplines2FulSperm')
py.arg('--datasets_dir', default='datasets')
py.arg('--load_size', type=int, default=286)  # load image to this size
py.arg('--crop_size', type=int, default=256)  # then crop to this size
py.arg('--batch_size', type=int, default=2)
py.arg('--epochs', type=int, default=200)
py.arg('--epoch_decay', type=int, default=100)  # epoch to start decaying learning rate
py.arg('--lr', type=float, default=0.0005)
py.arg('--beta_1', type=float, default=0.5)
py.arg('--adversarial_loss_mode', default='lsgan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
py.arg('--gradient_penalty_mode', default='none', choices=['none', 'dragan', 'wgan-gp'])
py.arg('--gradient_penalty_weight', type=float, default=10.0)
py.arg('--cycle_loss_weight', type=float, default=10.0)
py.arg('--identity_loss_weight', type=float, default=0.0)
py.arg('--pool_size', type=int, default=50)  # pool size to store fake samples
py.arg('--trainA', type=str, default='trainFulBezierImage')  # pool size to store fake samples
py.arg('--trainB', type=str, default='trainFulSpermImage')  # pool size to store fake samples
py.arg('--testA', type=str, default='testFulBezierImage')  # pool size to store fake samples
py.arg('--testB', type=str, default='testFulSpermImage')  # pool size to store fake samples
py.arg('--fileExtension', type=str, default='png')  # pool size to store fake samples
args = py.args()

channels = 3
# output_dir
output_dir = py.join('output', args.output_dir)
py.mkdir(output_dir)

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)


# ==============================================================================
# =                                    data                                    =
# ==============================================================================

A_img_paths = py.glob(py.join(args.datasets_dir, args.dataset, args.trainA), '*.'+args.fileExtension)
B_img_paths = py.glob(py.join(args.datasets_dir, args.dataset, args.trainB), '*.'+args.fileExtension)
A_B_dataset, len_dataset = data.make_zip_dataset(A_img_paths, B_img_paths, args.batch_size, args.load_size, args.crop_size, training=True, repeat=False, channels=channels)

A2B_pool = data.ItemPool(args.pool_size)
B2A_pool = data.ItemPool(args.pool_size)

A_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, args.testA), '*.'+args.fileExtension)
B_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, args.testB), '*.'+args.fileExtension)
A_B_dataset_test, _ = data.make_zip_dataset(A_img_paths_test, B_img_paths_test, args.batch_size, args.load_size, args.crop_size, training=False, repeat=True, channels=channels)


# ==============================================================================
# =                                   models                                   =
# ==============================================================================

G_A2B_Enc = moduleStochastic.ForwardGeneratorEncoder(input_shape=(args.crop_size, args.crop_size, channels), output_channels=channels)
G_A2B_Dec = moduleStochastic.ForwardGeneratorDecoder(input_shape=(*G_A2B_Enc.output_shape[1:3], G_A2B_Enc.output_shape[3]//2), output_channels=channels)

G_B2A_Enc = moduleStochastic.BackwardGeneratorEncoder(input_shape=(args.crop_size, args.crop_size, channels), output_channels=channels)
G_B2A_Dec = moduleStochastic.BackwardGeneratorDecoder(input_shape=(*G_A2B_Enc.output_shape[1:3], G_A2B_Enc.output_shape[3]//2), output_channels=channels)

D_A = module.ConvDiscriminator(input_shape=(args.crop_size, args.crop_size, channels))
D_B = module.ConvDiscriminator(input_shape=(args.crop_size, args.crop_size, channels))

d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn(args.adversarial_loss_mode)
cycle_loss_fn = tf.losses.MeanAbsoluteError()
identity_loss_fn = tf.losses.MeanAbsoluteError()

G_lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
D_lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
G_optimizer = keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=args.beta_1)
D_optimizer = keras.optimizers.Adam(learning_rate=D_lr_scheduler, beta_1=args.beta_1)


# ==============================================================================
# =                                 train step                                 =
# ==============================================================================

def G_A2B(A, codes, training=True):
    A_latent = G_A2B_Enc(A, training=training)
    A_latent_u, A_latent_logvar = tf.split(A_latent, 2, axis=-1)
    latent = reparametrize(A_latent_u, A_latent_logvar)
    A2B = G_A2B_Dec([latent, codes], training=training)
    return A2B, latent

def G_B2A(B, training=True):
    B_latent, B_code = G_B2A_Enc(B, training=training)
    B_latent_u, B_latent_logvar = tf.split(B_latent, 2, axis=-1)
    B_code_u, B_code_logvar = tf.split(B_code, 2, axis=-1)
    latent = reparametrize(B_latent_u, B_latent_logvar)
    B2A = G_B2A_Dec(latent, training=training)
    return B2A, latent, B_code_u, B_code_logvar

def G_A2A(A, training=True):
    A_latent = G_A2B_Enc(A, training=training)
    A_latent_u, A_latent_logvar = tf.split(A_latent, 2, axis=-1)
    latent = reparametrize(A_latent_u, A_latent_logvar)
    A2A = G_B2A_Dec(latent, training=training)
    return A2A

def G_B2B(B, training=True):
    B_latent, B_code = G_B2A_Enc(B, training=training)
    B_latent_u, B_latent_logvar = tf.split(B_latent, 2, axis=-1)
    B_code_u, B_code_logvar = tf.split(B_code, 2, axis=-1)
    latent = reparametrize(B_latent_u, B_latent_logvar)
    codes = reparametrize(B_code_u, B_code_logvar)
    B2B = G_A2B_Dec([latent, codes], training=training)
    return B2B

@tf.function
def reparametrize(u, logvar):
    eps = tf.random.normal(shape=u.shape)
    return eps * tf.exp(logvar * .5) + u

@tf.function
def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),axis=raxis)


# @tf.function
def train_G(A, B):
    with tf.GradientTape() as t:
        # Standard Cycle-GAN
        B2A, B_latent, B_code_u, B_code_logvar = G_B2A(B, training=True)
        rand_code = tf.random.normal(shape=B_code_u.shape)
        A2B, A_latent = G_A2B(A, rand_code, training=True)

        A2B2A, A2B2A_latent, A2B2A_code_u, A2B2A_code_logvar = G_B2A(A2B, training=True)
        B_code = reparametrize(B_code_u, B_code_logvar)
        A2B2A_code = reparametrize(A2B2A_code_u, A2B2A_code_logvar)

        B2A2B, B2A2B_latent = G_A2B(B2A, B_code, training=True)

        # Modification
        # Reparametrized identity calculation
        A2A = G_A2A(A, training=True)
        B2B = G_B2B(B, training=True)

        # Standard Cycle-GAN
        A2B_d_logits = D_B(A2B, training=True)
        B2A_d_logits = D_A(B2A, training=True)

        A2B_g_loss = g_loss_fn(A2B_d_logits)
        B2A_g_loss = g_loss_fn(B2A_d_logits)
        A2B2A_cycle_loss = cycle_loss_fn(A, A2B2A)
        B2A2B_cycle_loss = cycle_loss_fn(B, B2A2B)
        A2A_id_loss = identity_loss_fn(A, A2A)
        B2B_id_loss = identity_loss_fn(B, B2B)

        G_loss = (A2B_g_loss + B2A_g_loss) + (A2B2A_cycle_loss + B2A2B_cycle_loss) * args.cycle_loss_weight + (
                    A2A_id_loss + B2B_id_loss) * args.identity_loss_weight

        # Modification
        # Identity in splines latent spaces
        spline_latent_loss = tf.reduce_mean(
            tf.losses.mse(A_latent, A2B2A_latent) + tf.losses.mse(B_latent, B2A2B_latent))

        # Maintain code_u and code_logvar as a normal distribution N(0, 1)
        logpz = tf.reduce_mean(log_normal_pdf(B_code, 0., 0.)) + \
                tf.reduce_mean(log_normal_pdf(A2B2A_code, 0., 0.)) + \
                tf.reduce_mean(log_normal_pdf(A_latent, 0., 0.)) + \
                tf.reduce_mean(log_normal_pdf(B_latent, 0., 0.)) # mean = 0 and logvar = 0 equivalent to var = 1

        G_loss += spline_latent_loss - logpz

    G_grad = t.gradient(G_loss, G_A2B_Enc.trainable_variables + G_A2B_Dec.trainable_variables + G_B2A_Enc.trainable_variables + G_B2A_Dec.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, G_A2B_Enc.trainable_variables + G_A2B_Dec.trainable_variables + G_B2A_Enc.trainable_variables + G_B2A_Dec.trainable_variables))

    return A2B, B2A, {'A2B_g_loss': A2B_g_loss,
                      'B2A_g_loss': B2A_g_loss,
                      'A2B2A_cycle_loss': A2B2A_cycle_loss,
                      'B2A2B_cycle_loss': B2A2B_cycle_loss,
                      'A2A_id_loss': A2A_id_loss,
                      'B2B_id_loss': B2B_id_loss,
                      'spline_latent_loss': spline_latent_loss} #,
                      # 'logpz': tf.reduce_sum(logpz)}


@tf.function
def train_D(A, B, A2B, B2A):
    with tf.GradientTape() as t:
        A_d_logits = D_A(A, training=True)
        B2A_d_logits = D_A(B2A, training=True)
        B_d_logits = D_B(B, training=True)
        A2B_d_logits = D_B(A2B, training=True)

        A_d_loss, B2A_d_loss = d_loss_fn(A_d_logits, B2A_d_logits)
        B_d_loss, A2B_d_loss = d_loss_fn(B_d_logits, A2B_d_logits)
        D_A_gp = gan.gradient_penalty(functools.partial(D_A, training=True), A, B2A, mode=args.gradient_penalty_mode)
        D_B_gp = gan.gradient_penalty(functools.partial(D_B, training=True), B, A2B, mode=args.gradient_penalty_mode)

        D_loss = (A_d_loss + B2A_d_loss) + (B_d_loss + A2B_d_loss) + (D_A_gp + D_B_gp) * args.gradient_penalty_weight

    D_grad = t.gradient(D_loss, D_A.trainable_variables + D_B.trainable_variables)
    D_optimizer.apply_gradients(zip(D_grad, D_A.trainable_variables + D_B.trainable_variables))

    return {'A_d_loss': A_d_loss + B2A_d_loss,
            'B_d_loss': B_d_loss + A2B_d_loss,
            'D_A_gp': D_A_gp,
            'D_B_gp': D_B_gp}


def train_step(A, B):
    A2B, B2A, G_loss_dict = train_G(A, B)

    # cannot autograph `A2B_pool`
    A2B = A2B_pool(A2B)  # or A2B = A2B_pool(A2B.numpy()), but it is much slower
    B2A = B2A_pool(B2A)  # because of the communication between CPU and GPU

    D_loss_dict = train_D(A, B, A2B, B2A)

    return G_loss_dict, D_loss_dict


@tf.function
def sample(A, B):
    B2A, B_latent, B_code_u, B_code_logvar = G_B2A(B, training=False)
    eps = tf.random.normal(shape=B_code_u.shape)
    A2B, A_latent = G_A2B(A, eps, training=False)
    A2B2A, _, _, _ = G_B2A(A2B, training=False)
    B2A2B, _ = G_A2B(B2A, reparametrize(B_code_u, B_code_logvar), training=False)
    A2A = G_A2A(A, training=False)
    B2B = G_B2B(B, training=False)
    return A2B, B2A, A2B2A, B2A2B, A2A, B2B


# ==============================================================================
# =                                    run                                     =
# ==============================================================================

# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# checkpoint
checkpoint = tl.Checkpoint(dict(G_A2B_Enc=G_A2B_Enc,
                                G_A2B_Dec=G_A2B_Dec,
                                G_B2A_Enc=G_B2A_Enc,
                                G_B2A_Dec=G_B2A_Dec,
                                D_A=D_A,
                                D_B=D_B,
                                G_optimizer=G_optimizer,
                                D_optimizer=D_optimizer,
                                ep_cnt=ep_cnt),
                           py.join(output_dir, 'checkpoints'),
                           max_to_keep=5)
try:  # restore checkpoint including the epoch counter
    checkpoint.restore().assert_existing_objects_matched()
except Exception as e:
    print(e)

# summary
train_summary_writer = tf.summary.create_file_writer(py.join(output_dir, 'summaries', 'train'))

# sample
test_iter = iter(A_B_dataset_test)
sample_dir = py.join(output_dir, 'samples_training')
py.mkdir(sample_dir)

# main loop
with train_summary_writer.as_default():
    for ep in tqdm.trange(args.epochs, desc='Epoch Loop'):
        if ep < ep_cnt:
            continue

        # update epoch counter
        ep_cnt.assign_add(1)

        # train for an epoch
        for A, B in tqdm.tqdm(A_B_dataset, desc='Inner Epoch Loop', total=len_dataset):
            G_loss_dict, D_loss_dict = train_step(A, B)

            # # summary
            tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')
            tl.summary(D_loss_dict, step=G_optimizer.iterations, name='D_losses')
            tl.summary({'learning rate': G_lr_scheduler.current_learning_rate}, step=G_optimizer.iterations, name='learning rate')

            # sample
            if G_optimizer.iterations.numpy() % 1 == 0:
                A, B = next(test_iter)
                A2B, B2A, A2B2A, B2A2B, A2A, B2B = sample(A, B)
                # img =  im.immerge(np.concatenate([tf.image.grayscale_to_rgb(A[:2]),
                #                                   tf.image.grayscale_to_rgb(A2B[:2]),
                #                                   tf.image.grayscale_to_rgb(A2B2A[:2]),
                #                                   tf.image.grayscale_to_rgb(A2A[:2]),
                #                                   tf.image.grayscale_to_rgb(B[:2]),
                #                                   tf.image.grayscale_to_rgb(B2A[:2]),
                #                                   tf.image.grayscale_to_rgb(B2A2B[:2]),
                #                                   tf.image.grayscale_to_rgb(B2B[:2])], axis=0), n_rows=2)
                img = im.immerge(np.concatenate([A[0],
                                                 A2B[0],
                                                 A2B2A[0],
                                                 A2A[0],
                                                 B[0],
                                                 B2A[0],
                                                 B2A2B[0],
                                                 B2B[0]], axis=0), n_rows=2)


                im.imwrite(img, py.join(sample_dir, 'iter-%09d.jpg' % G_optimizer.iterations.numpy()))
                if len(img.shape) == 3:
                    img = tf.expand_dims(img, axis=0)
                # tl.summary_img({'images': img}, step=G_optimizer.iterations,
                #                name='images')

        # save checkpoint
        checkpoint.save(ep)
