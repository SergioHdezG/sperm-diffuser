import functools

import cv2

import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tensorflow.keras as keras
import tf2lib as tl
import tf2gan as gan
import tqdm

import data
import moduleModified
import module
from spllib.spline_utils import bezierSpline2image

# tf.config.run_functions_eagerly(True)


# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--output_dir', default='BezierSplines2SingleSpermModified')
py.arg('--dataset', default='BezierSplines2SingleSperm')
py.arg('--datasets_dir', default='datasets')
py.arg('--load_size', type=int, default=140)  # load image to this size
py.arg('--crop_size', type=int, default=140)  # then crop to this size
py.arg('--batch_size', type=int, default=16)
py.arg('--epochs', type=int, default=200)
py.arg('--epoch_decay', type=int, default=100)  # epoch to start decaying learning rate
py.arg('--lr', type=float, default=0.00005)
py.arg('--beta_1', type=float, default=0.5)
py.arg('--adversarial_loss_mode', default='lsgan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
py.arg('--gradient_penalty_mode', default='wgan-gp', choices=['none', 'dragan', 'wgan-gp'])
py.arg('--gradient_penalty_weight', type=float, default=10.0)
py.arg('--cycle_loss_weight', type=float, default=10.0)
py.arg('--identity_loss_weight', type=float, default=0.0)
py.arg('--pool_size', type=int, default=50)  # pool size to store fake samples
args = py.args()

trainAfolder = 'trainSingleBezierSplines'
trainBfolder = 'trainSingleSperm'
testAfolder = 'testSingleBezierSplines'
testBfolder = 'testSingleSperm'
extensionA = '*.json'
extensionB = '*.png'

# output_dir
output_dir = py.join('output', args.output_dir)
py.mkdir(output_dir)

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)


# ==============================================================================
# =                                    data                                    =
# ==============================================================================

A_img_paths = py.glob(py.join(args.datasets_dir, args.dataset, trainAfolder), extensionA)
B_img_paths = py.glob(py.join(args.datasets_dir, args.dataset, trainBfolder), extensionB)
A_B_dataset, len_dataset = data.make_zip_datasetSplines2SSingleperm(A_img_paths, B_img_paths, args.batch_size, args.load_size, args.crop_size, training=True, repeat=False)

A2B_pool = data.ItemPool(args.pool_size)
B2A_pool = data.ItemPool(args.pool_size)

A_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, testAfolder), extensionA)
B_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, testBfolder), extensionB)
A_B_dataset_test, _ = data.make_zip_datasetSplines2SSingleperm(A_img_paths_test, B_img_paths_test, args.batch_size, args.load_size, args.crop_size, training=False, repeat=True)


# ==============================================================================
# =                                   models                                   =
# ==============================================================================

G_A2B_Enc = moduleModified.LatentSplines2SingleSpermGeneratorEncoder(input_shape=(10))
G_A2B_Dec = moduleModified.LatentSplines2SingleSpermGeneratorDecoder(output_channels=1, dim=32, n_downsamplings=2, norm='instance_norm')

G_B2A_Enc = moduleModified.LatentSingleSperm2SplineGeneratorEncoder(input_shape=(140, 140, 1), output_channels=10, dim=32, n_downsamplings=2, norm='instance_norm')
G_B2A_Dec = moduleModified.LatentSingleSperm2SplineGeneratorDecoder()

D_A = module.SplineDiscriminator(input_shape=(10, ), dim=32, norm='instance_norm')
D_B = module.ConvDiscriminator(input_shape=(140, 140, 1), dim=32)

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
    A2B = G_A2B_Dec([A_latent, codes], training=training)
    return A2B, A_latent

def G_B2A(B, training=True):
    B_latent, B_code = G_B2A_Enc(B, training=training)
    B_code_u, B_code_logvar = tf.split(B_code, 2, axis=1)
    B2A = G_B2A_Dec(B_latent, training=training)
    return B2A, B_latent, B_code_u, B_code_logvar

def G_A2A(A, training=True):
    A_latent = G_A2B_Enc(A, training=training)
    A2A = G_B2A_Dec(A_latent, training=training)
    return A2A

def G_B2B(B, training=True):
    B_latent, B_code = G_B2A_Enc(B, training=training)
    B_code_u, B_code_logvar = tf.split(B_code, 2, axis=1)
    codes = reparametrize(B_code_u, B_code_logvar)
    B2B = G_A2B_Dec([B_latent, codes], training=training)
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

@tf.function
def train_G(A, B):
    with tf.GradientTape() as t:

        # Standard Cycle-GAN
        B2A, B_latent, B_code_u, B_code_logvar = G_B2A(B, training=True)
        rand_code = tf.random.normal(shape=B_code_u.shape)
        A2B, A_latent = G_A2B(A, rand_code, training=True)

        A2B2A, A2B2A_latent, _, _ = G_B2A(A2B, training=True)
        z = reparametrize(B_code_u, B_code_logvar)
        B2A2B, B2A2B_latent = G_A2B(B2A, z, training=True)

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

        G_loss = (A2B_g_loss + B2A_g_loss) + (A2B2A_cycle_loss + B2A2B_cycle_loss) * args.cycle_loss_weight + (A2A_id_loss + B2B_id_loss) * args.identity_loss_weight

        # Modification
        # Identity in splines latent spaces
        spline_latent_loss = tf.reduce_mean(tf.losses.mse(A_latent, A2B2A_latent) + tf.losses.mse(B_latent, B2A2B_latent))

        # Maintain code_u and code_logvar as a normal distribution N(0, 1)
        logpz = log_normal_pdf(z, 0., 0.) # mean = 0 and logvar = 0 equivalent to var = 1

        G_loss += spline_latent_loss - logpz
    G_grad = t.gradient(G_loss, G_A2B_Enc.trainable_variables + G_A2B_Dec.trainable_variables+  G_B2A_Enc.trainable_variables + G_B2A_Dec.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, G_A2B_Enc.trainable_variables + G_A2B_Dec.trainable_variables+  G_B2A_Enc.trainable_variables + G_B2A_Dec.trainable_variables))

    return A2B, B2A, {'A2B_g_loss': A2B_g_loss,
                      'B2A_g_loss': B2A_g_loss,
                      'A2B2A_cycle_loss': A2B2A_cycle_loss,
                      'B2A2B_cycle_loss': B2A2B_cycle_loss,
                      'A2A_id_loss': A2A_id_loss,
                      'B2B_id_loss': B2B_id_loss,
                      'spline_latent_loss': spline_latent_loss}


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
            if G_optimizer.iterations.numpy() % 250 == 0:
                A, B = next(test_iter)
                A2B, B2A, A2B2A, B2A2B, A2A, B2B = sample(A, B)
                A2B = tf.image.grayscale_to_rgb(A2B[:1])
                A2B = tf.image.resize(A2B, [300, 300])
                B = tf.image.grayscale_to_rgb(B[:1])
                B = tf.image.resize(B, [300, 300])
                B2A2B = tf.image.grayscale_to_rgb(B2A2B[:1])
                B2A2B = tf.image.resize(B2A2B, [300, 300])
                B2B = tf.image.grayscale_to_rgb(B2B[:1])
                B2B = tf.image.resize(B2B, [300, 300])

                A_img = (tf.convert_to_tensor(bezierSpline2image(A[:1].numpy()), dtype=tf.float32)/255.)*2 -1
                B2A_img = (tf.convert_to_tensor(bezierSpline2image(B2A[:1].numpy()), dtype=tf.float32)/255.)*2 -1
                A2B2A_img = (tf.convert_to_tensor(bezierSpline2image(A2B2A[:1].numpy()), dtype=tf.float32)/255.)*2 -1
                A2A_img = (tf.convert_to_tensor(bezierSpline2image(A2A[:1].numpy()), dtype=tf.float32)/255.)*2 -1

                img = im.immerge(np.concatenate([A_img, A2B, A2B2A_img, A2A_img, B, B2A_img, B2A2B, B2B], axis=0), n_rows=2)
                im.imwrite(img, py.join(sample_dir, 'iter-%09d-img.jpg' % G_optimizer.iterations.numpy()))

        # save checkpoint
        checkpoint.save(ep)
