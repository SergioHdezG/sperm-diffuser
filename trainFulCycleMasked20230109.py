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
import moduleModified

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--output_dir', default='BezierImage2SpermImageMaskedNoD_A')
py.arg('--dataset', default='FulBezier2FulSpermMasked')
py.arg('--datasets_dir', default='datasets')
py.arg('--load_size', type=int, default=286)  # load image to this size
py.arg('--crop_size', type=int, default=256)  # then crop to this size
py.arg('--batch_size', type=int, default=2)
py.arg('--epochs', type=int, default=200)
py.arg('--epoch_decay', type=int, default=100)  # epoch to start decaying learning rate
py.arg('--G_lr', type=float, default=0.0002)
py.arg('--D_lr', type=float, default=0.0002)
py.arg('--beta_1', type=float, default=0.5)
py.arg('--adversarial_loss_mode', default='lsgan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
py.arg('--gradient_penalty_mode', default='none', choices=['none', 'dragan', 'wgan-gp'])
py.arg('--reconstruction_loss_mode', default='none', choices=['none', 'synth_bce'])
py.arg('--gradient_penalty_weight', type=float, default=10.0)
py.arg('--cycle_loss_weight', type=float, default=10.0)
py.arg('--identity_loss_weight', type=float, default=0.0)
py.arg('--normal_prob_loss_weight', type=float, default=0.001)
py.arg('--pool_size', type=int, default=50)  # pool size to store fake samples
py.arg('--trainA', type=str, default='trainFulBezierImage')  # pool size to store fake samples
py.arg('--trainB', type=str, default='trainFulSpermImage')  # pool size to store fake samples
py.arg('--trainMaskA', type=str, default='trainSynthMask')  # pool size to store fake samples
py.arg('--trainMaskB', type=str, default='trainRealMask')  # pool size to store fake samples

py.arg('--testA', type=str, default='testFulBezierImage')  # pool size to store fake samples
py.arg('--testB', type=str, default='testFulSpermImage')  # pool size to store fake samples
py.arg('--testMaskA', type=str, default='testSynthMask')  # pool size to store fake samples
py.arg('--testMaskB', type=str, default='testRealMask')  # pool size to store fake samples
py.arg('--fileExtension', type=str, default='png')  # pool size to store fake samples
args = py.args()

channels = 1

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
A_mask_paths = py.glob(py.join(args.datasets_dir, args.dataset, args.trainMaskA), '*.'+args.fileExtension)
B_mask_paths = py.glob(py.join(args.datasets_dir, args.dataset, args.trainMaskB), '*.'+args.fileExtension)
A_B_dataset, len_dataset = data.make_zip_dataset_masked(A_img_paths, B_img_paths, A_mask_paths, B_mask_paths, args.batch_size, args.load_size, args.crop_size, training=True, repeat=False, channels=channels)

A2B_pool = data.ItemPool(args.pool_size)
B2A_pool = data.ItemPool(args.pool_size)

A_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, args.testA), '*.'+args.fileExtension)
B_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, args.testB), '*.'+args.fileExtension)
A_mask_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, args.trainMaskA), '*.'+args.fileExtension)
B_mask_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, args.trainMaskB), '*.'+args.fileExtension)
A_B_dataset_test, _ = data.make_zip_dataset_masked(A_img_paths_test, B_img_paths_test, A_mask_paths_test, B_mask_paths_test, 1, args.load_size, args.crop_size, training=False, repeat=True, channels=channels)


# ==============================================================================
# =                                   models                                   =
# ==============================================================================

G_A2B_Enc = moduleModified.ForwardResidualGeneratorEncoder(input_shape=(args.crop_size, args.crop_size, channels), output_channels=channels)
G_B2A_Enc = moduleModified.BackwardResidualGeneratorEncoder(input_shape=(args.crop_size, args.crop_size, channels), output_channels=channels)


G_A2B_Dec = moduleModified.ForwardResidualGeneratorDecoder(latent_input_shape=G_A2B_Enc.outputs[0].shape[1:],
                                                            code_input_shape=(*G_B2A_Enc.outputs[1].shape[1:-1], G_B2A_Enc.outputs[1].shape[-1]//2),
                                                            res1_input_shape=G_A2B_Enc.outputs[1].shape[1:],
                                                            res2_input_shape=G_A2B_Enc.outputs[2].shape[1:],
                                                            res3_input_shape=G_A2B_Enc.outputs[3].shape[1:],
                                                            output_channels=channels)

D_B = module.ConvDiscriminator(input_shape=(args.crop_size, args.crop_size, channels))

d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn(args.adversarial_loss_mode)

if args.reconstruction_loss_mode == 'synth_bce':
    cycle_loss_fn_A = tf.losses.BinaryCrossentropy()
    identity_loss_fn_A = tf.losses.BinaryCrossentropy()
    cycle_loss_fn_B = tf.losses.MeanAbsoluteError()
    identity_loss_fn_B = tf.losses.MeanAbsoluteError()
    reconstruction_mask_loss = tf.losses.MeanAbsoluteError()
else:
    cycle_loss_fn_A = tf.losses.MeanAbsoluteError()
    identity_loss_fn_A = tf.losses.MeanAbsoluteError()
    cycle_loss_fn_B = tf.losses.MeanAbsoluteError()
    identity_loss_fn_B = tf.losses.MeanAbsoluteError()
    reconstruction_mask_loss = tf.losses.MeanAbsoluteError()

G_lr_scheduler = module.LinearDecay(args.G_lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
D_lr_scheduler = module.LinearDecay(args.D_lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
G_optimizer = keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=args.beta_1)
D_optimizer = keras.optimizers.Adam(learning_rate=D_lr_scheduler, beta_1=args.beta_1)


# ==============================================================================
# =                                 train step                                 =
# ==============================================================================

def G_A2B(A, codes, training=True):
    A_latent, res1, res2, res3 = G_A2B_Enc(A, training=training)
    A2B = G_A2B_Dec([A_latent, codes, res1, res2, res3], training=training)
    return A2B, A_latent, res1, res2, res3

def G_B2B(B, training=True):
    B_latent, B_code, res1, res2, res3 = G_B2A_Enc(B, training=training)
    B2B = G_A2B_Dec([B_latent, B_code, res1, res2, res3], training=training)
    return B2B, B_latent, B_code, res1, res2, res3


@tf.function
def reparametrize(u, logvar):
    eps = tf.random.normal(shape=u.shape)
    return eps * tf.exp(logvar * .5) + u

@tf.function
def log_normal_pdf(sample, mean, logvar, raxis=-1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)

@tf.function
def split_mean_var(input):
    mean, logvar = tf.split(input, num_or_size_splits=2, axis=-1)
    samples = reparametrize(mean, tf.clip_by_value(logvar, -10., 10.))
    return samples

# @tf.function
def train_G(A, B, A_mask, B_mask):
    with tf.GradientTape() as t:
        # Standard Cycle-GAN
        B2B, B_latent, B_code, B_res1, B_res2, B_res3 = G_B2B(B, training=True)
        # rand_code = tf.random.normal(shape=(*B_code.shape[:-1], B_code.shape[-1]//2))
        code = split_mean_var(B_code)
        A2B, A_latent, A_res1, A_res2, A_res3 = G_A2B(A, code, training=True)

        # Standard Cycle-GAN
        A2B_d_logits = D_B(A2B, training=True)
        B2B_d_logits = D_B(B2B, training=True)

        A2B_g_loss = g_loss_fn(A2B_d_logits)
        B2B_g_loss = g_loss_fn(B2B_d_logits)

        B2B_mask_loss = reconstruction_mask_loss(A*A_mask, B2B*A_mask)
        A2B_mask_loss = reconstruction_mask_loss(B*A_mask, A2B*A_mask)

        G_loss = A2B_g_loss + B2B_g_loss + B2B_mask_loss + A2B_mask_loss

        # Modification
        # Identity in splines latent spaces
        latent_loss = tf.reduce_mean(tf.losses.mse(tf.stop_gradient(A_latent), B_latent))

        G_loss += latent_loss

        # Maintain code_u and code_logvar as a normal distribution N(0, 1)
        B_code_log = split_mean_var(B_code)
        A_latent_log = split_mean_var(A_latent)
        B_latent_log = split_mean_var(B_latent)
        logpz = tf.reduce_mean(log_normal_pdf(B_code_log, 0., 0.)) + \
                tf.reduce_mean(log_normal_pdf(A_latent_log, 0., 0.)) + \
                tf.reduce_mean(log_normal_pdf(B_latent_log, 0., 0.)) # mean = 0 and logvar = 0 equivalent to var = 1

        G_loss += - logpz * args.normal_prob_loss_weight
    G_grad = t.gradient(G_loss, G_A2B_Enc.trainable_variables + G_A2B_Dec.trainable_variables + G_B2A_Enc.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, G_A2B_Enc.trainable_variables + G_A2B_Dec.trainable_variables + G_B2A_Enc.trainable_variables))

    return A2B, B2A, {'A2B_g_loss': A2B_g_loss,
                      'B2B_g_loss': B2B_g_loss,
                      'logpz': logpz,
                      'A2B_mask_loss': A2B_mask_loss,
                      'B2B_mask_loss': B2B_mask_loss,
                      'latent_loss': latent_loss}


@tf.function
def train_D(A, B, A2B, B2A,  A_mask, B_mask):
    with tf.GradientTape() as t:
        B_d_logits = D_B(B, training=True)
        A2B_d_logits = D_B(A2B, training=True)

        B_d_loss, A2B_d_loss = d_loss_fn(B_d_logits, A2B_d_logits)

        D_B_gp = gan.gradient_penalty(functools.partial(D_B, training=True), B, A2B, mode=args.gradient_penalty_mode)

        D_loss = (B_d_loss + A2B_d_loss) + D_B_gp * args.gradient_penalty_weight

    D_grad = t.gradient(D_loss, D_B.trainable_variables)
    D_optimizer.apply_gradients(zip(D_grad, D_B.trainable_variables))

    return {'B_d_loss': B_d_loss + A2B_d_loss,
            'D_B_gp': D_B_gp}


def train_step(A, B, A_mask, B_mask):
    A2B, B2A, G_loss_dict = train_G(A, B, A_mask, B_mask)

    # cannot autograph `A2B_pool`
    A2B = A2B_pool(A2B)  # or A2B = A2B_pool(A2B.numpy()), but it is much slower
    B2A = B2A_pool(B2A)  # because of the communication between CPU and GPU

    D_loss_dict = train_D(A, B, A2B, B2A, A_mask, B_mask)

    return G_loss_dict, D_loss_dict


@tf.function
def sample(A, B):
    B2B, B_latent, B_code, B_res1, B_res2, B_res3 = G_B2B(B, training=False)
    rand_code = tf.random.normal(shape=(*B_code.shape[:-1], B_code.shape[-1] // 2))
    A2B, A_latent, A_res1, A_res2, A_res3 = G_A2B(A, rand_code, training=False)
    return A2B, B2B


# ==============================================================================
# =                                    run                                     =
# ==============================================================================

# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# checkpoint
checkpoint = tl.Checkpoint(dict(G_A2B_Enc=G_A2B_Enc,
                                G_A2B_Dec=G_A2B_Dec,
                                G_B2A_Enc=G_B2A_Enc,
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
        for A, B, A_mask, B_mask in tqdm.tqdm(A_B_dataset, desc='Inner Epoch Loop', total=len_dataset):
            G_loss_dict, D_loss_dict = train_step(A, B, A_mask, B_mask)

            # # summary
            tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')
            tl.summary(D_loss_dict, step=G_optimizer.iterations, name='D_losses')
            tl.summary({'learning rate': G_lr_scheduler.current_learning_rate}, step=G_optimizer.iterations, name='learning rate')

            # sample
            if G_optimizer.iterations.numpy() % 250 == 0:
                try:
                    A, B, A_mask, B_mask = next(test_iter)
                except StopIteration:
                    test_iter = iter(A_B_dataset_test)
                    A, B, A_mask, B_mask = next(test_iter)

                A2B, B2B = sample(A, B)
                img = im.immerge(np.concatenate([A,
                                                  A2B,
                                                  B,
                                                  B2B], axis=0), n_rows=2)

                if channels == 1:
                    img = np.squeeze(np.stack((img,)*3, axis=-1))
                im.imwrite(img, py.join(sample_dir, 'iter-%09d.jpg' % G_optimizer.iterations.numpy()))
                if len(img.shape) == 3:
                    img = tf.expand_dims(img, axis=0)
                tl.summary_img({'images': img}, step=G_optimizer.iterations,
                               name='images')

        # save checkpoint
        checkpoint.save(ep)
