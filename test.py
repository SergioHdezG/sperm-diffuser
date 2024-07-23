import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tf2lib as tl

import data
import module

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--experiment_dir', default="output/november/SynthBezier2RealImg")
py.arg('--batch_size', type=int, default=8)
# py.arg('--testA', type=str, default='testFullBezierImage')  # pool size to store fake samples
py.arg('--testA', type=str, default='synthdata_20240308/images2')  # pool size to store fake samples
py.arg('--testB', type=str, default='testFullSpermImage')  # pool size to store fake samples
py.arg('--load_size_A', type=int, default=500)  # load image to this size
py.arg('--load_size_B', type=int, default=500)  # load image to this size
py.arg('--crop_size', type=int, default=500)  # then crop to this size
py.arg('--dataset', default='SynthBezier2RealImg')
py.arg('--checkpoint_dir', default="output/november/SynthBezier2RealImg3/checkpoints_BCK2")
test_args = py.args()
args = py.args_from_yaml(py.join(test_args.experiment_dir, 'settings.yml'))
args.__dict__.update(test_args.__dict__)


# ==============================================================================
# =                                    test                                    =
# ==============================================================================

# data
A_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, args.testA), '*.png')
B_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, args.testB), '*.png')
A_dataset_test = data.make_dataset(A_img_paths_test, args.batch_size, args.load_size_A, args.crop_size,
                                   training=False, drop_remainder=False, shuffle=False, repeat=1)
B_dataset_test = data.make_dataset(B_img_paths_test, args.batch_size, args.load_size_B, args.crop_size,
                                   training=False, drop_remainder=False, shuffle=False, repeat=1)

# A_B_dataset_test, _ = data.make_zip_dataset(A_img_paths_test, B_img_paths_test, 1, args.load_size_A, args.crop_size, training=False, repeat=True, load_size_B=args.load_size_B)


# model
G_A2B = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))
G_B2A = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))

# resotre
tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), args.checkpoint_dir).restore()


@tf.function
def sample_A2B(A):
    A2B = G_A2B(A, training=False)
    A2B2A = G_B2A(A2B, training=False)
    return A2B, A2B2A


@tf.function
def sample_B2A(B):
    B2A = G_B2A(B, training=False)
    B2A2B = G_A2B(B2A, training=False)
    return B2A, B2A2B



save_dir = py.join(args.experiment_dir, '08052024_500x500/samples_testing3', 'A2B')
py.mkdir(save_dir)
i = 0
for A in A_dataset_test:
    A2B, A2B2A = sample_A2B(A)
    for A_i, A2B_i, A2B2A_i in zip(A, A2B, A2B2A):
        # img = np.concatenate([A_i.numpy(), A2B_i.numpy(), A2B2A_i.numpy()], axis=1)
        img = np.concatenate([A_i.numpy(), A2B_i.numpy()], axis=1)

        im.imwrite(img, py.join(save_dir, py.name_ext(A_img_paths_test[i])))
        i += 1

save_dir = py.join(args.experiment_dir, '08052024_500x500/samples_testing3', 'B2A')
py.mkdir(save_dir)
i = 0
for B in B_dataset_test:
    B2A, B2A2B = sample_B2A(B)
    for B_i, B2A_i, B2A2B_i in zip(B, B2A, B2A2B):
        # img = np.concatenate([B_i.numpy(), B2A_i.numpy(), B2A2B_i.numpy()], axis=1)
        img = np.concatenate([B_i.numpy(), B2A_i.numpy()], axis=1)
        im.imwrite(img, py.join(save_dir, py.name_ext(B_img_paths_test[i])))
        i += 1

save_dir = py.join(args.experiment_dir, '08052024_500x500/samples_testing4', 'A2B')
py.mkdir(save_dir)
i = 0
for A in A_dataset_test:
    A2B, A2B2A = sample_A2B(A)
    for A2B_i in A2B:
        im.imwrite(A2B_i.numpy(), py.join(save_dir, py.name_ext(A_img_paths_test[i])))
        i += 1

save_dir = py.join(args.experiment_dir, '08052024_500x500/samples_testing4', 'B2A')
py.mkdir(save_dir)
i = 0
for B in B_dataset_test:
    B2A, B2A2B = sample_B2A(B)
    for B2A_i in B2A:
        im.imwrite(B2A_i.numpy(), py.join(save_dir, py.name_ext(B_img_paths_test[i])))
        i += 1