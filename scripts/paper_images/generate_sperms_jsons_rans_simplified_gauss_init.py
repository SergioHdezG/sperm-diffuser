import json
import math
import pdb
import random
from json import JSONEncoder

import diffuser.datasets
import diffuser.sampling as sampling
import diffuser.utils as utils
from diffuser.utils.arrays import batch_to_device, to_np, to_device, apply_dict
import einops
import numpy as np
import torch
from diffuser.utils.training import cycle
import os
from diffuser.utils.video import save_video
from diffuser.models.helpers import apply_conditioning, apply_direction_coord_end_conditioning
from PIL import Image
from diffuser.environments.utils.sperm_rendering import vec2angle
from diffuser.environments.utils.Bezier import Bezier
import matplotlib as mpl

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

def save2json(dict, path):
    class NumpyArrayEncoder(JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return JSONEncoder.default(self, obj)

    encodedNumpyData = json.dumps(dict, cls=NumpyArrayEncoder)
    with open(path, 'w') as f:
        json.dump(encodedNumpyData, f)

mean_n_sperms = 174.6
std_n_sperm = 34.13
n_sequences = 3
mean_displacement = 23
std_displacement = 6
disp_min = 16
disp_max = 60
seq_len = 1

savebase = 'diffuser/datasets/synthdata_progressive_sperm'

make_subfolders = False
use_end_condition = False

n_sperm_per_seq = [int(np.random.normal(mean_n_sperms, std_n_sperm)) for _ in range(n_sequences)]

class Parser(utils.Parser):
    dataset: str = 'SingleSpermBezierIncrementsDataAugSimplified-v0'
    config: str = 'config.sperm'

args = Parser().parse_args('plan')

#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#

## load diffusion model and value function from disk
diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed
)

diffusion = diffusion_experiment.ema
diff_trainer = diffusion_experiment.trainer
dataset, _ = diffusion_experiment.dataset
renderer, _ = diffusion_experiment.renderer

logger_config = utils.Config(
    utils.Logger,
    renderer=renderer,
    logpath=args.savepath,
    vis_freq=args.vis_freq,
    max_render=args.max_render,
)

logger = logger_config()

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

dataloader = cycle(torch.utils.data.DataLoader(
            dataset, batch_size=int(np.max(n_sperm_per_seq)), num_workers=0, shuffle=True, pin_memory=True
        ))

gauss_means = [-0.7101236, 0.012660508, -0.4724761, -0.011081022, -0.30175892, 0.013444074, -0.111315355, 0.03597443,
              0.11064559, -0.012201412]
gauss_std = [0.18651162, 0.060374398, 0.16854781, 0.1470919, 0.12989329, 0.19624966, 0.15786962, 0.10110918, 0.053684674,
            0.03615527]
gauss_displ_mean = [3.637117, -0.00015417, 0.8920469, -0.0450125]
gauss_displ_std = [1.312792, 0.000920, 0.33587, 0.29901]


def sample_param(index, batch):
    return np.random.normal(gauss_means[index], gauss_std[index], batch)


def sample_displacement(batch):
    return np.transpose([np.random.normal(gauss_displ_mean[0], gauss_displ_std[0], batch),
                         np.random.normal(gauss_displ_mean[1], gauss_displ_std[1], batch)])


def sample_new_cond(shape, batch_size):
    sequences = np.zeros(shape)
    for k in range(len(gauss_means)):
        sequences[:, k] = sample_param(k, batch_size)

    sequences[:, -4] = np.random.normal(gauss_displ_mean[0], gauss_displ_std[0], n_sperm_per_seq[i])
    sequences[:, -3] = np.random.normal(gauss_displ_mean[1], gauss_displ_std[1], n_sperm_per_seq[i])
    sequences[:, -2] = np.random.normal(gauss_displ_mean[2], gauss_displ_std[2], n_sperm_per_seq[i])
    sequences[:, -1] = np.random.normal(gauss_displ_mean[3], gauss_displ_std[3], n_sperm_per_seq[i])

    return sequences

def save_numpy_array_as_gif(frames, gif_path, duration=100):
    # frames: A list or numpy array of image frames
    # gif_path: The path where you want to save the GIF
    # duration: The duration (in milliseconds) for each frame

    # Ensure frames are in the correct format (8-bit, RGBA)
    frames = [Image.fromarray(frame.astype('uint8'), 'RGB') for frame in frames]

    # Save the frames as a GIF
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0  # 0 means loop indefinitely
    )


def angles_to_vectors(angles):
    # Convert angles from degrees to radians
    angles_rad = np.radians(angles)
    # Compute x and y components of the unit vectors
    x_components = np.cos(angles_rad)
    y_components = np.sin(angles_rad)
    # Stack x and y components horizontally to form vectors
    vectors = np.column_stack((x_components, y_components))
    return vectors

np.random.seed(1)
random.seed(1)
torch.manual_seed(1)


def create_conditions_sperm_aug_simplified_env(batch):
    mean_velocity = 3.27
    std_velocity = 0.71
    if use_end_condition:
        batch.conditions[args.horizon - 1] = batch.conditions[args.horizon - 1][:n_sperm_per_seq[i]]
    batch_size = batch.conditions[0].shape[0]

    return batch

def create_conditions_sperm_aug_env(batch):
    if use_end_condition:
        batch.conditions[args.horizon - 1] = batch.conditions[args.horizon - 1][:n_sperm_per_seq[i]]
    batch_size = batch.conditions[0].shape[0]

    head_init_coords = np.zeros((batch_size, 2))
    head_init_coords[:, 0] = np.random.rand(batch_size) * 1280
    head_init_coords[:, 1] = np.random.rand(batch_size) * 1024
    # angle = np.expand_dims(np.clip(np.random.rand(0.01114, 0.58136, batch_size), -0.9999, 0.9999), axis=-1)
    angle_init = np.random.randint(0, 360, batch_size)
    if use_end_condition:
        unit_vector = angles_to_vectors(angle_init)
        displacement = np.clip(np.random.normal(mean_displacement, std_displacement, (batch_size, 1)), disp_min,
                               disp_max)

        x_square = np.square(np.expand_dims(unit_vector[:, 0], axis=-1) * displacement)
        y_square = np.square(np.expand_dims(unit_vector[:, 1], axis=-1) * displacement)

        displacement_x = np.sqrt(x_square)
        displacement_y = np.sqrt(y_square)
        displacement = np.concatenate([displacement_x, displacement_y], axis=-1)

        head_end_coords = head_init_coords + displacement
        angle_end = angle_init + np.random.normal(0, 1)
    angle_init = (((angle_init / 180) + 1) % 2) - 1.
    if use_end_condition:
        angle_end = (((angle_end / 180) + 1) % 2) - 1.
    init_cond = np.random.rand(batch_size, batch.conditions[0].shape[1])
    init_cond[:, -3] = angle_init
    init_cond[:, -2] = (head_init_coords[:, 0] / 1280) * 2 - 1
    init_cond[:, -1] = (head_init_coords[:, 1] / 1024) * 2 - 1
    init_normalized = dataset.normalizer.normalize(init_cond, 'observations')
    batch.conditions[0][:, -3] = torch.from_numpy(init_normalized[:, -3])
    batch.conditions[0][:, -2:] = torch.from_numpy(init_normalized[:, -2:])
    if use_end_condition:
        end_cond = np.random.rand(batch_size, batch.conditions[0].shape[1])
        end_cond[:, -3] = angle_end
        end_cond[:, -2] = (head_end_coords[:, 0] / 1280) * 2 - 1
        end_cond[:, -1] = (head_end_coords[:, 1] / 1024) * 2 - 1

        end_normalized = dataset.normalizer.normalize(end_cond, 'observations')
        batch.conditions[args.horizon - 1][:, -3] = torch.from_numpy(end_normalized[:, -3])
        batch.conditions[args.horizon - 1][:, -2:] = torch.from_numpy(end_normalized[:, -2:])

    return batch


def rotate2Dvec(v, theta):
    c, s = np.cos(theta), np.sin(theta)
    r = np.array(((c, -s), (s, c)))
    v = np.dot(r, v)
    return v

for i in range(n_sequences):
    observations = []
    paths = []
    for j in range(seq_len):
        batch = dataloader.__next__()
        batch.conditions[0] = batch.conditions[0][:n_sperm_per_seq[i]]

        if batch.conditions[0].shape[-1] == 14:
            batch = create_conditions_sperm_aug_simplified_env(batch)
        else:
            batch = create_conditions_sperm_aug_env(batch)

        # conditions = to_device(batch.conditions, 'cuda:0')
        # # diffusion.apply_conditioning = apply_direction_coord_end_conditioning
        # unnormalized_conds = dataset.normalizer.unnormalize(batch.conditions[0], 'observations')
        # init_cond = np.random.rand(n_sperm_per_seq[i], batch.conditions[0].shape[1])
        # init_cond[:, -4] = np.random.normal(gauss_displ_mean[0], gauss_displ_std[0], n_sperm_per_seq[i])
        # init_cond[:, -3] = np.random.normal(gauss_displ_mean[1], gauss_displ_std[1], n_sperm_per_seq[i])
        # init_cond[:, -2] = np.random.normal(gauss_displ_mean[2], gauss_displ_std[2], n_sperm_per_seq[i])
        # init_cond[:, -1] = np.random.normal(gauss_displ_mean[3], gauss_displ_std[3], n_sperm_per_seq[i])

        gauss_cond = sample_new_cond((n_sperm_per_seq[i], batch.conditions[0].shape[1]), n_sperm_per_seq[i])

        batch.conditions[0] = dataset.normalizer.normalize(torch.from_numpy(gauss_cond), 'observations')
        conditions = to_device(batch.conditions, 'cuda:0')

        ## [ n_samples x horizon x (action_dim + observation_dim) ]
        samples = diffusion(conditions)

        trajectories = to_np(samples.trajectories)

        ## [ n_samples x horizon x observation_dim ]
        normed_observations = trajectories[:, :, dataset.action_dim:]

        ## [ n_samples x (horizon + 1) x observation_dim ]
        unnormalized_obs = dataset.normalizer.unnormalize(normed_observations, 'observations')
        observations.append(unnormalized_obs)
        # observations.append(np.expand_dims(dataset.normalizer.unnormalize(normed_observations, 'observations')[:, -1, :], axis=1))



    observations = np.dstack(observations)
    for k in range(len(observations)):
        if make_subfolders:
            path_seq = os.path.join(savebase, args.diffusion_loadpath.split('/')[-1], f'field_{i}', 'json_bezier_spline', f'field_{i}_{k}')
        else:
            path_seq = os.path.join(savebase, args.diffusion_loadpath.split('/')[-1], f'field_{i}_{k}')

        os.makedirs(path_seq, exist_ok=True)

        state = np.array(observations[k])

        first_iter = True
        init_head = (np.random.rand(2) * 2) - 1
        aux_head = init_head
        aux_velocity = [0., 0.]
        traj = []
        rand_rotation = np.random.rand()*2*np.pi

        for kk in range(state.shape[0]):

            img_size = (1024, 1280)
            if state.shape[-1] == 14 or state.shape[-1] == 12:
                params = np.reshape(state[kk, :10], (5, 2))
                velocity = state[kk, -4:-2]
                correction_angle_vector = state[kk, -2:]

                rot_velocity = rotate2Dvec(np.array(velocity), rand_rotation)
                rot_correction_angle_vector = rotate2Dvec(np.array(correction_angle_vector), rand_rotation)

                velocity_angle = vec2angle(rot_velocity, normalize=False)
                correction_angle = vec2angle(rot_correction_angle_vector, normalize=False)


                aux_velocity_angle = vec2angle(np.array(velocity), normalize=False)
                aux_correction_angle = vec2angle(np.array(correction_angle_vector), normalize=False)

                # a1 = np.radians(aux_velocity_angle - aux_correction_angle)
                # value1 = (a1 + np.pi) % (2 * np.pi) - np.pi
                # a2 = np.radians(velocity_angle - correction_angle)
                # value2 = (a2 + np.pi) % (2 * np.pi) - np.pi
                # if value1 > np.pi/2 or value2 > np.pi/2 or np.abs(value2-value1)> 0.1:
                #     print('alpha diff: ', value1, value2)
                #     rot_velocity = rotate2Dvec(np.array(velocity), rand_rotation)
                #     rot_correction_angle_vector = rotate2Dvec(np.array(correction_angle_vector), rand_rotation)
                #
                #     velocity_angle = vec2angle(rot_velocity, normalize=False)
                #     correction_angle = vec2angle(rot_correction_angle_vector, normalize=False)
                # rot = mpl.transforms.Affine2D().rotate_deg(correction_angle)
                # rot_params = rot.transform(params)
                
                linspace = np.linspace(0., 1., num=20)
                norm_curve = Bezier.Curve(linspace, params)


                params_p = (params + 1) * 70
                curve_p = (norm_curve + 1) * 70
                angle_p = correction_angle  # * 180.
                aux_x = ((aux_head[0] + 1.) / 2.) * img_size[1]
                aux_y = img_size[0]-((aux_head[1] + 1.) / 2.) * img_size[0]


                spline_params = {"frame": 'None',
                                 "spline_params": params_p,
                                 "spline_line_space": curve_p,
                                 "correction_angle": float(angle_p),
                                 "img_shape": (140, 140),
                                 'head_coordinates': [float(aux_x), float(aux_y), kk],
                                 'sperm_id': str(k)
                                 }

                norm_x = (aux_x / img_size[1]) * 2 - 1
                norm_y = (aux_y / img_size[0]) * 2 - 1
                norm_vel_x = (aux_velocity[0] / 20)
                norm_vel_y = (aux_velocity[1] / 20)

                if correction_angle > 180.:
                    angle = (correction_angle - 360)/180
                else:
                    angle = correction_angle/180

                obs = np.concatenate([np.reshape(norm_curve, 40), state[kk, :10], [norm_vel_x, norm_vel_y, angle, norm_x, norm_y]])

                aux_head[0] = aux_head[0] + ((rot_velocity[0] / img_size[1]) * 2)
                aux_head[1] = aux_head[1] + ((rot_velocity[1] / img_size[0]) * 2)
                aux_velocity[0] = rot_velocity[0]
                aux_velocity[1] = rot_velocity[1]

                traj.append(obs)
            else:
                params = np.reshape(state[kk, 40:50], (5, 2))

                curve = np.reshape(state[kk, :40], (20, 2))
                angle = state[kk, -3]
                aux_head.append(state[kk, -2:])
                displacement = state[kk, -5:-3]

                params = (params + 1) * 70
                curve = (curve + 1) * 70
                angle = angle * 180.

                if first_iter:
                    first_iter = False
                else:
                    aux_head[kk][0] = aux_head[kk - 1][0] + ((displacement[0] * 20 / img_size[1]) * 2)
                    aux_head[kk][1] = aux_head[kk - 1][1] + ((displacement[1] * 20 / img_size[0]) * 2)

                aux_x = ((aux_head[kk][0] + 1.) / 2.) * img_size[1]
                aux_y = img_size[0]-((aux_head[kk][1] + 1.) / 2.) * img_size[0]

                spline_params = {"frame": 'None',
                                 "spline_params": params,
                                 "spline_line_space": curve,
                                 "correction_angle": float(angle),
                                 "img_shape": (140, 140),
                                 'head_coordinates': [float(aux_x), float(aux_y), kk],
                                 'sperm_id': str(k)
                                 }

            save2json(spline_params, os.path.join(path_seq, str(kk).zfill(3) + '.json'))

        if state.shape[-1] == 14 or state.shape[-1] == 12:
            paths.append(traj)

    observations = np.concatenate(observations, axis=1)
    if state.shape[-1] == 14 or state.shape[-1] == 12:
        paths = np.array(paths)

    savepath = os.path.join(args.savepath, f'sample-{i}')
    if state.shape[-1] == 14 or state.shape[-1] == 12:
        images, images_debug = diff_trainer.renderer[0].composite_full_image(savepath, paths, dim=(1024, 1280), resize=(512, 640), ext='.png', use_ema=False)
    else:
        images, images_debug = diff_trainer.renderer[0].composite_full_image(savepath, observations, dim=(1024, 1280), resize=(512, 640), ext='.png')


    save_video(os.path.join(args.savepath, f'sample-{i}.mp4'), images, fps=10)
    save_video(os.path.join(args.savepath, f'sample-{i}_debug.mp4'), images_debug, fps=10)

    save_numpy_array_as_gif(images_debug, os.path.join(args.savepath, f'sample-{i}_debug.gif'), duration=150)





