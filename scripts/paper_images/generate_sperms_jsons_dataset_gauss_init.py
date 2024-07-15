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
n_sequences = 28
mean_displacement = 23
std_displacement = 6
disp_min = 16
disp_max = 60
seq_len = 2

savebase = 'diffuser/datasets/synthdata_20240308'

moving_DDPM = 'diffusion/defaults_H16_T20/2024-02-23:14-42'
slow_DDPM = 'diffusion/defaults_H16_T20/2024-02-26:20-07'
inmotile_DDPM = 'diffusion/defaults_H16_T20/2024-03-05:16-29'


make_subfolders = True
use_end_condition = False

n_sperm_per_seq = [int(np.random.normal(mean_n_sperms, std_n_sperm)) for _ in range(n_sequences)]
percentages = np.random.rand(n_sequences, 3)
total_percentage = np.sum(percentages, axis=-1)
normalized_percentages = percentages/np.expand_dims(total_percentage, axis=-1)
instances_per_group = np.asarray(normalized_percentages * np.expand_dims(n_sperm_per_seq, axis=-1), dtype=np.int64)

print('Percentages: ', normalized_percentages)
percentages = np.mean(normalized_percentages, axis=-1)
print('Avg. Percentages: ', percentages)

class Parser(utils.Parser):
    dataset: str = 'SingleSpermBezierIncrementsDataAugSimplified-v0'
    config: str = 'config.sperm'

args = Parser().parse_args('plan')
args.diffusion_loadpath = moving_DDPM
#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#

## load diffusion model and value function from disk
diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, moving_DDPM,
    epoch=args.diffusion_epoch, seed=args.seed
)

diffusion_moving = diffusion_experiment.ema
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

## load diffusion model and value function from disk
diffusion_experiment_slow = utils.load_diffusion(
    args.loadbase, args.dataset, slow_DDPM,
    epoch=args.diffusion_epoch, seed=args.seed
)

diffusion_slow = diffusion_experiment_slow.ema

## load diffusion model and value function from disk
diffusion_experiment_inmotile = utils.load_diffusion(
    args.loadbase, args.dataset, inmotile_DDPM,
    epoch=args.diffusion_epoch, seed=args.seed
)

diffusion_inmotile = diffusion_experiment_inmotile.ema

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

def rotate2Dvec(v, theta):
    c, s = np.cos(theta), np.sin(theta)
    r = np.array(((c, -s), (s, c)))
    v = np.dot(r, v)
    return v

for i in range(n_sequences):
    observations_moving = []
    observations_slow = []
    observations_inmotile = []

    paths = []

    batch = dataloader.__next__()
    batch.conditions[0] = batch.conditions[0][:n_sperm_per_seq[i]]

    for j in range(seq_len):

        if j == 0:
            gauss_cond = sample_new_cond((n_sperm_per_seq[i], dataset.observation_dim), n_sperm_per_seq[i])

            ## batch.conditions[0] = dataset.normalizer.normalize(torch.from_numpy(gauss_cond), 'observations')
            conditions = dataset.normalizer.normalize(torch.from_numpy(gauss_cond), 'observations')
            conditions = {0: torch.from_numpy(gauss_cond)}
            conditions = to_device(conditions, 'cuda:0')
        else:
            normed_observations = np.concatenate([normed_obs_moving, normed_obs_slow, normed_obs_inmotile], axis=0)
            conditions[0] = torch.from_numpy(normed_observations[:, -1])
            conditions = to_device(conditions, 'cuda:0')

        ## [ n_samples x horizon x (action_dim + observation_dim) ]
        samples_moving = diffusion_moving(conditions)
        samples_slow = diffusion_slow(conditions)
        samples_inmotile = diffusion_inmotile(conditions)


        trajectories_moving = to_np(samples_moving.trajectories)[:instances_per_group[i, 0]]
        trajectories_slow = to_np(samples_slow.trajectories)[instances_per_group[i, 0]:instances_per_group[i, 0]+instances_per_group[i, 1]]
        trajectories_inmotile = to_np(samples_inmotile.trajectories)[instances_per_group[i, 0]+instances_per_group[i, 1]:]

        ## [ n_samples x horizon x observation_dim ]
        normed_obs_moving = trajectories_moving[:, :, dataset.action_dim:]
        normed_obs_slow = trajectories_slow[:, :, dataset.action_dim:]
        normed_obs_inmotile = trajectories_inmotile[:, :, dataset.action_dim:]

        ## [ n_samples x (horizon + 1) x observation_dim ]
        unnormalized_obs_moving = dataset.normalizer.unnormalize(normed_obs_moving, 'observations')
        unnormalized_obs_slow = dataset.normalizer.unnormalize(normed_obs_slow, 'observations')
        unnormalized_obs_inmotile = dataset.normalizer.unnormalize(normed_obs_inmotile, 'observations')

        observations_moving.append(unnormalized_obs_moving)
        observations_slow.append(unnormalized_obs_slow)
        observations_inmotile.append(unnormalized_obs_inmotile)

    # observations_moving = np.dstack(observations_moving)
    observations_moving = np.concatenate(observations_moving, axis=1)
    # observations_slow = np.dstack(observations_slow)
    observations_slow = np.concatenate(observations_slow, axis=1)
    # observations_inmotile = np.dstack(observations_inmotile)
    observations_inmotile = np.concatenate(observations_inmotile, axis=1)


    observations = np.concatenate([observations_moving, observations_slow, observations_inmotile], axis=0)
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
            params = np.reshape(state[kk, :10], (5, 2))
            velocity = state[kk, -4:-2]
            correction_angle_vector = state[kk, -2:]

            rot_velocity = rotate2Dvec(np.array(velocity), rand_rotation)
            rot_correction_angle_vector = rotate2Dvec(np.array(correction_angle_vector), rand_rotation)

            velocity_angle = vec2angle(rot_velocity, normalize=False)
            correction_angle = vec2angle(rot_correction_angle_vector, normalize=False)

            aux_velocity_angle = vec2angle(np.array(velocity), normalize=False)
            aux_correction_angle = vec2angle(np.array(correction_angle_vector), normalize=False)

            linspace = np.linspace(0., 1., num=20)
            norm_curve = Bezier.Curve(linspace, params)


            params_p = (params + 1) * 70
            curve_p = (norm_curve + 1) * 70
            angle_p = correction_angle  # * 180.
            aux_x = ((aux_head[0] + 1.) / 2.) * img_size[1]
            aux_y = img_size[0]-((aux_head[1] + 1.) / 2.) * img_size[0]

            if k < len(observations_moving):
                category = 0
            elif k < len(observations_moving) + len(observations_slow):
                category = 1
            else:
                category = 2

            spline_params = {"frame": 'None',
                             "spline_params": params_p,
                             "spline_line_space": curve_p,
                             "correction_angle": float(angle_p),
                             "img_shape": (140, 140),
                             'head_coordinates': [float(aux_x), float(aux_y), kk],
                             'sperm_id': str(k),
                             'category': category,
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

            save2json(spline_params, os.path.join(path_seq, str(kk).zfill(3) + '.json'))

        paths.append(traj)

    observations = np.concatenate(observations, axis=1)
    paths = np.array(paths)

    savepath = os.path.join(args.savepath, f'sample-{i}')
    images, images_debug = diff_trainer.renderer[0].composite_full_image(savepath, paths, dim=(1024, 1280), resize=(512, 640), ext='.png', use_ema=False)

    save_video(os.path.join(args.savepath, f'sample-{i}.mp4'), images, fps=10)
    save_video(os.path.join(args.savepath, f'sample-{i}_debug.mp4'), images_debug, fps=10)

    save_numpy_array_as_gif(images_debug, os.path.join(args.savepath, f'sample-{i}_debug.gif'), duration=150)





