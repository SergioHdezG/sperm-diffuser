import json
import random
from json import JSONEncoder
from typing import List, Any

import joblib

from diffuser.environments.sperm import SingleSpermBezierIncrementsDataAugSimplified
import numpy as np
import torch
import os
from PIL import Image

from diffuser.environments.utils.Bezier import Bezier
from diffuser.environments.utils.sperm_rendering import vec2angle
from scripts.baseline.timeSeriesLSTMsperm import LSTMModel

from sklearn.preprocessing import MinMaxScaler
import joblib


# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#

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
n_sequences = 6
mean_displacement = 23
std_displacement = 6
disp_min = 16
disp_max = 60
seq_len = 28
hist_info = 4
n_copies = 10
savebase = 'scripts/baseline/moving/lstm4h_real_condition'
data_file = 'diffuser/datasets/BezierSplinesData/moving'

make_subfolders = False
cond_train = True
#visualization_only = False

# -----------------------------------------------------------------------------#
# ---------------------------------- loading ----------------------------------#
# -----------------------------------------------------------------------------#

diffusion_loadpath = 'scripts/baseline/moving/lstm4h_real_condition/'
env = SingleSpermBezierIncrementsDataAugSimplified(data_file=data_file)

# dataset_train = env.get_dataset()
# dataset_test = env.get_dataset()
# dataset_train = dataset_train['observations']
# dataset_test = dataset_test['observations']

model = LSTMModel()
model.load_state_dict(torch.load(os.path.join(diffusion_loadpath, "best.pt")))
scaler = joblib.load(os.path.join(diffusion_loadpath, "scaler.save"))

# -----------------------------------------------------------------------------#
# --------------------------------- main loop ---------------------------------#
# -----------------------------------------------------------------------------#

gauss_means = [-0.7101236, 0.012660508, -0.4724761, -0.011081022, -0.30175892, 0.013444074, -0.111315355, 0.03597443,
               0.11064559, -0.012201412]
gauss_std = [0.18651162, 0.060374398, 0.16854781, 0.1470919, 0.12989329, 0.19624966, 0.15786962, 0.10110918,
             0.053684674,
             0.03615527]
gauss_displ_mean = [3.637117, -0.00015417, 0.8920469, -0.0450125]
gauss_displ_std = [1.312792, 0.000920, 0.33587, 0.29901]


def sample_param(index, batch):
    return np.random.normal(gauss_means[index], gauss_std[index], batch)


def sample_displacement(batch):
    return np.transpose([np.random.normal(gauss_displ_mean[0], gauss_displ_std[0], batch),
                         np.random.normal(gauss_displ_mean[1], gauss_displ_std[1], batch)])


# def sample_new_cond(shape, batch_size):
#     sequences = np.zeros(shape)
#     for k in range(len(gauss_means)):
#         sequences[:, k] = sample_param(k, batch_size)
#
#     sequences[:, -4] = np.random.normal(gauss_displ_mean[0], gauss_displ_std[0], n_sperm_per_seq[i])
#     sequences[:, -3] = np.random.normal(gauss_displ_mean[1], gauss_displ_std[1], n_sperm_per_seq[i])
#     sequences[:, -2] = np.random.normal(gauss_displ_mean[2], gauss_displ_std[2], n_sperm_per_seq[i])
#     sequences[:, -1] = np.random.normal(gauss_displ_mean[3], gauss_displ_std[3], n_sperm_per_seq[i])
#
#     return sequences


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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trainset = env.get_dataset_trajectories()
testset = env.get_dataset_trajectories()

if cond_train:
    inference_set = trainset
else:
    inference_set = testset
for i in range(len(inference_set['observations'])):
    observations = []
    paths = []

    traj = np.zeros((n_copies, *inference_set['observations'][i].shape))

    for j in range(n_copies):
        traj[j] = inference_set['observations'][i]
    # traj = np.expand_dims(inference_set['observations'][i])
    # traj = torch.from_numpy(np.expand_dims(inference_set['observations'][i], axis=0))

    traj = traj[:, :4, :]
    for k in range(traj.shape[1]):
        traj[:, k] = scaler.transform(traj[:, k])

    #if visualization_only:
        # Initialize a list to store the forecasted values
    trajectories = [traj[:, k] for k in range(hist_info)]
    #else:
    #    trajectories = [traj[:, -1]]

    norm_traj = torch.from_numpy(traj)



    # Use the last 30 data points as the starting point
    historical_data = norm_traj

    with torch.no_grad():
        for _ in range(seq_len):
            # Prepare the historical_data tensor
            historical_data_tensor = torch.as_tensor(historical_data).view(n_copies, hist_info, 14).float().to(device)
            # Use the model to predict the next value
            predicted_value = model(historical_data_tensor)
            predicted_value = predicted_value.cpu().numpy()[:, 0]
            # Append the predicted value to the forecasted_values list
            trajectories.append(scaler.inverse_transform(predicted_value))

            # Update the historical_data sequence by removing the oldest value and adding the predicted value
            historical_data = np.roll(historical_data, shift=-1, axis=1)
            historical_data[:, -1] = predicted_value


        trajectories = np.transpose(trajectories, (1, 0, 2))

        observations.append(trajectories)

    observations = np.concatenate(observations, axis=1)
    for k in range(len(observations)):
        if make_subfolders:
            path_seq = os.path.join(savebase, diffusion_loadpath.split('/')[-1], f'field_{i}',
                                    'json_bezier_spline', f'field_{i}_{k}')
        else:
            path_seq = os.path.join(savebase, diffusion_loadpath.split('/')[-1], f'field_{i}_{k}')

        os.makedirs(path_seq, exist_ok=True)

        state = np.array(observations[k])

        first_iter = True
        init_head = (np.random.rand(2) * 2) - 1
        aux_head = init_head
        aux_velocity = [0., 0.]
        traj = []
        rand_rotation = np.random.rand() * 2 * np.pi

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
            aux_y = img_size[0] - ((aux_head[1] + 1.) / 2.) * img_size[0]

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
                angle = (correction_angle - 360) / 180
            else:
                angle = correction_angle / 180

            obs = np.concatenate(
                [np.reshape(norm_curve, 40), state[kk, :10], [norm_vel_x, norm_vel_y, angle, norm_x, norm_y]])

            aux_head[0] = aux_head[0] + ((rot_velocity[0] / img_size[1]) * 2)
            aux_head[1] = aux_head[1] + ((rot_velocity[1] / img_size[0]) * 2)
            aux_velocity[0] = rot_velocity[0]
            aux_velocity[1] = rot_velocity[1]

            traj.append(obs)

            save2json(spline_params, os.path.join(path_seq, str(kk).zfill(3) + '.json'))

        if state.shape[-1] == 14 or state.shape[-1] == 12:
            paths.append(traj)







