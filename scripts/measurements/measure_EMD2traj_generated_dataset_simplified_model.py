import pdb

import diffuser.sampling as sampling
import diffuser.utils as utils
from diffuser.environments.sperm import *
from diffuser.utils.arrays import batch_to_device, to_np, to_device, apply_dict
import einops
import numpy as np
import torch
from diffuser.utils.training import cycle
import os
from diffuser.utils.video import save_video
from diffuser.models.helpers import apply_conditioning
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.special import rel_entr
from scipy.stats import wasserstein_distance
from frechetdist import frdist
from functools import cmp_to_key


colors = {
    'blue':    '#377eb8',
    'orange':  '#ff7f00',
    'green':   '#4daf4a',
    'pink':    '#f781bf',
    'brown':   '#a65628',
    'purple':  '#984ea3',
    'gray':    '#999999',
    'red':     '#e41a1c',
    'yellow':  '#dede00'
}

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=list(colors.values()))


def rotate2Dvec(v, theta):
    c, s = np.cos(theta), np.sin(theta)
    r = np.array(((c, -s), (s, c)))
    v = np.dot(r, v)
    return v

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'SingleSpermBezierIncrementsDataAugSimplified-v0'
    config: str = 'config.sperm'

args = Parser().parse_args('diffusion')

cycle_color = False
save_figures = True
stopped_samples = False
compare_train_test = False
compare_gauss = True
unnormalize_params = 70.
unnormalize_displ = 20.
n_condition_frames = 0

data_file = 'diffuser/datasets/BezierSplinesData/stopped'
synth_data_file = 'diffuser/datasets/synthdata_GaussModel_20240512_stopped'

#-----------------------------------------------------------------------------#
#------------------------------loading real data------------------------------#
#-----------------------------------------------------------------------------#

env = SingleSpermBezierIncrementsDataAug(data_file=data_file)

trainset = env.get_dataset_trajectories()
testset = env.get_dataset_trajectories()
trainset = trainset['observations']
testset = testset['observations']
#-----------------------------------------------------------------------------#
#------------------------------loading synth data-----------------------------#
#-----------------------------------------------------------------------------#


if compare_train_test:
    synthset = trainset
    save_figures = False
    synth_data_file = data_file

else:
    env2 = SingleSpermBezierIncrementsSynth(data_file=synth_data_file)

    synthset = env2.get_dataset_trajectories()
    synthset = synthset['observations']

if compare_train_test or compare_gauss:
    synthset_aux = synthset.copy()
    for _ in range(len(testset)-1):
        synthset.extend(synthset_aux)

min_shape = np.min([te.shape[0] for te in testset])
synthset = [sy[:min_shape] for sy in synthset]
testset = [te[:min_shape] for te in testset]

synth_params = [sy[:, 40:50] * unnormalize_params for sy in synthset]
test_params = [te[:, 40:50] * unnormalize_params for te in testset]

synth_displacement = [sy[1:, -5:-3] * unnormalize_displ for sy in synthset]
test_displacement = [te[1:, -5:-3] * unnormalize_displ for te in testset]

for i in range(len(synth_displacement)):
    angle = np.radians(vec2angle(synth_displacement[i][0], normalize=False))
    for j in range(synth_displacement[i].shape[0]):
        synth_displacement[i][j] = rotate2Dvec(np.array(synth_displacement[i][j]), -angle)
for i in range(len(test_displacement)):
    angle = np.radians(vec2angle(test_displacement[i][0], normalize=False))
    for j in range(test_displacement[i].shape[0]):
        test_displacement[i][j] = rotate2Dvec(np.array(test_displacement[i][j]), -angle)


synth_alpha = [sy[:, -3] for sy in synthset]
test_alpha = [te[:, -3] for te in testset]

synth_head_coord = [sy[:, -2:] * unnormalize_displ for sy in synthset]
test_head_coord = [te[:, -2:] * unnormalize_displ for te in testset]

synth2measure = [sy[:, 40:-2] for sy in synthset]
test2measure = [te[:, 40:-2] for te in testset]

for i in range(len(synth2measure)):
    synth2measure[i][:, 40:50] = synth2measure[i][:, 40:50] * unnormalize_params
    synth2measure[i][:, -5:-3] = synth2measure[i][:, -5:-3] * unnormalize_displ

for i in range(len(test2measure)):
    test2measure[i][:, 40:50] = test2measure[i][:, 40:50] * unnormalize_params
    test2measure[i][:, -5:-3] = test2measure[i][:, -5:-3] * unnormalize_displ

def average_displacement_error(dataset1, dataset2, suma=True):
    """
    Calculate the Average Displacement Error (ADE) between two datasets of curves.

    Args:
    - dataset1: List of numpy arrays representing curves in dataset 1
    - dataset2: List of numpy arrays representing curves in dataset 2

    Returns:
    - ade: Average Displacement Error
    """
    num_curves = min(len(dataset1), len(dataset2))
    total_distance = []

    for i in range(num_curves):
        curve1 = dataset1[i]
        curve2 = dataset2[i]

        # Ensure curves have the same number of points
        min_length = min(len(curve1), len(curve2))
        curve1 = curve1[:min_length]
        curve2 = curve2[:min_length]

        # Compute the Euclidean distance between corresponding points
        distances = np.linalg.norm(curve1 - curve2, axis=1)
        if suma:
            total_distance.append(np.sum(distances))
        else:
            total_distance.append(np.mean(distances))

    # Calculate the average displacement error
    ade = np.mean(total_distance)
    return ade


def average_displacement_error_multiples_samples(dataset1, dataset2,n_condition_frames, n_samples=10 , suma=False):
    """
    Calculate the Average Displacement Error (ADE) between two datasets of curves.

    Args:
    - dataset1: List of numpy arrays representing curves in dataset 1
    - dataset2: List of numpy arrays representing curves in dataset 2

    Returns:
    - ade: Average Displacement Error
    https://ieeexplore.ieee.org/document/10213104
    \begin{equation*}\text{ADE}=\frac{1}{\mathrm{n}} \sum_{\mathrm{i}=1}^{\mathrm{n}}\frac{1}{\mathrm{t}_{\text {pred }}} \sum_{\mathrm{t}=\mathrm{t}_{\text {obs }}+1}^{\mathrm{t}_{\text {obs }}+\mathrm{t}_{\text {prod }}}\sqrt{\begin{matrix} (\mathrm{x}_{\mathrm{i}}^{\mathrm{t}}-\hat{\mathrm{x}}_{\mathrm{i}}^{\mathrm{t}})^{2}+ \\ (\mathrm{y}_{\mathrm{i}}^{\mathrm{t}}-\hat{\mathrm{y}}_{\mathrm{i}}^{\mathrm{t}})^{2}+ \\ (\mathrm{z}_{\mathrm{i}}^{\mathrm{t}}-\hat{\mathrm{z}}_{\mathrm{i}}^{\mathrm{t}})^{2}\end{matrix}}\tag{12}\end{equation*}
    """

    total_distance = []

    for i, curve1 in enumerate(dataset2):
        # Create a list to store distances for each synthetic example
        distances_mean = []
        curve1 = curve1[n_condition_frames:]
        for j in range(n_samples):
            # Get the corresponding synthetic example from dataset1
            curve2 = dataset1[n_samples * i + j][n_condition_frames:]

            # Ensure curves have the same number of points
            min_length = min(len(curve1), len(curve2))
            curve1 = curve1[:min_length]
            curve2 = curve2[:min_length]

            # Compute the Euclidean distance between corresponding points
            distances = np.linalg.norm(curve1 - curve2, axis=1)  # distances = np.sqrt(np.sum(np.square(curve1 - curve2), axis=1))

            if suma:
                distances_mean.append(np.sum(distances))
            else:
                distances_mean.append(np.mean(distances))

        # Average the distances for all synthetic examples
        average_distance = np.mean(distances_mean)

        # Add the average distance to the total distance
        total_distance.append(average_distance)

    # Calculate the average displacement error
    ade = np.mean(total_distance)
    return ade



def min_average_displacement_error(dataset1, dataset2, n_condition_frames, n_samples=10,  suma=False):
    """
    Calculate the minimum Average Displacement Error (minADE) between two datasets of curves.

    Args:
    - dataset1: List of numpy arrays representing curves in dataset 1
    - dataset2: List of numpy arrays representing curves in dataset 2

    Returns:
    - min_ade: Minimum Average Displacement Error
    https://openaccess.thecvf.com/content/ICCV2023/papers/Weng_Joint_Metrics_Matter_A_Better_Standard_for_Trajectory_Forecasting_ICCV_2023_paper.pdf
    Este paper usa la suma de distances en lugar de la media
    """
    average_distance = []
    for i, curve1 in enumerate(dataset2):
        # Create a list to store distances for each synthetic example
        distances_mean = []
        curve1 = curve1[n_condition_frames:]
        for j in range(n_samples):
            curve2 = dataset1[n_samples * i + j][n_condition_frames:]

            # Ensure curves have the same number of points
            min_length = min(len(curve1), len(curve2))
            curve1 = curve1[:min_length]
            curve2 = curve2[:min_length]

            # Compute the Euclidean distance between corresponding points
            distances = np.linalg.norm(curve1 - curve2, axis=1)  # distances = np.sqrt(np.sum(np.square(curve1 - curve2), axis=1))

            if suma:
                distances_mean.append(np.sum(distances))
            else:
                distances_mean.append(np.mean(distances))

        # Average the distances for all synthetic examples
        average_distance.append(np.min(distances_mean))

    # Update min_ade if the average_distance is smaller
    made = np.mean(average_distance)
    return made

def wasserstain_distance_multi(dataset1, dataset2, n_condition_frames, n_samples=10):
    """
    Compare each instance from dataset 1 with the n_samples corresponding instances in dataset 2.

    Args:
    - dataset1: List of numpy arrays representing instances in dataset 1
    - dataset2: List of numpy arrays representing instances in dataset 2

    Returns:
    - wasserstein_distances: List of Wasserstein distances between each instance in dataset1 and its 10 corresponding instances in dataset2
    """
    wasserstein_distances = []

    for instance1 in dataset2:
        instance_distances = []
        instance1 = instance1[n_condition_frames:]
        for i in range(0, len(dataset1), n_samples):
            # Take 10 instances from dataset2 corresponding to the current instance1
            instances2 = dataset1[i:i+n_samples][:, n_condition_frames:]

            # Compute the Wasserstein distance between instance1 and each instance2
            instance_distances.extend([wasserstein_distance(instance1.flatten(), instance2.flatten()) for instance2 in instances2])

        # Append the minimum Wasserstein distance to the list of distances
        wasserstein_distances.append(min(instance_distances))

    return np.mean(wasserstein_distances)

def final_displacement_error(dataset1, dataset2):
    """
    Calculate the Final Displacement Error (FDE) between two datasets of trajectories.

    Args:
    - dataset1: List of numpy arrays representing trajectories in dataset 1
    - dataset2: List of numpy arrays representing trajectories in dataset 2

    Returns:
    - fde: Final Displacement Error
    """
    num_trajectories = min(len(dataset1), len(dataset2))
    total_distance = []

    for i in range(num_trajectories):
        trajectory1 = dataset1[i]
        trajectory2 = dataset2[i]

        # Ensure trajectories have at least one point
        if len(trajectory1) > 0 and len(trajectory2) > 0:
            # Extract the final points of the trajectories
            final_point1 = trajectory1[-1]
            final_point2 = trajectory2[-1]

            # Compute the Euclidean distance between the final points
            distance = np.linalg.norm(final_point1 - final_point2)

            total_distance.append(distance)

    # Calculate the average final displacement error
    fde = np.mean(total_distance)
    return fde

synth_params_keys = [np.sum(synth_params[i][0]) for i in range(len(synth_params))]
synth_param_index = np.argsort(synth_params_keys)
synth_params = list(np.asarray(synth_params)[synth_param_index])

test_params_keys = [np.sum(test_params[i][0]) for i in range(len(test_params))]
test_param_index = np.argsort(test_params_keys)
test_params = list(np.asarray(test_params)[test_param_index])

count = 0

cadena = ""
for i in range(0, 9, 2):

    if len(synth_params) == len(test_params):
        ADE_params = average_displacement_error(np.array(synth_params)[:, :, i:i + 2],
                                                np.array(test_params)[:, :, i:i + 2])
        dist_params = wasserstein_distance(np.concatenate(synth_params)[:, i:i + 2].flatten(),
                                           np.concatenate(test_params)[:, i:i + 2].flatten())

    else:
        n_samples = len(synth_params)//len(test_params)
        ADE_params = average_displacement_error_multiples_samples(np.array(synth_params)[:, :, i:i + 2], np.array(test_params)[:, :, i:i + 2], n_condition_frames, n_samples)
        # dist_params = wasserstain_distance_multi(np.array(synth_params)[:, i:i + 2], np.array(test_params)[:, i:i + 2], n_condition_frames, n_samples)
        dist_params = wasserstain_distance_multi(np.array(synth_params)[:, :, i:i + 2], np.array(test_params)[:, :, i:i + 2], n_condition_frames, n_samples)
        minADE_params = min_average_displacement_error(np.array(synth_params)[:, :, i:i + 2],
                                                np.array(test_params)[:, :, i:i + 2], n_condition_frames, n_samples)
    print('minADE(synthP_' + str(4 - count) + ', realP-' + str(4 - count) + '): ', minADE_params)


    print('W(synthP_' + str(4-count) + ', realP_' + str(4-count) +'): ' + str(dist_params))
    print('ADE(synthP_' + str(4-count) + ', realP-' + str(4-count) +'): ', ADE_params)

    cadena += str(ADE_params) + "\t" + str(minADE_params) + "\t" + str(dist_params) + "\t"
    count += 1

    # dist_disp = wasserstein_distance(np.concatenate(synth_displacement).flatten(), np.concatenate(test_displacement).flatten())
    # ADE_disp = average_displacement_error(synth_displacement, test_displacement)
    # print('W(synthD, realD): ',  dist_disp)
    # print('ADE(synthD, realD): ', ADE_disp, '\n')
    #
    # dist_alpha = wasserstein_distance(np.concatenate(synth_alpha).flatten(), np.concatenate(test_alpha).flatten())
    # print('W(synthAlpha, realAlpha): ',  dist_alpha, '\n')
    #
    # dist_compl = wasserstein_distance(np.concatenate(synth2measure).flatten(), np.concatenate(test2measure).flatten())
    # ADE_compl = average_displacement_error(synth2measure, test2measure)
    # print('W(synth, real): ', dist_compl)
    # print('ADE(synth, real): ', ADE_compl, '\n')
    #
    # dist_head = wasserstein_distance(np.concatenate(synth_head_coord).flatten(), np.concatenate(test_head_coord).flatten())
    # ADE_head = average_displacement_error(synth_head_coord, test_head_coord)
    # FDE_head = final_displacement_error(synth_head_coord, test_head_coord)
    #
    # print('W(synthHead, realHead): ' + str(dist_head))
    # print('ADE(synthHead, realHead): ', ADE_head)
    # print('FDE(synthHead, realHead): ', FDE_head, '\n\n')

print(cadena)
synth_coord = []
test_coord = []
for i in range(len(synth_displacement)):
    synth_coord.append(np.array([[0., 0.] for _ in range(synth_displacement[0].shape[0])]))
    for j in range(1, synth_displacement[0].shape[0]):
        synth_coord[i][j] = synth_coord[i][j-1] + synth_displacement[i][j]
for i in range(len(test_displacement)):
    test_coord.append(np.array([[0., 0.] for _ in range(test_displacement[0].shape[0])]))
    for j in range(1, test_displacement[0].shape[0]):
        test_coord[i][j] = test_coord[i][j-1] + test_displacement[i][j]

synth_coord = list(np.asarray(synth_coord)[synth_param_index])
test_coord = list(np.asarray(test_coord)[test_param_index])

if len(synth_coord) == len(test_coord):
    dist_params = wasserstein_distance(np.concatenate(synth_coord)[:, i:i + 2].flatten(),
                                       np.concatenate(test_coord)[:, i:i + 2].flatten())
    ADE_params = average_displacement_error(np.array(synth_coord)[:, :, i:i + 2],
                                            np.array(test_coord)[:, :, i:i + 2])
else:
    n_samples = len(synth_coord) // len(test_params)
    ADE_params = average_displacement_error_multiples_samples(np.array(synth_coord), np.array(test_coord),n_condition_frames,  n_samples)
    dist_params = wasserstain_distance_multi(np.array(synth_coord), np.array(test_coord), n_condition_frames, n_samples)
    minADE_params = min_average_displacement_error(np.array(synth_coord), np.array(test_coord), n_condition_frames, n_samples)
    print('minADE(synth_traj, real_traj): ', minADE_params)

print('W(synth_traj, real_traj): ' + str(dist_params))
print('ADE(synth_traj, real_traj): ', ADE_params)

print(str(ADE_params) + "\t" + str(minADE_params) + "\t" + str(dist_params))
for k in range(len(test_coord)):
    plt.figure()
    fig, ax = plt.subplots(figsize = (12, 6))

    if cycle_color:
        color = list(colors.values())[k]
    else:
        color = colors['blue']
    indx = len(synth_coord)//len(test_coord) * k
    for i in range((len(synth_coord)//len(test_coord))):
        if i == 0:
            plt.plot(synth_coord[indx+i][:, 0], synth_coord[indx+i][:, 1], linewidth=3, c=color, label='Synthetic', alpha=0.5, antialiased=True)
        else:
            plt.plot(synth_coord[indx + i][:, 0], synth_coord[indx + i][:, 1], linewidth=3,  c=color, alpha=0.5, antialiased=True)

    if cycle_color:
        color = list(colors.values())[(k+3)%len(colors.values())]
    else:
        color = colors['brown']
    plt.plot(test_coord[k][:, 0], test_coord[k][:, 1], linewidth=3, c=color, linestyle='--', label='Real', antialiased=True)
    # plt.legend(fontsize=40)
    # plt.xlabel('x')
    # plt.ylabel('y')

    if stopped_samples:
        plt.xlim(-0.5 * unnormalize_displ/2, 0.5 * unnormalize_displ)
        plt.ylim(-0.5 * unnormalize_displ, 0.5 * unnormalize_displ)
        plt.xticks([-0.5 * unnormalize_displ/2, 0 * unnormalize_displ, 0.5 * unnormalize_displ], fontsize=40)
        plt.yticks([-0.5 * unnormalize_displ, 0 * unnormalize_displ, 0.5 * unnormalize_displ], fontsize=40)
    else:
        plt.xlim(-0.5 * unnormalize_displ, 5. * unnormalize_displ)
        plt.ylim(-4. * unnormalize_displ, 4. * unnormalize_displ)
        plt.xticks([0, 2 * unnormalize_displ, 4 * unnormalize_displ], fontsize=40)
        plt.yticks([-4 * unnormalize_displ, -2 * unnormalize_displ, 0 * unnormalize_displ, 2 * unnormalize_displ, 4 * unnormalize_displ], fontsize=40)
    fig.tight_layout()

    if save_figures:
        plt.savefig(os.path.join(synth_data_file, 'synth_traj_' + str(k) + '.png'))

    # plt.figure()
    # fig, ax = plt.subplots(figsize = (12, 8))
    # plt.plot(test_coord[k][:, 0], test_coord[k][:, 1], linewidth=3, c=color)
    # plt.legend()
    # fig.tight_layout()
    # plt.xlabel('Real trajectories')
    # plt.xlim(-1., 5.)
    # plt.ylim(-1., 5.)
    #
    # if save_figures:
    #     plt.savefig(os.path.join(synth_data_file, 'real_traj_' + str(k) + '.png'))

    plt.figure()
    fig, ax = plt.subplots(figsize = (12, 6))

    plt.plot(test_coord[k][:, 0], test_coord[k][:, 1], linewidth=3, c=color, linestyle='--', label='Real', antialiased=True)

    # plt.legend(fontsize=40)
    # plt.xlabel('Gaussian trajectories')
    if stopped_samples:
        plt.xlim(-0.5 * unnormalize_displ/2, 0.5 * unnormalize_displ)
        plt.ylim(-0.5 * unnormalize_displ, 0.5 * unnormalize_displ)
        plt.xticks([-0.5 * unnormalize_displ/2, 0 * unnormalize_displ, 0.5 * unnormalize_displ], fontsize=40)
        plt.yticks([-0.5 * unnormalize_displ, 0 * unnormalize_displ, 0.5 * unnormalize_displ], fontsize=40)
    else:
        plt.xlim(-0.5 * unnormalize_displ, 5. * unnormalize_displ)
        plt.ylim(-4. * unnormalize_displ, 4. * unnormalize_displ)
        plt.xticks([0, 2 * unnormalize_displ, 4 * unnormalize_displ], fontsize=40)
        plt.yticks([-4 * unnormalize_displ, -2 * unnormalize_displ, 0 * unnormalize_displ, 2 * unnormalize_displ, 4 * unnormalize_displ], fontsize=40)
    fig.tight_layout()

plt.figure()
fig, ax = plt.subplots(figsize = (12, 6))

for i in range(len(test_coord)):
    if cycle_color:
        color = list(colors.values())[i]
    else:
        color = colors['blue']
    plt.plot(test_coord[i][:, 0], test_coord[i][:, 1], linewidth=3, c=color, label='Traj '+str(i), antialiased=True)
# plt.legend(fontsize=40)
# plt.xlabel('Real trajectories')

plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.grid(True)
#plt.xticks([0, 2, 4], fontsize=40)
#plt.yticks([-4, -2, 0, 2, 4], fontsize=40)
if stopped_samples:
    #plt.xlim(-0.5, 0.5)
    #plt.ylim(-0.5, 0.5)
    #plt.xticks([-0.5, 0, 0.5], fontsize=40)
    #plt.yticks([-0.5, 0, 0.5], fontsize=40)
    plt.xlim(-0.5 * unnormalize_displ/2, 0.5 * unnormalize_displ)
    plt.ylim(-0.5 * unnormalize_displ, 0.5 * unnormalize_displ)
    plt.xticks([-0.5 * unnormalize_displ/2, 0 * unnormalize_displ, 0.5 * unnormalize_displ], fontsize=40)
    plt.yticks([-0.5 * unnormalize_displ, 0 * unnormalize_displ, 0.5 * unnormalize_displ], fontsize=40)
else:
    # plt.xlim(-0.5, 5.)
    # plt.ylim(-4., 4.)
    # plt.xticks([0, 2, 4], fontsize=40)
    # plt.yticks([-4, -2, 0, 2, 4], fontsize=40)
    plt.xlim(-0.5 * unnormalize_displ, 5. * unnormalize_displ)
    plt.ylim(-4. * unnormalize_displ, 4. * unnormalize_displ)
    plt.xticks([0, 2 * unnormalize_displ, 4 * unnormalize_displ], fontsize=40)
    plt.yticks([-4 * unnormalize_displ, -2 * unnormalize_displ, 0 * unnormalize_displ, 2 * unnormalize_displ,
                4 * unnormalize_displ], fontsize=40)

fig.tight_layout()

if save_figures:
    plt.savefig(os.path.join(synth_data_file, 'all_real_traj' + '.png'))