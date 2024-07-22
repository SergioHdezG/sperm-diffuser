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


#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

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

class Parser(utils.Parser):
    dataset: str = 'SingleSpermBezierIncrementsDataAugSimplified-v0'
    config: str = 'config.sperm'

args = Parser().parse_args('diffusion')

save_figures = True
data_file = 'diffuser/datasets/BezierSplinesData/progressive'
synth_data_file = 'diffuser/datasets/synthdata_progressive_sperm/progressive_data'

figures_path = 'diffuser/datasets/synthdata_progressive_sperm'

#-----------------------------------------------------------------------------#
#------------------------------loading real data------------------------------#
#-----------------------------------------------------------------------------#


th = 0.00  # Filtering lower values on the displacement modulus resulting of first state of a trajectory.
# These are 0 because there is no way to  measure displascement on the first frame. Use th = 0.08 for progressive and slow
# progressive sperm cells and th = -0.01 for inmotile sperm.

# Calculate KL divergence
def kl_mvn(mean_p, cov_p, mean_q, cov_q):
    kl_divergence = 0.5 * (
        np.log(np.linalg.det(cov_q) / np.linalg.det(cov_p))
        - len(mean_p)
        + np.trace(np.linalg.inv(cov_q) @ cov_p)
        + (mean_q - mean_p) @ np.linalg.inv(cov_q) @ (mean_q - mean_p)
    )
    return kl_divergence

def kl_norm(mean_p, cov_p, mean_q, cov_q):
    kl_divergence = 0.5 * (
        np.log(cov_q / cov_p)  # Note: No covariance matrix in univariate case
        + (cov_p**2 + (mean_p - mean_q)**2) / (2 * cov_q**2) - 0.5
    )
    return kl_divergence

def calculate_kl_mvn(full_data_params, trainset_params, testset_params, synhtset_params, synht_trainset_params, synht_testset_params):
    for p in range(5):
        synth_param_test = synht_testset_params[:, p].transpose(1, 0)
        synth_param_train = synht_trainset_params[:, p].transpose(1, 0)
        synth_param_data = synhtset_params[:, p].transpose(1, 0)

        synth_test_mean = np.mean(synth_param_test, axis=1)
        synth_train_mean = np.mean(synth_param_train, axis=1)
        synth_data_mean = np.mean(synth_param_data, axis=1)

        synth_test_cov = np.cov(synth_param_test)
        synth_train_cov = np.cov(synth_param_train)
        synth_data_cov = np.cov(synth_param_data)

        param_test = testset_params[:, p].transpose(1, 0)
        param_train = trainset_params[:, p].transpose(1, 0)
        param_data = full_data_params[:, p].transpose(1, 0)

        test_mean = np.mean(param_test, axis=1)
        train_mean = np.mean(param_train, axis=1)
        data_mean = np.mean(param_data, axis=1)

        test_cov = np.cov(param_test)
        train_cov = np.cov(param_train)
        data_cov = np.cov(param_data)

        # test_dist_p = multivariate_normal(mean=test_mean, cov=test_cov)
        # synth_test_dist_q = multivariate_normal(mean=synth_test_mean, cov=synth_test_cov)
        #
        # train_dist_p = multivariate_normal(mean=train_mean, cov=train_cov)
        # synth_train_dist_q = multivariate_normal(mean=synth_train_mean, cov=synth_train_cov)
        #
        data_dist_p = multivariate_normal(mean=data_mean, cov=data_cov)
        synth_data_dist_q = multivariate_normal(mean=synth_data_mean, cov=synth_data_cov)
        # https://stackoverflow.com/questions/76761210/kl-and-js-divergence-analysis-of-pdfs-of-numbers


        test_kl_divergence = kl_mvn(synth_test_mean, synth_test_cov, test_mean, test_cov)  # np.sum(rel_entr(test_dist_p.pdf(pos), synth_test_dist_q.pdf(pos)))
        test2train_kl_divergence = kl_mvn(synth_test_mean, synth_test_cov, train_mean, train_cov)  #  np.sum(rel_entr(train_dist_p.pdf(pos), synth_test_dist_q.pdf(pos)))
        test2data_kl_divergence = kl_mvn(synth_test_mean, synth_test_cov, data_mean, data_cov)  # np.sum(rel_entr(data_dist_p.pdf(pos), synth_test_dist_q.pdf(pos)))

        train2train_kl_divergence = kl_mvn(synth_train_mean, synth_train_cov, train_mean, train_cov)  # np.sum(rel_entr(train_dist_p.pdf(pos), synth_train_dist_q.pdf(pos)))
        datat2data_kl_divergence = kl_mvn(synth_data_mean, synth_data_cov, data_mean, data_cov)  #np.sum(rel_entr(data_dist_p.pdf(pos), synth_data_dist_q.pdf(pos)))

        realtrain2realtest_kl_divergence = kl_mvn(train_mean, train_cov, test_mean, test_cov)

        print('P(' + str(4-p) + ')=> KL(test, test) KL(test, train) KL(test, data) KL(train, train) KL(data, data) KL(real_train, real_test)')
        print('\t' + str(test_kl_divergence) + '\t' + str(test2train_kl_divergence) + '\t' + str(test2data_kl_divergence)
              + '\t' + str(train2train_kl_divergence) + '\t' + str(datat2data_kl_divergence) + '\t' + str(realtrain2realtest_kl_divergence) + '\n\n')

        print('P(' + str(
            4-p) + ')=> mean(synth_test)  cov(synth_test) mean(synth_train)  cov(synth_train) mean(synth_data)  '
                 'cov(synth_data)')
        print('\t' + str(synth_test_mean) + '\t' + str(synth_test_cov) + '\t' + str(synth_train_mean) + '\t' + str(synth_train_cov) + '\t' +
              str(synth_data_mean) + '\t' + str(synth_data_cov))
        print('P(' + str(
            4-p) + ')=> mean(test)  cov(test) mean(train)  cov(train) mean(data)  cov(data)')
        print('\t' + str(test_mean) + '\t' + str(test_cov) + '\t' + str(train_mean) + '\t' + str(train_cov) + '\t' +
              str(data_mean) + '\t' + str(data_cov) + '\n\n\n\n')

        # x, y = np.mgrid[-1:1:.01, -1:1:.01]
        # pos = np.dstack((x, y))
        # fig1 = plt.figure()
        # ax1 = fig1.add_subplot(111)
        # ax1.contourf(x, y, synth_data_dist_q.pdf(pos))
        #
        # ax2 = fig1.add_subplot(112)
        # ax2.contourf(x, y, data_dist_p.pdf(pos))
        #
        # plt.show()

    synht_testset_params = np.concatenate([synht_testset_params[:, :, 0], synht_testset_params[:, :, 1]], axis=-1)
    synht_trainset_params = np.concatenate([synht_trainset_params[:, :, 0], synht_trainset_params[:, :, 1]], axis=-1)
    synhtset_params = np.concatenate([synhtset_params[:, :, 0], synhtset_params[:, :, 1]], axis=-1)
    testset_params = np.concatenate([testset_params[:, :, 0], testset_params[:, :, 1]], axis=-1)
    trainset_params = np.concatenate([trainset_params[:, :, 0], trainset_params[:, :, 1]], axis=-1)
    full_data_params = np.concatenate([full_data_params[:, :, 0], full_data_params[:, :, 1]], axis=-1)

    synth_test_mean = np.mean(synht_testset_params.transpose(1, 0), axis=1)
    synth_train_mean = np.mean(synht_trainset_params.transpose(1, 0), axis=1)
    synth_data_mean = np.mean(synhtset_params.transpose(1, 0), axis=1)

    synth_test_cov = np.cov(synht_testset_params.transpose(1, 0))
    synth_train_cov = np.cov(synht_trainset_params.transpose(1, 0))
    synth_data_cov = np.cov(synhtset_params.transpose(1, 0))

    test_mean = np.mean(testset_params.transpose(1, 0), axis=1)
    train_mean = np.mean(trainset_params.transpose(1, 0), axis=1)
    data_mean = np.mean(full_data_params.transpose(1, 0), axis=1)

    test_cov = np.cov(testset_params.transpose(1, 0))
    train_cov = np.cov(trainset_params.transpose(1, 0))
    data_cov = np.cov(full_data_params.transpose(1, 0))

    test_kl_divergence = kl_mvn(synth_test_mean, synth_test_cov, test_mean, test_cov)
    test2train_kl_divergence = kl_mvn(synth_test_mean, synth_test_cov, train_mean, train_cov)
    test2data_kl_divergence = kl_mvn(synth_test_mean, synth_test_cov, data_mean ,data_cov)

    train2train_kl_divergence = kl_mvn(synth_train_mean, synth_train_cov, train_mean, train_cov)
    datat2data_kl_divergence = kl_mvn(synth_data_mean, synth_data_cov, data_mean, data_cov)

    realtrain2realtest_kl_divergence = kl_mvn(train_mean, train_cov, test_mean, test_cov)

    print('P=> KL(test, test) KL(test, train) KL(test, data) KL(train, train) KL(data, data) KL(real_train, real_test)')
    print('\t' + str(test_kl_divergence) + '\t' + str(test2train_kl_divergence) + '\t' + str(test2data_kl_divergence)
          + '\t' + str(train2train_kl_divergence) + '\t' + str(datat2data_kl_divergence) + '\t' + str(realtrain2realtest_kl_divergence) +'\n\n\n\n')

env = SingleSpermBezierIncrementsDataAug(data_file=data_file)

trainset = env.get_dataset()
testset = env.get_dataset()
trainset = trainset['observations']
testset = testset['observations']
full_data = np.concatenate([trainset, testset], axis=0)

train_median = np.mean(trainset, axis=0)
train_mean = np.mean(trainset, axis=0)
train_std = np.std(trainset, axis=0)
train_max = np.max(trainset, axis=0)
train_min = np.min(trainset, axis=0)
train_rms = np.sqrt(np.mean(trainset**2, axis=0))

test_median = np.mean(testset, axis=0)
test_mean = np.mean(testset, axis=0)
test_std = np.std(testset, axis=0)
test_max = np.max(testset, axis=0)
test_min = np.min(testset, axis=0)
test_rms = np.sqrt(np.mean(testset**2, axis=0))

data_median = np.mean(full_data, axis=0)
data_mean = np.mean(full_data, axis=0)
data_std = np.std(full_data, axis=0)
data_max = np.max(full_data, axis=0)
data_min = np.min(full_data, axis=0)
data_rms = np.sqrt(np.mean(full_data**2, axis=0))

#-----------------------------------------------------------------------------#
#------------------------------loading synth data-----------------------------#
#-----------------------------------------------------------------------------#

env = SingleSpermBezierIncrementsSynth(data_file=synth_data_file)

synth_trainset = env.get_dataset()
synth_testset = env.get_dataset()
synth_trainset = synth_trainset['observations']
synth_testset = synth_testset['observations']

# Printing real data metrics
print('All data => mean: ', data_mean, '\nmedian: ', data_median, '\nstd: ', data_std, '\nmax: ', data_max, '\nmin: ', data_min)
print('Train => mean: ', train_mean, '\nmedian: ', train_median, '\nstd: ', train_std, '\nmax: ', train_max, '\nmin: ', train_min)
print('Test => mean: ', test_mean, '\nmedian: ', test_median, '\nstd: ', test_std, '\nmax: ', test_max, '\nmin: ', test_min)

synthset = np.concatenate([synth_trainset, synth_testset], axis=0)
synth_median = np.mean(synthset, axis=0)
synth_mean = np.mean(synthset, axis=0)
synth_std = np.std(synthset, axis=0)
synth_max = np.max(synthset, axis=0)
synth_min = np.min(synthset, axis=0)
synth_rms = np.sqrt(np.mean(synthset**2, axis=0))

print('Synth => mean: ', synth_mean, '\nmedian: ', synth_median, '\nstd: ', synth_std, '\nmax: ', synth_max, '\nmin: ', synth_min)

testset_params = np.reshape(testset[:, 40:50], (testset.shape[0], 5, 2))
trainset_params = np.reshape(trainset[:, 40:50], (trainset.shape[0], 5, 2))
full_data_params = np.reshape(full_data[:, 40:50], (full_data.shape[0], 5, 2))

synht_testset_params = np.reshape(synth_testset[:, 40:50], (synth_testset.shape[0], 5, 2))
synht_trainset_params = np.reshape(synth_trainset[:, 40:50], (synth_trainset.shape[0], 5, 2))
synhtset_params = np.reshape(synthset[:, 40:50], (synthset.shape[0], 5, 2))


calculate_kl_mvn(full_data_params, trainset_params, testset_params, synhtset_params, synht_trainset_params, synht_testset_params)

def calc_module(data):
    return np.sqrt(np.square(data[0]) + np.square(data[1]))

def print_displacement_metrics(data, module, name):
    print(name + ' disp => mean: [' + str(np.mean(data[0])) + ', ' + str(np.mean(data[1])) +
          '] std: [' + str(np.std(data[0])) + ', ' + str(np.std(data[1])) +
          '] module mean: ' + str(np.mean(module)) + ' module std' + str(np.std(module)))

    return module



testset_displacement = testset[:, -5:-3].transpose(1, 0)
synth_testset_displacement = synth_testset[:, -5:-3].transpose(1, 0)
trainset_displacement = trainset[:, -5:-3].transpose(1, 0)
synth_trainset_displacement = synth_trainset[:, -5:-3].transpose(1, 0)
full_data_displacement = full_data[:, -5:-3].transpose(1, 0)
synthset_displacement = synthset[:, -5:-3].transpose(1, 0)

modu_testset_displacement = calc_module(testset_displacement)
ind = [modu_testset_displacement > th][0]
modu_testset_displacement = modu_testset_displacement[ind]
testset_displacement = testset_displacement[:, ind]  # Filtering
# lower values resulting of first state of a trajectory. These are 0. becouse there is no way to measure displascement on the first frame.
print_displacement_metrics(testset_displacement, modu_testset_displacement, 'test')

modu_synth_testset_displacement = calc_module(synth_testset_displacement)
ind = [modu_synth_testset_displacement > th][0]
modu_synth_testset_displacement = modu_synth_testset_displacement[ind]
synth_testset_displacement = synth_testset_displacement[:, ind]  # Filtering
# lower values resulting of first state of a trajectory. These are 0. becouse there is no way to measure displascement on the first frame.
print_displacement_metrics(synth_testset_displacement, modu_synth_testset_displacement,'synth_test')

modu_trainset_displacement = calc_module(trainset_displacement)
ind = [modu_trainset_displacement > th][0]
modu_trainset_displacement = modu_trainset_displacement[ind]
trainset_displacement = trainset_displacement[:, ind]
print_displacement_metrics(trainset_displacement, modu_trainset_displacement, 'train')

modu_synth_trainset_displacement = calc_module(synth_trainset_displacement)
ind = [modu_synth_trainset_displacement > th][0]
modu_synth_trainset_displacement = modu_synth_trainset_displacement[ind]
synth_trainset_displacement = synth_trainset_displacement[:, ind]
print_displacement_metrics(synth_trainset_displacement, modu_synth_trainset_displacement, 'synth_test')

modu_full_data_displacement = calc_module(full_data_displacement)
ind = [modu_full_data_displacement > th][0]
modu_full_data_displacement = modu_full_data_displacement[ind]
full_data_displacement = full_data_displacement[:, ind]
print_displacement_metrics(full_data_displacement, modu_full_data_displacement, 'dataset')

modu_synthset_displacement = calc_module(synthset_displacement)
ind = [modu_synthset_displacement > th][0]
modu_synthset_displacement = modu_synthset_displacement[ind]
synthset_displacement = synthset_displacement[:, ind]
print_displacement_metrics(synthset_displacement, modu_synthset_displacement, 'synth_dataset')

# plt.hist(modu_synth_testset_displacement)
# plt.show()

print('Synth MEAN modu: ', np.mean(modu_synth_testset_displacement), " STD modu: ", np.std(modu_synth_testset_displacement))
print('Test MEAN modu: ', np.mean(modu_testset_displacement), " STD modu: ", np.std(modu_testset_displacement))

kl_test2test = kl_norm(np.mean(modu_synth_testset_displacement), np.std(modu_synth_testset_displacement),
                       np.mean(modu_testset_displacement), np.std(modu_testset_displacement))
kl_test2train = kl_norm(np.mean(modu_synth_testset_displacement), np.std(modu_synth_testset_displacement),
                       np.mean(modu_trainset_displacement), np.std(modu_trainset_displacement))
kl_test2data = kl_norm(np.mean(modu_synth_testset_displacement), np.std(modu_synth_testset_displacement),
                       np.mean(modu_full_data_displacement), np.std(modu_full_data_displacement))
kl_train2train = kl_norm(np.mean(modu_synth_trainset_displacement), np.std(modu_synth_trainset_displacement),
                       np.mean(modu_trainset_displacement), np.std(modu_trainset_displacement))
kl_data2data = kl_norm(np.mean(modu_synthset_displacement), np.std(modu_synthset_displacement),
                       np.mean(modu_full_data_displacement), np.std(modu_full_data_displacement))
kl_realtrain2reltest = kl_norm(np.mean(modu_trainset_displacement), np.std(modu_trainset_displacement),
                       np.mean(modu_testset_displacement), np.std(modu_testset_displacement))

def plot_hist(i, data, name):
    plt.figure(i)
    plt.hist(data)
    if save_figures:
        plt.savefig(os.path.join(figures_path, name +'.png'))


plot_hist(15, modu_synth_testset_displacement, 'hist_modu_synth_testset_displacement')
plot_hist(16, modu_testset_displacement, 'hist_modu_testset_displacement')
plot_hist(17, modu_synth_trainset_displacement, 'hist_modu_synth_trainset_displacement')
plot_hist(18, modu_trainset_displacement, 'hist_modu_trainset_displacement')
plot_hist(19, modu_synthset_displacement, 'hist_modu_synthset_displacement')
plot_hist(20, modu_full_data_displacement, 'hist_modu_full_data_displacement')

print('module KL(test, test): ', kl_test2test)
print('module KL(test, train): ', kl_test2train)
print('module KL(test, data): ', kl_test2data)
print('module KL(train, train): ', kl_train2train)
print('module KL(data, data): ', kl_data2data)
print('module KL(real train, real test): ', kl_realtrain2reltest)

kl_test2test = kl_mvn(np.mean(synth_testset_displacement, axis=1), np.cov(synth_testset_displacement),
                       np.mean(testset_displacement, axis=1), np.cov(testset_displacement))
kl_test2train = kl_mvn(np.mean(synth_testset_displacement, axis=1), np.cov(synth_testset_displacement),
                       np.mean(trainset_displacement, axis=1), np.cov(trainset_displacement))
kl_test2data = kl_mvn(np.mean(synth_testset_displacement, axis=1), np.cov(synth_testset_displacement),
                       np.mean(full_data_displacement, axis=1), np.cov(full_data_displacement))
kl_train2train = kl_mvn(np.mean(synth_trainset_displacement, axis=1), np.cov(synth_trainset_displacement),
                       np.mean(trainset_displacement, axis=1), np.cov(trainset_displacement))
kl_data2data = kl_mvn(np.mean(synthset_displacement, axis=1), np.cov(synthset_displacement),
                       np.mean(full_data_displacement, axis=1), np.cov(full_data_displacement))
kl_realtrain2reltest = kl_mvn(np.mean(trainset_displacement, axis=1), np.cov(trainset_displacement),
                       np.mean(testset_displacement, axis=1), np.cov(testset_displacement))

print('disp MVN KL(test, test): ', kl_test2test)
print('disp MVN KL(test, train): ', kl_test2train)
print('disp MVN KL(test, data): ', kl_test2data)
print('disp MVN KL(train, train): ', kl_train2train)
print('disp MVN KL(data, data): ', kl_data2data)
print('disp MVN KL(real train, real test): ', kl_realtrain2reltest)


plt.figure(0)
bar_wide = 0.1
fig, ax = plt.subplots(figsize = (12, 8))
params1 = list(test_mean[40:50])  # test_std, test_median, test_max, test_min]
params2 = list(synth_mean[40:50])  # train_std, train_median, train_max, train_min]
params3 = list(train_mean[40:50])  # test_std, test_median, test_max, test_min]
params4 = list(data_mean[40:50])  # test_std, test_median, test_max, test_min]


br1 = np.arange(len(params1))
br2 = [x + bar_wide for x in br1]
br3 = [x + bar_wide for x in br2]
br4 = [x + bar_wide for x in br3]
br5 = [x + bar_wide for x in br4]
br6 = [x + bar_wide for x in br5]
br7 = [x + bar_wide for x in br6]
br8 = [x + bar_wide for x in br7]

plt.bar(br1, params4, width=bar_wide, label='all data mean')
plt.bar(br2, params3, width=bar_wide, label='trainset mean')
plt.bar(br3, params1, width=bar_wide, label='testset mean')
plt.bar(br4, params2, width=bar_wide, label='synthset mean')

print('Data mean params:\t', params4, '\nTrain mean params:\t', params3, '\nTest mean params:\t', params1, '\nSynth mean params\t', params2)

params1 = list(test_std[40:50])  # test_std, test_median, test_max, test_min]
params2 = list(synth_std[40:50])  # train_std, train_median, train_max, train_min]
params3 = list(train_std[40:50])  # test_std, test_median, test_max, test_min]
params4 = list(data_std[40:50])  # test_std, test_median, test_max, test_min]

plt.bar(br5, params4, width=bar_wide, label='all data std')
plt.bar(br6, params3, width=bar_wide, label='trainset std')
plt.bar(br7, params1, width=bar_wide, label='testset std')
plt.bar(br8, params2, width=bar_wide, label='synthset std')

print('\nData std params\t', params4, '\nTrain std params\t', params3, '\nTest std params:\t', params1, '\nSynth std params\t', params2)

plt.legend()
plt.xlabel('Params')

plt.title('Dataset params')
if save_figures:
    plt.savefig(os.path.join(figures_path, 'params_bars.png'))

plt.figure(1)
bar_wide = 0.2
fig, ax = plt.subplots(figsize = (12, 8))

params1 = list(np.concatenate([test_mean[-5:-2], [test_rms[-3]]], axis=0))  # test_std, test_median, test_max, test_min]
params2 = list(np.concatenate([synth_mean[-5:-2], [synth_rms[-3]]], axis=0))  # train_std, train_median, train_max, train_min]
params3 = list(np.concatenate([train_mean[-5:-2], [train_rms[-3]]], axis=0))  # test_std, test_median, test_max, test_min]
params4 = list(np.concatenate([data_mean[-5:-2], [data_rms[-3]]], axis=0))  # train_std, train_median, train_max, train_min]

br1 = np.arange(len(params1))
br2 = [x + bar_wide for x in br1]
br3 = [x + bar_wide for x in br2]
br4 = [x + bar_wide for x in br3]
br5 = [x + bar_wide for x in br4]
br6 = [x + bar_wide for x in br5]
br7 = [x + bar_wide for x in br6]
br8 = [x + bar_wide for x in br7]

plt.bar(br1, params4, width=bar_wide, label='all data mean')
plt.bar(br2, params3, width=bar_wide, label='trainset mean')
plt.bar(br3, params1, width=bar_wide, label='testset mean')
plt.bar(br4, params2, width=bar_wide, label='synthset mean')

print('Data mean spatial params:\t', params4, '\nTrain mean spatial params:\t', params3, '\nTest mean spatial params:\t', params1, '\nSynth mean spatial params\t', params2)

params1 = list(test_std[-5:-2])  # test_std, test_median, test_max, test_min]
params2 = list(synth_std[-5:-2])  # train_std, train_median, train_max, train_min]
params3 = list(train_std[-5:-2])  # test_std, test_median, test_max, test_min]
params4 = list(data_std[-5:-2])  # train_std, train_median, train_max, train_min]

br1 = np.arange(len(params1))
br2 = [x + bar_wide for x in br1]
br3 = [x + bar_wide for x in br2]
br4 = [x + bar_wide for x in br3]
br5 = [x + bar_wide for x in br4]
br6 = [x + bar_wide for x in br5]
br7 = [x + bar_wide for x in br6]
br8 = [x + bar_wide for x in br7]

plt.bar(br5, params4, width=bar_wide, label='all data std')
plt.bar(br6, params3, width=bar_wide, label='trainset std')
plt.bar(br7, params1, width=bar_wide, label='testset std')
plt.bar(br8, params2, width=bar_wide, label='synthset std')

print('Data std spatial params\t ', params4, '\nTrain std spatial params\t ', params3, '\nTest std spatial params\t ', params1, '\nSynth std spatial params\t', params2)

plt.legend()
plt.xlabel('Features')
ax.set_xticklabels(['', '', 'x_displ', '', 'y_disp', '', 'angles', '', 'angles_rms', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''])

plt.title('Dataset params')
if save_figures:
    plt.savefig(os.path.join(figures_path, 'spatial_params_bars.png'))

params = []
tail = []
angle = []
displacement = []
coordinates = []
counter = 0
for d in testset:

    params.append(np.reshape(d[40:50], (5, 2)))
    tail.append(np.reshape(d[:40], (20, 2)))
    angle.append(d[-3])
    displacement.append(d[-5:-3])
    coordinates.append(d[-2:])

    counter += 1

    legend = plt.legend()


def scater_2d(vector, mode='test', printlns=False, alpha=0.15):

    x = vector[:, :, 0]
    y = vector[:, :, 1]

    x = np.transpose(x, (1, 0))
    y = np.transpose(y, (1, 0))

    plt.figure(figsize=(12, 6))

    count = 0
    for i in range(x.shape[0]):
        x_ = np.ravel(x[i])
        y_ = np.ravel(y[i])

        mean_x = np.mean(x_)
        mean_y = np.mean(y_)
        std_x = np.std(x_)
        std_y = np.std(y_)

        # Create a scatter plot of the points
        plt.scatter(x_, y_, label='Data Points', edgecolors=None, alpha=alpha)

        # Plot the mean point as a red star
        # plt.scatter(mean_x, mean_y, color='red', marker='*', s=200, label='Mean')

        # Plot ellipses representing standard deviation
        # ellipse_x = mean_x + std_x * np.cos(np.linspace(0, 2 * np.pi, 100))
        # ellipse_y = mean_y + std_y * np.sin(np.linspace(0, 2 * np.pi, 100))
        # plt.plot(ellipse_x, ellipse_y, 'r--', label='1 Std Dev')

        if printlns:
            print(mode, ' param ', count, ' mean(x, y) = (', mean_x, ', ', mean_y, '); std(x, y) = (', std_x, ' ,', std_y,')')

        count += 1
    # Set labels and legend
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.legend()
    plt.xlim(-1.3, 0.7)
    plt.ylim(-0.5, 0.5)
    plt.xticks([-1, -0.5, 0, 0.5], fontsize=40)
    plt.yticks([-0.5, 0, 0.5], fontsize=40)

    plt.grid(True)
    fig.tight_layout()

    if save_figures:
        plt.savefig(os.path.join(figures_path, 'params_scatter_' +mode+ '.png'))


def lines_2d(vector, mode='test', printlns=False, alpha=0.15):

    x = vector[:, :, 0]
    y = vector[:, :, 1]

    # x = np.transpose(x, (1, 0))
    # y = np.transpose(y, (1, 0))

    plt.figure(figsize=(12, 6))

    count = 0
    for i in range(x.shape[0]):
        x_ = x[i]
        y_ = y[i]

        mean_x = np.mean(x_)
        mean_y = np.mean(y_)
        std_x = np.std(x_)
        std_y = np.std(y_)

        # Create a scatter plot of the points
        plt.plot(x_, y_,  label='Data Points', alpha=alpha, linestyle='-', color=colors['green'])

        if printlns:
            print(mode, ' param ', count, ' mean(x, y) = (', mean_x, ', ', mean_y, '); std(x, y) = (', std_x, ' ,', std_y,')')

        count += 1
    # Set labels and legend
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.legend()
    plt.xlim(-1.3, 0.7)
    plt.ylim(-0.5, 0.5)
    # plt.tick_params(left=False, right=False, labelleft=False,
    #                 labelbottom=False, bottom=False)
    plt.xticks([-1, -0.5, 0, 0.5], fontsize=40)
    plt.yticks([-0.5, 0, 0.5], fontsize=40)
    # Show the plot
    # plt.title('lines' + mode)

    plt.grid(True)
    fig.tight_layout()

    if save_figures:
        plt.savefig(os.path.join(figures_path, 'params_lines_' +mode+ '.png'))

plt.figure(2)
scater_2d(np.asarray(params), printlns=False, alpha=0.4)

plt.figure(3)
scater_2d(np.asarray(tail), mode='tail_test', alpha=0.2)

plt.figure(4)
lines_2d(np.asarray(tail), mode='tail_test', printlns=False, alpha=0.1)

params = []
tail = []
angle = []
displacement = []
coordinates = []
counter = 0
for d in trainset:

    params.append(np.reshape(d[40:50], (5, 2)))
    tail.append(np.reshape(d[:40], (20, 2)))
    angle.append(d[-3])
    displacement.append(d[-5:-3])
    coordinates.append(d[-2:])

    counter += 1

plt.figure(14)
scater_2d(np.asarray(params), printlns=False, alpha=0.4, mode='train')

plt.figure(5)
scater_2d(np.asarray(tail), mode='tail_train', alpha=0.01)

plt.figure(6)
lines_2d(np.asarray(tail), mode='tail_train', printlns=False, alpha=0.1)

params = []
tail = []
angle = []
displacement = []
coordinates = []
counter = 0
for d in full_data:

    params.append(np.reshape(d[40:50], (5, 2)))
    tail.append(np.reshape(d[:40], (20, 2)))
    angle.append(d[-3])
    displacement.append(d[-5:-3])
    coordinates.append(d[-2:])

    counter += 1

plt.figure(15)
scater_2d(np.asarray(params), printlns=False, alpha=0.2, mode='data')

plt.figure(7)
scater_2d(np.asarray(tail), mode='tail_all', alpha=0.01)

plt.figure(8)
lines_2d(np.asarray(tail), mode='tail_all', printlns=False, alpha=0.1)

print('Test angle mean: ', np.mean(angle), ' std: ', np.std(angle), ' min: ', np.min(angle), ' max: ', np.max(angle))

params = []
tail = []
angle = []
displacement = []
coordinates = []

model_tail = []
counter = 0
for d in synthset:

    params.append(np.reshape(d[40:50], (5, 2)))
    tail.append(np.reshape(d[:40], (20, 2)))

    linspace = np.linspace(0., 1., num=20)
    model_tail.append(Bezier.Curve(linspace, np.reshape(d[40:50], (5, 2))))

    angle.append(d[-3])
    displacement.append(d[-5:-3])
    coordinates.append(d[-2:])

    counter += 1

plt.figure(9)
scater_2d(np.asarray(params), mode='synth', printlns=False, alpha=0.05)
plt.figure(10)
scater_2d(np.asarray(tail), mode='tail_synth', printlns=False, alpha=0.05)
plt.figure(11)
scater_2d(np.asarray(model_tail), mode='tail_bezier_spline_synth', printlns=False, alpha=0.05)


plt.figure(12)
lines_2d(np.asarray(tail), mode='tail_synth', printlns=False, alpha=0.05)
plt.figure(13)
lines_2d(np.asarray(model_tail), mode='tail_bezier_spline_synth', printlns=False, alpha=0.05)





