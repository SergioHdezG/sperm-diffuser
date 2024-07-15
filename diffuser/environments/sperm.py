import math
import random
import sys
import time
from os import path

from diffuser.environments.utils.Bezier import Bezier
from diffuser.environments.utils.sperm_rendering import vec2angle

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import os
# from environments.env_base import EnvInterface, ActionSpaceInterface
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms
from numpy import array as arr
# from Bezier import Bezier
# from shapely.geometry import LineString
# from shapely.geometry import Point
# import cv2
from collections import deque
import matplotlib.path as mpath
import matplotlib as mpl
# from utils.quality_metrics import issm, ssim, fsim
import json
import ast

import numpy as np

import gym
from gym import spaces

GET_TRAIN_SET = True

class SingleSpermBezierDeepmimic(gym.Env):
    def __init__(self,
                 render_mode='rgb_array', data_file='diffuser/datasets/BezierSplinesData/moving'):
        self.timesteps = 0

        self.action_space = spaces.Box(low=-1, high=1., shape=(13,), dtype=np.float32)
        high = np.array([10. for i in range(107)]).astype(np.float32)
        low = np.array([-10. for i in range(107)]).astype(np.float32)
        self.observation_space = spaces.Box(low, high)

        self.data_file = data_file

        video = 'diffuser/datasets/test-1-29-field_1_30/frames'

        self.frames = self._read_frames(video)
        for path, directories, files in os.walk(self.data_file):
            print(path)
            if len(files) > 0 and files[0].endswith('.json'):
                self.states, self.coords, self.heads, self.angles, self.frames, self.params, self.comp_coords, self.comp_heads, self.comp_angl, self.comp_params = self._read_trajectory(path)
                break
        self._max_episode_steps = len(self.states)-1
        self.max_timesteps = 15
        self.min_timesteps = 5
        self.step_counter = 0

        self.render_mode = render_mode

        self.current_coords = None
        self.current_coords1 = None
        self.current_coords2 = None
        self.current_head = None
        self.current_angle = None
        self.parameters = None
        self.state0 = None
        self.state1 = None
        self.state2 = None

        self.accum_tail_error = 0.
        self.accum_tail_mom_error = 0.

        self._train_set = None
        self._test_set = None

    def get_dataset(self):
        train_states_to_save = []
        train_actions_to_save = []
        train_next_states_to_save = []
        train_terminals_to_save = []
        test_states_to_save = []
        test_actions_to_save = []
        test_next_states_to_save = []
        test_terminals_to_save = []

        random.seed(1)
        for path, directories, files in os.walk(self.data_file):
            print(path)
            if len(files) > 0 and files[0].endswith('.json'):
                self.states, self.coords, self.heads, self.angles, self.frames, self.params, self.comp_coords, self.comp_heads, self.comp_angl, self.comp_params = self._read_trajectory(
                    path)
                states = self.states[:-1]
                next_states = self.states[1:]
                actions = [np.concatenate([np.ravel(p), [a], h[:-1]], axis=0) for p, a, h in
                           zip(self.params[1:], self.angles[1:], self.heads[1:])]
                terminals = np.concatenate([np.asarray(states[1:])[:, -1] == 0.0, [True]])

                if random.random() > 0.1:
                    train_states_to_save.extend(states)
                    train_actions_to_save.extend(actions)
                    train_next_states_to_save.extend(next_states)
                    train_terminals_to_save.extend(terminals)
                else:
                    test_states_to_save.extend(states)
                    test_actions_to_save.extend(actions)
                    test_next_states_to_save.extend(next_states)
                    test_terminals_to_save.extend(terminals)

        train_dict = {'actions': np.asarray(train_actions_to_save, dtype=np.float32),
                # 'infos/action_log_probs': np.ones((states.shape[0], 1),
                # 'infos/qpos': np.ones((states.shape[0], 1),
                # 'infos/qvel': np.ones((states.shape[0], 1),
                'next:observations': np.asarray(train_next_states_to_save, dtype=np.float32),
                'observations': np.asarray(train_states_to_save, dtype=np.float32),
                'rewards': np.ones((len(train_actions_to_save), 1)),
                'terminals': np.array(train_terminals_to_save),
                'timeouts': np.zeros((len(train_actions_to_save), 1)),
                }

        self._train_set = train_dict

        test_dict = {'actions': np.asarray(test_actions_to_save, dtype=np.float32),
                # 'infos/action_log_probs': np.ones((states.shape[0], 1),
                # 'infos/qpos': np.ones((states.shape[0], 1),
                # 'infos/qvel': np.ones((states.shape[0], 1),
                'next:observations': np.asarray(test_next_states_to_save, dtype=np.float32),
                'observations': np.asarray(test_states_to_save, dtype=np.float32),
                'rewards': np.ones((len(test_actions_to_save), 1)),
                'terminals': np.asarray(test_terminals_to_save),
                'timeouts': np.zeros((len(test_actions_to_save), 1)),
                }

        self._test_set = test_dict

        global GET_TRAIN_SET
        if GET_TRAIN_SET:
            dict = self._train_set
            GET_TRAIN_SET = False
        else:
            dict = self._test_set
            GET_TRAIN_SET = True
        return dict

    def reset(self, seed=0):
        index = random.randint(1, self.max_timesteps - self.min_timesteps)

        state = self.states[index]
        self.step_counter = index
        self.timesteps = 0

        self.state0 = self.states[index][:53]  # Don't take te time encoding
        self.state1 = self.states[index-1][:53]  #  Don't take te time encoding

        self.current_coords = self.coords[index]
        self.current_coords1 = self.coords[index-1]
        self.current_coords2 = self.coords[index-1]
        self.current_head = self.heads[index]
        self.current_angle = self.angles[index]
        self.parameters = None
        self.accum_tail_error = 0.
        self.accum_tail_mom_error = 0.

        return state, {'loss/tail_loss': 0.,
                       'loss/x_loss': 0.,
                       'loss/y_loss': 0.,
                       'loss/angle_loss': 0.,
                       'reward/tail_rew': 0.,
                       'reward/x_rew': 0.,
                       'reward/y_rew': 0.,
                       'reward/angle_rew': 0.,
                       'action/parameters': np.zeros(shape=self.action_space.shape[0]-3),
                       'action/angle': 0.,
                       'action/x': 0.,
                       'action/y': 0.,
                       'loss/momentum_head_loss': 0.,
                       'loss/momentum_angle_loss': 0.,
                       'loss/momentum_tail_loss': 0.,
                       'reward/momentum_head_loss': 0.,
                       'reward/momentum_angle_loss': 0.,
                       'reward/momentum_tail_loss': 0.,
                        }

    def step(self, a):
        linspace = np.linspace(0., 1., num=20)
        angle = a[-1]
        x = a[-2]
        y = a[-3]
        parameters = a[:-3].reshape((int(a[:-3].shape[0] / 2), 2))*1.25

        model = Bezier.Curve(linspace, parameters)

        tail_loss = self._bezier_spline_loss2(model, self.coords[self.step_counter+1])
        x_loss = np.square(x - self.heads[self.step_counter+1][0])
        y_loss = np.square(y - self.heads[self.step_counter+1][1])
        angle_loss = np.square(angle - self.angles[self.step_counter+1])
        momentum_head_loss = (np.sqrt(np.square(x - self.heads[self.step_counter][0]) + np.square(y - self.heads[self.step_counter][1]))) \
                              - (np.sqrt(np.square(self.heads[self.step_counter+1][0] - self.heads[self.step_counter][0]) + np.square(self.heads[self.step_counter+1][1] - self.heads[self.step_counter][1])))
        momentum_angle_loss = np.sqrt(np.square(angle - self.angles[self.step_counter])) - np.sqrt(np.square(self.angles[self.step_counter+1] - self.angles[self.step_counter]))
        momentum_tail_loss = np.sum(np.sqrt(np.square(model - self.coords[self.step_counter])) - np.sqrt(np.square(self.coords[self.step_counter+1] - self.coords[self.step_counter])))
        params_loss = self._bezier_spline_loss2(parameters, self.params[self.step_counter+1])

        tail_coeff = 0.
        x_coeff = 1.
        y_coeff = 1.
        angle_coeff = 1.
        m_h_coeff = 0
        m_a_coeff = 0
        m_t_coeff = 0.
        param_coeff = 1.

        # reward = tail_coeff * ((np.exp(-7. * tail_loss)*2-1) + x_coeff * (np.exp(-7. * x_loss)*2-1) + y_coeff * (np.exp(
        #     -7. * y_loss)*2-1) + angle_coeff * (np.exp(-7. * angle_loss))*2-1) + m_h_coeff*(np.exp(-7. * momentum_head_loss)*2-1) \
        #     + m_a_coeff*(np.exp(-7. * momentum_angle_loss)*2-1) + m_t_coeff*(np.exp(-7. * momentum_tail_loss)*2-1)

        reward = param_coeff*(np.exp(-7*params_loss)*2-1) + angle_coeff * (np.exp(-7. * angle_loss)*2-1) + x_coeff * (np.exp(-7. * x_loss)*2-1) + y_coeff * (np.exp(-7. * y_loss)*2-1)

        state = np.concatenate([np.ravel(model), [angle], np.ravel(parameters), [x, y, (self.step_counter)/self._max_episode_steps]], axis=0)

        self.accum_tail_error += tail_loss
        self.accum_tail_mom_error += momentum_tail_loss
        self.state1 = np.copy(self.state0)
        self.state0 = state[:-1]  #  Don't take te time encoding

        state = np.concatenate([state, self.state1])
        self.current_coords2 = self.current_coords1
        self.current_coords1 = self.current_coords
        self.current_coords = model
        self.current_head = [x, y, (self.step_counter)/self._max_episode_steps]
        self.current_angle = angle
        self.parameters = parameters

        self.step_counter += 1
        self.timesteps += 1

        done = self.timesteps >= self.max_timesteps or self.step_counter >= self._max_episode_steps or self.accum_tail_error > 5. #or self.accum_tail_mom_error > 3.

        return state, reward, done, done, {'loss/tail_loss': tail_loss,
                                           'loss/x_loss': x_loss,
                                           'loss/y_loss': y_loss,
                                           'loss/angle_loss': angle_loss,
                                           'reward/tail_rew': np.exp(-7.*tail_loss)*2-1,
                                           'reward/x_rew': x_coeff * np.exp(-7.*x_loss)*2-1,
                                           'reward/y_rew': y_coeff * np.exp(-7.*y_loss)*2-1,
                                           'reward/angle_rew': np.exp(-7.*angle_loss)*2-1,
                                           'action/parameters': parameters,
                                           'action/angle': angle,
                                           'action/x': x,
                                           'action/y': y,
                                           'loss/momentum_head_loss': momentum_head_loss,
                                           'loss/momentum_angle_loss': momentum_angle_loss,
                                           'loss/momentum_tail_loss': momentum_tail_loss,
                                           'reward/momentum_head_loss': np.exp(-7. * momentum_head_loss)*2-1,
                                           'reward/momentum_angle_loss': np.exp(-7. * momentum_angle_loss)*2-1,
                                           'reward/momentum_tail_loss': np.exp(-7. * momentum_tail_loss)*2-1,
                                           'loss/params_loss': params_loss,
                                           'rewards/params': np.exp(-3*params_loss)*2-1,
                                            }

    def render(self):
        pass
        # img = plt2opencv((self.current_coords+1)*70., (140, 140), (280, 280), parameters=(self.parameters+1)*70)
        # rot = mpl.transforms.Affine2D().rotate_deg(
        #     self.current_angle*180)
        # rot_params = rot.transform((self.parameters+1)*70)
        # displacement = (self.parameters[-1]+1)*70 - rot_params[-1]
        # rot_params += displacement
        # rot_coords = rot.transform(self.current_coords)
        # displacement = self.current_coords[-1] - rot_coords[-1]
        # rot_coords += displacement
        # rot_img = plt2opencv((rot_coords+1)*70, (140, 140), (280, 280), parameters=rot_params)
        #
        # spatial_img = plt2opencvcoords(self.current_head[:-1], (rot_coords+1)*70,  resize=(280, 280))
        #
        # new_frame = np.zeros((280*2, 280*4, 3), dtype=np.uint8)
        #
        # if self.step_counter < len(self.coords):
        #     prev_img = plt2opencv2layers((self.current_coords1 + 1) * 70., (self.current_coords2 + 1) * 70., (140, 140),
        #                             (280, 280))
        #
        #     # real_img = plt2opencv((self.coords[self.step_counter]+1)*70., (140, 140), (280, 280))
        #     new_frame[280:, :280] = prev_img
        #
        #     text = plotvalues([self.current_head[:-1]], [self.current_angle], img_size=(280, 280))
        #     new_frame[:280, 280 * 3:] = text
        #     new_frame[:280, :280*3] = cv2.resize(self.frames[self.step_counter], (280 * 3, 280))
        #     try:
        #         real_traj = plt2opencvlist([(self.coords[c] + 1) * 70. for c in range(np.max(0, self.step_counter-3), self.step_counter+1)], (140, 140),
        #                                 (280, 280))
        #         new_frame[:280, 280*2:280*3] = cv2.resize(real_traj, (280, 280))
        #     except:
        #         pass
        #
        # new_frame[280:, 280:280 * 2] = rot_img
        # new_frame[280:, 280 * 2:280*3] = img
        # new_frame[280:, 280 * 3:] = spatial_img
        # return new_frame

    def _read_frames(self, frame_path):
        frames = []
        for path, directories, files in os.walk(frame_path):
            files.sort()
            for f in files:
                if f.endswith('.jpg'):
                    frames.append(plt.imread(os.path.join(path, f)))
        return frames


    def _read_trajectory(self, spline_path, frames=None, num=25):

        states = []
        coords = []
        heads = []
        angles = []
        params = []
        for path, directories, files in os.walk(spline_path):
            files.sort()
            for f in files:
                if f.endswith('.json'):
                    json_data = read_json(os.path.join(path, f))
                    point_pairs = np.asarray(json_data['spline_line_space'])
                    parameters = np.asarray(json_data['spline_params'])/70.-1

                    # point_pairs = np.concatenate([np.expand_dims(curve[:, 0], axis=-1), np.expand_dims(curve[:, 1], axis=-1)], axis=-1)
                    point_pairs[:, 0] = point_pairs[:, 0]/70. - 1
                    point_pairs[:, 1] = point_pairs[:, 1]/70. - 1

                    ravel_point_pairs = np.ravel(point_pairs)
                    angle = json_data['correction_angle']/180.
                    head = json_data['head_coordinates']
                    head = [head[0], 1024 - head[1], head[2]]

                    head[0] = (head[0] / 1280)*2 - 1
                    head[1] = (head[1] / 1024)*2 - 1
                    head[2] = head[2] / len(files)
                    states.append(np.concatenate([ravel_point_pairs, np.ravel(parameters), [angle], head], axis=0))
                    coords.append(point_pairs)
                    heads.append(head)
                    angles.append(angle)
                    params.append(parameters)

        new_states = []
        new_coords = []
        new_heads = []
        new_angles = []
        new_frames = []
        new_params = []
        for i in range(len(states)):
            j = np.maximum(i - 1, 0)
            if j != i:
                new_states.append(states[i][:-1])  #np.concatenate([states[i][:-1], states[j][:-1]]))
                new_coords.append(coords[i])
                new_heads.append(heads[i])
                new_angles.append(angles[i])
                if frames is not None:
                    new_frames.append(frames[i])
                new_params.append(params[i])
        return new_states, new_coords, new_heads, new_angles, new_frames, new_params, coords, heads, angles, params

    def _bezier_spline_loss2(self, x, y):
        loss = []
        # for ys in y:
        #     min = 100
        #     for xs in x:
        #         if np.abs(xs[0] - ys[0]) < 0.1:
        #             min = np.sqrt(np.square(xs[0] - ys[0]) + np.square(xs[1] - ys[1]))
        #     loss.append(min)
        for xs, ys in zip(x, y):
            loss.append(np.square(xs[0] - ys[0]) + np.square(xs[1] - ys[1]))
        return np.sum(loss)

class SingleSpermBezierIncrements(SingleSpermBezierDeepmimic):
    def _read_trajectory(self, spline_path, frames=None, num=1):

        states = []
        coords = []
        heads = []
        angles = []
        params = []
        for path, directories, files in os.walk(spline_path):
            files.sort()

            first_iter = True
            old_head = [0.0, 0.0]
            for f in files:
                if f.endswith('.json'):
                    json_data = read_json(os.path.join(path, f))
                    point_pairs = np.asarray(json_data['spline_line_space'])
                    parameters = np.asarray(json_data['spline_params'])/70.-1

                    # point_pairs = np.concatenate([np.expand_dims(curve[:, 0], axis=-1), np.expand_dims(curve[:, 1], axis=-1)], axis=-1)
                    point_pairs[:, 0] = point_pairs[:, 0]/70. - 1
                    point_pairs[:, 1] = point_pairs[:, 1]/70. - 1

                    ravel_point_pairs = np.ravel(point_pairs)
                    angle = json_data['correction_angle']/180.
                    head = json_data['head_coordinates']
                    head = [head[0], 1024 - head[1], head[2]]

                    if first_iter:
                        old_head[0] = head[0]
                        old_head[1] = head[1]
                        first_iter = False
                    head_displacement = np.clip([float(head[0]-old_head[0])/20., float(head[1]-old_head[1])/20.], -1.0, 1.0)
                    old_head[0] = head[0]
                    old_head[1] = head[1]
                    head[0] = (head[0] / 1280)*2 - 1
                    head[1] = (head[1] / 1024)*2 - 1
                    head[2] = head[2] / len(files)
                    states.append(np.concatenate([ravel_point_pairs, np.ravel(parameters), head_displacement, [angle], head], axis=0))
                    coords.append(point_pairs)
                    heads.append(head)
                    angles.append(angle)
                    params.append(parameters)


        new_states = []
        new_coords = []
        new_heads = []
        new_angles = []
        new_frames = []
        new_params = []
        for i in range(len(states)):
            j = np.maximum(i - 1, 0)
            if j != i:
                new_states.append(states[i][:-1])  #np.concatenate([states[i][:-1], states[j][:-1]]))
                new_coords.append(coords[i])
                new_heads.append(heads[i])
                new_angles.append(angles[i])
                if frames is not None:
                    new_frames.append(frames[i])
                new_params.append(params[i])
        return new_states, new_coords, new_heads, new_angles, new_frames, new_params, coords, heads, angles, params


    def get_dataset(self):
        train_states_to_save = []
        train_actions_to_save = []
        train_next_states_to_save = []
        train_terminals_to_save = []
        test_states_to_save = []
        test_actions_to_save = []
        test_next_states_to_save = []
        test_terminals_to_save = []

        random.seed(1)
        for path, directories, files in os.walk(self.data_file):
            print(path)
            if len(files) > 0 and files[0].endswith('.json'):
                self.states, self.coords, self.heads, self.angles, self.frames, self.params, self.comp_coords, self.comp_heads, self.comp_angl, self.comp_params = self._read_trajectory(
                    path)
                states = self.states[:-1]
                next_states = self.states[1:]
                actions = [np.concatenate([np.ravel(p), [a], [(h[0]-p_h[0])*102.4, (h[1]-p_h[1])*128]], axis=0) for p, a, h, p_h in
                           zip(self.params[1:], self.angles[1:], self.heads[1:], self.heads[:-1])]
                terminals = np.concatenate([np.asarray(states[1:])[:, -1] == 0.0, [True]])

                if random.random() > 0.1:
                    train_states_to_save.extend(states)
                    train_actions_to_save.extend(actions)
                    train_next_states_to_save.extend(next_states)
                    train_terminals_to_save.extend(terminals)
                else:
                    test_states_to_save.extend(states)
                    test_actions_to_save.extend(actions)
                    test_next_states_to_save.extend(next_states)
                    test_terminals_to_save.extend(terminals)

        train_dict = {'actions': np.asarray(train_actions_to_save, dtype=np.float32),
                # 'infos/action_log_probs': np.ones((states.shape[0], 1),
                # 'infos/qpos': np.ones((states.shape[0], 1),
                # 'infos/qvel': np.ones((states.shape[0], 1),
                'next:observations': np.asarray(train_next_states_to_save, dtype=np.float32),
                'observations': np.asarray(train_states_to_save, dtype=np.float32),
                'rewards': np.ones((len(train_actions_to_save), 1)),
                'terminals': np.array(train_terminals_to_save),
                'timeouts': np.zeros((len(train_actions_to_save), 1)),
                }

        self._train_set = train_dict

        test_dict = {'actions': np.asarray(test_actions_to_save, dtype=np.float32),
                # 'infos/action_log_probs': np.ones((states.shape[0], 1),
                # 'infos/qpos': np.ones((states.shape[0], 1),
                # 'infos/qvel': np.ones((states.shape[0], 1),
                'next:observations': np.asarray(test_next_states_to_save, dtype=np.float32),
                'observations': np.asarray(test_states_to_save, dtype=np.float32),
                'rewards': np.ones((len(test_actions_to_save), 1)),
                'terminals': np.asarray(test_terminals_to_save),
                'timeouts': np.zeros((len(test_actions_to_save), 1)),
                }

        self._test_set = test_dict

        global GET_TRAIN_SET
        if GET_TRAIN_SET:
            dict = self._train_set
            GET_TRAIN_SET = False
        else:
            dict = self._test_set
            GET_TRAIN_SET = True
        return dict

    def step(self, a):
        linspace = np.linspace(0., 1., num=20)
        angle = a[-3]
        delta_x = a[-2]/102.4
        delta_y = a[-1]/128
        x = self.current_head[0] + delta_x
        y = self.current_head[1] + delta_y

        parameters = a[:-3].reshape((int(a[:-3].shape[0] / 2), 2))*1.25

        model = Bezier.Curve(linspace, parameters)

        tail_loss = self._bezier_spline_loss2(model, self.coords[self.step_counter+1])
        x_loss = np.square(x - self.heads[self.step_counter+1][0])
        y_loss = np.square(y - self.heads[self.step_counter+1][1])
        angle_loss = np.square(angle - self.angles[self.step_counter+1])
        momentum_head_loss = (np.sqrt(np.square(x - self.heads[self.step_counter][0]) + np.square(y - self.heads[self.step_counter][1]))) \
                              - (np.sqrt(np.square(self.heads[self.step_counter+1][0] - self.heads[self.step_counter][0]) + np.square(self.heads[self.step_counter+1][1] - self.heads[self.step_counter][1])))
        momentum_angle_loss = np.sqrt(np.square(angle - self.angles[self.step_counter])) - np.sqrt(np.square(self.angles[self.step_counter+1] - self.angles[self.step_counter]))
        momentum_tail_loss = np.sum(np.sqrt(np.square(model - self.coords[self.step_counter])) - np.sqrt(np.square(self.coords[self.step_counter+1] - self.coords[self.step_counter])))
        params_loss = self._bezier_spline_loss2(parameters, self.params[self.step_counter+1])

        tail_coeff = 0.
        x_coeff = 1.
        y_coeff = 1.
        angle_coeff = 1.
        m_h_coeff = 0
        m_a_coeff = 0
        m_t_coeff = 0.
        param_coeff = 1.

        # reward = tail_coeff * ((np.exp(-7. * tail_loss)*2-1) + x_coeff * (np.exp(-7. * x_loss)*2-1) + y_coeff * (np.exp(
        #     -7. * y_loss)*2-1) + angle_coeff * (np.exp(-7. * angle_loss))*2-1) + m_h_coeff*(np.exp(-7. * momentum_head_loss)*2-1) \
        #     + m_a_coeff*(np.exp(-7. * momentum_angle_loss)*2-1) + m_t_coeff*(np.exp(-7. * momentum_tail_loss)*2-1)

        reward = param_coeff*(np.exp(-7*params_loss)*2-1) + angle_coeff * (np.exp(-7. * angle_loss)*2-1) + x_coeff * (np.exp(-7. * x_loss)*2-1) + y_coeff * (np.exp(-7. * y_loss)*2-1)

        state = np.concatenate([np.ravel(model), [angle], np.ravel(parameters), [x, y, (self.step_counter)/self._max_episode_steps]], axis=0)

        self.accum_tail_error += tail_loss
        self.accum_tail_mom_error += momentum_tail_loss
        self.state1 = np.copy(self.state0)
        self.state0 = state[:-1]  #  Don't take te time encoding

        state = np.concatenate([state, self.state1])
        self.current_coords2 = self.current_coords1
        self.current_coords1 = self.current_coords
        self.current_coords = model
        self.current_head = [x, y, (self.step_counter)/self._max_episode_steps]
        self.current_angle = angle
        self.parameters = parameters
        self.current_head[0] = x
        self.current_head[1] = y
        self.step_counter += 1
        self.timesteps += 1

        done = self.timesteps >= self.max_timesteps or self.step_counter >= self._max_episode_steps or self.accum_tail_error > 5. #or self.accum_tail_mom_error > 3.

        return state, reward, done, done, {'loss/tail_loss': tail_loss,
                                           'loss/x_loss': x_loss,
                                           'loss/y_loss': y_loss,
                                           'loss/angle_loss': angle_loss,
                                           'reward/tail_rew': np.exp(-7.*tail_loss)*2-1,
                                           'reward/x_rew': x_coeff * np.exp(-7.*x_loss)*2-1,
                                           'reward/y_rew': y_coeff * np.exp(-7.*y_loss)*2-1,
                                           'reward/angle_rew': np.exp(-7.*angle_loss)*2-1,
                                           'action/parameters': parameters,
                                           'action/angle': angle,
                                           'action/x': x,
                                           'action/y': y,
                                           'loss/momentum_head_loss': momentum_head_loss,
                                           'loss/momentum_angle_loss': momentum_angle_loss,
                                           'loss/momentum_tail_loss': momentum_tail_loss,
                                           'reward/momentum_head_loss': np.exp(-7. * momentum_head_loss)*2-1,
                                           'reward/momentum_angle_loss': np.exp(-7. * momentum_angle_loss)*2-1,
                                           'reward/momentum_tail_loss': np.exp(-7. * momentum_tail_loss)*2-1,
                                           'loss/params_loss': params_loss,
                                           'rewards/params': np.exp(-3*params_loss)*2-1,
                                            }

class SingleSpermBezierIncrementsDataAug(SingleSpermBezierIncrements):
    # State: [Tail coords, Spline params, displacement, angle, head coordinates]

    def _read_trajectory(self, spline_path, frames=None, num=1, test=False):
        def displacement(coords, x_rand, y_rand):
            x_disp = x_rand - coords[0][0]
            y_disp = y_rand - coords[0][1]
            coords = np.asarray(coords)

            coords[:, 0] += x_disp
            coords[:, 1] += y_disp
            # coords[:, 0] = coords[:, 0]*2-1
            # coords[:, 1] = coords[:, 1] * 2 - 1
            return coords[-1, :]
        states = []
        coords = []
        heads = []
        angles = []
        params = []


        for path, directories, files in os.walk(spline_path):
            files.sort()

            first_iter = True
            old_head = [0.0, 0.0]

            rand_rot = random.random() * 2 - 1
            x_rand = ((random.random()) * 1280) #* 0.9 + 0.05
            y_rand = ((random.random()) * 1024) #* 0.9 + 0.05

            head_list = []
            real_head_list = []
            temp_states = []
            temp_coords = []
            temp_heads = []
            temp_angles = []
            temp_params = []
            for f in files:
                if f.endswith('.json'):
                    json_data = read_json(os.path.join(path, f))
                    point_pairs = np.asarray(json_data['spline_line_space'])
                    parameters = np.asarray(json_data['spline_params'])/70.-1

                    # point_pairs = np.concatenate([np.expand_dims(curve[:, 0], axis=-1), np.expand_dims(curve[:, 1], axis=-1)], axis=-1)
                    point_pairs[:, 0] = point_pairs[:, 0]/70. - 1
                    point_pairs[:, 1] = point_pairs[:, 1]/70. - 1

                    ravel_point_pairs = np.ravel(point_pairs)

                    if not test:
                        angle = (((json_data['correction_angle']/180 + rand_rot)+1) % 2) -1.

                        head = json_data['head_coordinates']
                        head = [head[0], 1024-head[1], head[2]]
                        real_head_list.append(head)
                        rot = mpl.transforms.Affine2D().rotate_deg(rand_rot * 180)
                        rot_heads = rot.transform(np.asarray(head[:-1]))
                        head_list.append(rot_heads)

                        head = [*displacement(np.asarray(head_list), x_rand, y_rand), head[-1]]
                    else:
                        angle = (((json_data['correction_angle'] / 180) + 1) % 2) - 1.

                        head = json_data['head_coordinates']
                        head = [head[0], 1024-head[1], head[2]]
                        real_head_list.append(head)

                    if first_iter:
                        old_head[0] = head[0]
                        old_head[1] = head[1]
                        first_iter = False
                    head_displacement = np.clip([float(head[0]-old_head[0])/20., float(head[1]-old_head[1])/20.], -1.0, 1.0)
                    old_head[0] = head[0]
                    old_head[1] = head[1]
                    head[0] = (head[0] / 1280)*2 - 1
                    head[1] = (head[1] / 1024)*2 - 1


                    temp_states.append(np.concatenate([ravel_point_pairs, np.ravel(parameters), head_displacement, [angle], head], axis=0))
                    temp_coords.append(point_pairs)
                    temp_heads.append(head)
                    temp_angles.append(angle)
                    temp_params.append(parameters)

                    # states.append(np.concatenate([ravel_point_pairs, np.ravel(parameters), head_displacement, [angle], head], axis=0))
                    # coords.append(point_pairs)
                    # heads.append(head)
                    # angles.append(angle)
                    # params.append(parameters)

            if not np.any(np.array(temp_heads)[:, :-1] > 1) and not np.any(np.array(temp_heads)[:, :-1] < -1):
                states.extend(temp_states[1:])
                coords.extend(temp_coords[1:])
                heads.extend(temp_heads[1:])
                angles.extend(temp_angles[1:])
                params.extend(temp_params[1:])

        new_states = []
        new_coords = []
        new_heads = []
        new_angles = []
        new_frames = []
        new_params = []
        for i in range(len(states)):
            new_states.append(states[i][:-1])  #np.concatenate([states[i][:-1], states[j][:-1]]))
            new_coords.append(coords[i])
            new_heads.append(heads[i])
            new_angles.append(angles[i])
            if frames is not None:
                new_frames.append(frames[i])
            new_params.append(params[i])
        return new_states, new_coords, new_heads, new_angles, new_frames, new_params, coords, heads, angles, params


    def get_dataset(self):
        n_augmentation = 1
        train_states_to_save = []
        train_actions_to_save = []
        train_next_states_to_save = []
        train_terminals_to_save = []
        test_states_to_save = []
        test_actions_to_save = []
        test_next_states_to_save = []
        test_terminals_to_save = []


        random.seed(1)
        np.random.seed(1)



        traj_files = []
        for path, directories, files in os.walk(self.data_file):
            print(path)
            if len(files) > 0 and files[0].endswith('.json'):
                traj_files.append([path, files])

        num_to_choose = int(len(traj_files) * 0.2)
        test_indices = np.random.choice(range(len(traj_files)), size=num_to_choose, replace=False)
        train_indices = np.setdiff1d(range(len(traj_files)), test_indices)
        for i in range(n_augmentation):
            for [path, files] in np.asarray(traj_files)[train_indices]:
                print(path)
                if len(files) > 0 and files[0].endswith('.json'):
                    self.states, self.coords, self.heads, self.angles, self.frames, self.params, self.comp_coords, self.comp_heads, self.comp_angl, self.comp_params = self._read_trajectory(
                        path)
                    states = self.states[:-1]
                    if len(states) > 0:
                        next_states = self.states[1:]
                        actions = [np.concatenate([np.ravel(p), [a], [(h[0]-p_h[0])*102.4, (h[1]-p_h[1])*128]], axis=0) for p, a, h, p_h in
                                   zip(self.params[1:], self.angles[1:], self.heads[1:], self.heads[:-1])]
                        terminals = np.concatenate([np.asarray(states[1:])[:, -1] == 0.0, [True]])

                        train_states_to_save.extend(states)
                        train_actions_to_save.extend(actions)
                        train_next_states_to_save.extend(next_states)
                        train_terminals_to_save.extend(terminals)


        train_dict = {'actions': np.asarray(train_actions_to_save, dtype=np.float32),
                # 'infos/action_log_probs': np.ones((states.shape[0], 1),
                # 'infos/qpos': np.ones((states.shape[0], 1),
                # 'infos/qvel': np.ones((states.shape[0], 1),
                'next:observations': np.asarray(train_next_states_to_save, dtype=np.float32),
                'observations': np.asarray(train_states_to_save, dtype=np.float32),
                'rewards': np.ones((len(train_actions_to_save), 1)),
                'terminals': np.array(train_terminals_to_save),
                'timeouts': np.zeros((len(train_actions_to_save), 1)),
                }

        self._train_set = train_dict

        for [path, files] in np.asarray(traj_files)[test_indices]:
            print(path)
            if len(files) > 0 and files[0].endswith('.json'):
                self.states, self.coords, self.heads, self.angles, self.frames, self.params, self.comp_coords, self.comp_heads, self.comp_angl, self.comp_params = self._read_trajectory(
                    path, test=True)

                states = self.states[:-1]
                next_states = self.states[1:]
                actions = [np.concatenate([np.ravel(p), [a], [(h[0] - p_h[0]) * 102.4, (h[1] - p_h[1]) * 128]], axis=0)
                           for p, a, h, p_h in
                           zip(self.params[1:], self.angles[1:], self.heads[1:], self.heads[:-1])]
                terminals = np.concatenate([np.asarray(states[1:])[:, -1] == 0.0, [True]])

                test_states_to_save.extend(states)
                test_actions_to_save.extend(actions)
                test_next_states_to_save.extend(next_states)
                test_terminals_to_save.extend(terminals)

        test_dict = {'actions': np.asarray(test_actions_to_save, dtype=np.float32),
                # 'infos/action_log_probs': np.ones((states.shape[0], 1),
                # 'infos/qpos': np.ones((states.shape[0], 1),
                # 'infos/qvel': np.ones((states.shape[0], 1),
                'next:observations': np.asarray(test_next_states_to_save, dtype=np.float32),
                'observations': np.asarray(test_states_to_save, dtype=np.float32),
                'rewards': np.ones((len(test_actions_to_save), 1)),
                'terminals': np.asarray(test_terminals_to_save),
                'timeouts': np.zeros((len(test_actions_to_save), 1)),
                }

        self._test_set = test_dict

        global GET_TRAIN_SET
        if GET_TRAIN_SET:
            dict = self._train_set
            GET_TRAIN_SET = False
        else:
            dict = self._test_set
            GET_TRAIN_SET = True
        return dict


    def get_dataset_trajectories(self):
        n_augmentation = 1
        train_states_to_save = []
        train_actions_to_save = []
        train_next_states_to_save = []
        train_terminals_to_save = []
        test_states_to_save = []
        test_actions_to_save = []
        test_next_states_to_save = []
        test_terminals_to_save = []


        random.seed(1)
        np.random.seed(1)



        traj_files = []
        for path, directories, files in os.walk(self.data_file):
            print(path)
            if len(files) > 0 and files[0].endswith('.json'):
                traj_files.append([path, files])

        num_to_choose = int(len(traj_files) * 0.2)
        test_indices = np.random.choice(range(len(traj_files)), size=num_to_choose, replace=False)
        train_indices = np.setdiff1d(range(len(traj_files)), test_indices)
        for i in range(n_augmentation):
            for [path, files] in np.asarray(traj_files)[train_indices]:
                print(path)
                if len(files) > 0 and files[0].endswith('.json'):
                    self.states, self.coords, self.heads, self.angles, self.frames, self.params, self.comp_coords, self.comp_heads, self.comp_angl, self.comp_params = self._read_trajectory(
                        path)
                    states = self.states[:-1]
                    if len(states) > 0:
                        next_states = self.states[1:]
                        actions = [np.concatenate([np.ravel(p), [a], [(h[0]-p_h[0])*102.4, (h[1]-p_h[1])*128]], axis=0) for p, a, h, p_h in
                                   zip(self.params[1:], self.angles[1:], self.heads[1:], self.heads[:-1])]
                        terminals = np.concatenate([np.asarray(states[1:])[:, -1] == 0.0, [True]])

                        train_states_to_save.append(np.asarray(states, dtype=np.float32))
                        train_actions_to_save.append(np.asarray(actions, dtype=np.float32))
                        train_next_states_to_save.append(np.asarray(next_states, dtype=np.float32))
                        train_terminals_to_save.append(np.asarray(terminals, dtype=np.float32))


        train_dict = {'actions': train_actions_to_save,
                'next:observations': train_next_states_to_save,
                'observations': train_states_to_save,
                'rewards': np.ones((len(train_actions_to_save), 1)),
                'terminals': np.array(train_terminals_to_save),
                'timeouts': np.zeros((len(train_actions_to_save), 1)),
                }


        self._train_set = train_dict

        for [path, files] in np.asarray(traj_files)[test_indices]:
            print(path)
            if len(files) > 0 and files[0].endswith('.json'):
                self.states, self.coords, self.heads, self.angles, self.frames, self.params, self.comp_coords, self.comp_heads, self.comp_angl, self.comp_params = self._read_trajectory(
                    path, test=True)

                states = self.states[:-1]
                next_states = self.states[1:]
                actions = [np.concatenate([np.ravel(p), [a], [(h[0] - p_h[0]) * 102.4, (h[1] - p_h[1]) * 128]], axis=0)
                           for p, a, h, p_h in
                           zip(self.params[1:], self.angles[1:], self.heads[1:], self.heads[:-1])]
                terminals = np.concatenate([np.asarray(states[1:])[:, -1] == 0.0, [True]])

                test_states_to_save.append(np.asarray(states, dtype=np.float32))
                test_actions_to_save.append(np.asarray(actions, dtype=np.float32))
                test_next_states_to_save.append(np.asarray(next_states, dtype=np.float32))
                test_terminals_to_save.append(np.asarray(terminals, dtype=np.float32))

        test_dict = {'actions': test_actions_to_save,
                # 'infos/action_log_probs': np.ones((states.shape[0], 1),
                # 'infos/qpos': np.ones((states.shape[0], 1),
                # 'infos/qvel': np.ones((states.shape[0], 1),
                'next:observations': test_next_states_to_save,
                'observations': test_states_to_save,
                'rewards': np.ones((len(test_actions_to_save), 1)),
                'terminals': test_terminals_to_save,
                'timeouts': np.zeros((len(test_actions_to_save), 1)),
                }
        self._test_set = test_dict

        global GET_TRAIN_SET
        if GET_TRAIN_SET:
            dict = self._train_set
            GET_TRAIN_SET = False
        else:
            dict = self._test_set
            GET_TRAIN_SET = True
        return dict

class SingleSpermBezierIncrementsDataAugSimplified(SingleSpermBezierIncrements):

    # State: [Spline params, velocity vector, correction angle vector]
    # Use jointly with the SequenceDatasetSpermNormalized
    def __init__(self,
                 render_mode='rgb_array', data_file='diffuser/datasets/BezierSplinesData/slow'):
        self.timesteps = 0

        self.action_space = spaces.Box(low=-1, high=1., shape=(2,), dtype=np.float32)
        high = np.array([10. for i in range(14)]).astype(np.float32)
        low = np.array([-10. for i in range(14)]).astype(np.float32)
        self.observation_space = spaces.Box(low, high)

        self.data_file = data_file

        video = 'diffuser/datasets/test-1-29-field_1_30/frames'

        self.frames = self._read_frames(video)
        for path, directories, files in os.walk(self.data_file):
            print(path)
            if len(files) > 0 and files[0].endswith('.json'):
                self.states, self.coords, self.angles, self.frames, self.params, self.comp_coords, self.comp_angl, self.comp_params = self._read_trajectory(
                    path)
                break
        self._max_episode_steps = len(self.states) - 1
        self.max_timesteps = 15
        self.min_timesteps = 5
        self.step_counter = 0

        self.render_mode = render_mode

        self.current_coords = None
        self.current_coords1 = None
        self.current_coords2 = None
        # self.current_head = None
        self.current_angle = None
        self.parameters = None
        self.state0 = None
        self.state1 = None
        self.state2 = None

        self.accum_tail_error = 0.
        self.accum_tail_mom_error = 0.

        self._train_set = None
        self._test_set = None

    def _read_trajectory(self, spline_path, frames=None, num=1, test=False):
        def rotate2Dvec(v, theta):
            c, s = np.cos(theta), np.sin(theta)
            r = np.array(((c, -s), (s, c)))
            v = np.dot(r, v)
            return v

        states = []
        velocities = []
        angles = []
        params = []


        for path, directories, files in os.walk(spline_path):
            files.sort()

            first_iter = True
            second_iter = True
            old_head = [0.0, 0.0]

            real_head_list = []
            temp_states = []
            temp_coords = []
            temp_angles = []
            temp_params = []
            for f in files:
                if f.endswith('.json'):
                    json_data = read_json(os.path.join(path, f))
                    point_pairs = np.asarray(json_data['spline_line_space'])
                    parameters = np.asarray(json_data['spline_params'])/70.-1

                    point_pairs[:, 0] = point_pairs[:, 0]/70. - 1
                    point_pairs[:, 1] = point_pairs[:, 1]/70. - 1

                    if not test:
                        global_angle = json_data['correction_angle']  # (((json_data['correction_angle'] / 180) + 1) % 2) - 1.
                        head = json_data['head_coordinates']
                        head = [head[0], 1024-head[1], head[2]]
                        real_head_list.append(head)
                    else:
                        global_angle = json_data['correction_angle']  # (((json_data['correction_angle'] / 180) + 1) % 2) - 1.
                        head = json_data['head_coordinates']
                        head = [head[0], 1024-head[1], head[2]]
                        real_head_list.append(head)

                    if first_iter:
                        old_head[0] = head[0]
                        old_head[1] = head[1]
                        first_iter = False

                    else:

                        velocity = [float(head[0]-old_head[0]), float(head[1]-old_head[1])]

                        if second_iter:  # Fix the correction angle to put the traj in the horizontal axis
                            global_vector_angle = np.radians(vec2angle(velocity, normalize=False))
                            initial_global_angle = global_angle
                            second_iter = False

                        vector_angle = np.radians(vec2angle(velocity, normalize=False))
                        correction_angle = np.radians(global_angle)

                        diff_a = vector_angle - correction_angle
                        diff_a = (diff_a + np.pi) % (2 * np.pi) - np.pi


                        # # Normalize the vectors to the horizontal axis
                        # corr_vector_x = np.cos(correction_angle)
                        # corr_vector_y = np.sin(correction_angle)

                        old_head[0] = head[0]
                        old_head[1] = head[1]

                        # velocity_v = np.array(velocity)
                        velocity_v = rotate2Dvec(np.array(velocity), -global_vector_angle)

                        # temp_states.append(np.concatenate([np.ravel(parameters), velocity_v, [corr_vector_x, corr_vector_y]], axis=0))
                        # temp_corr_vector.append([corr_vector_x, corr_vector_y])
                        temp_coords.append(velocity_v)
                        temp_angles.append(diff_a)
                        temp_params.append(parameters)

            for p, a, c in zip(temp_params, temp_angles, temp_coords):
                # Normalize the vectors to the horizontal axis
                corr_vector_x = np.cos(a)
                corr_vector_y = np.sin(a)
                temp_states.append(np.concatenate([np.ravel(p), c, [corr_vector_x, corr_vector_y]], axis=0))

            states.extend(temp_states)
            velocities.extend(temp_coords)
            angles.extend(temp_angles)
            params.extend(temp_params)

        new_states = []
        new_velocity = []
        new_angles = []
        new_frames = []
        new_params = []
        for i in range(len(states)):
            new_states.append(states[i])  #np.concatenate([states[i][:-1], states[j][:-1]]))
            new_velocity.append(velocities[i])
            new_angles.append(angles[i])
            if frames is not None:
                new_frames.append(frames[i])
            new_params.append(params[i])

        check_angles = np.array([np.radians(vec2angle(angle, normalize=False)) for angle in np.array(new_states)[:, -4:-2]])
        for aux1, aux2 in zip(range(len(new_angles)), range(len(check_angles))):
            a = new_angles[aux1] - check_angles[aux2]
            value = (a + np.pi) % (2*np.pi) - np.pi
            if value > np.pi/2:  # If the difference between the two angles is too large, we check the problem and make a correction
                a = (new_angles[aux1] + np.pi) - check_angles[aux2]
                if a < np.pi/6:  # If the difference is lower than 30ª the angle is rotated 180ª.
                    new_angles[aux1] = new_angles[aux1] + np.pi
                    corr_vector_x = np.cos(new_angles[aux1])
                    corr_vector_y = np.sin(new_angles[aux1])
                    new_states[aux1][-2] = corr_vector_x
                    new_states[aux1][-1] = corr_vector_y

                else:
                    new_angles[aux1] = new_angles[aux1]
                    corr_vector_x = np.cos(new_angles[aux1])
                    corr_vector_y = np.sin(new_angles[aux1])
                    new_states[aux1][-2] = corr_vector_x
                    new_states[aux1][-1] = corr_vector_y
                    print('sperm.py simplified environment CHECK THIS ISSUE')
        return new_states, new_velocity, new_angles, new_frames, new_params, velocities, angles, params


    def get_dataset(self):
        n_augmentation = 1
        train_states_to_save = []
        train_actions_to_save = []
        train_next_states_to_save = []
        train_terminals_to_save = []
        test_states_to_save = []
        test_actions_to_save = []
        test_next_states_to_save = []
        test_terminals_to_save = []


        random.seed(1)
        np.random.seed(1)



        traj_files = []
        for path, directories, files in os.walk(self.data_file):
            print(path)
            if len(files) > 0 and files[0].endswith('.json'):
                traj_files.append([path, files])

        num_to_choose = int(len(traj_files) * 0.2)
        test_indices = np.random.choice(range(len(traj_files)), size=num_to_choose, replace=False)
        train_indices = np.setdiff1d(range(len(traj_files)), test_indices)
        for i in range(n_augmentation):
            for [path, files] in np.asarray(traj_files)[train_indices]:
                print(path)
                if len(files) > 0 and files[0].endswith('.json'):
                    self.states, self.velocities, self.angles, self.frames, self.params, self.comp_velocities, self.comp_angl, self.comp_params = self._read_trajectory(
                        path)
                    states = self.states[:-1]
                    if len(states) > 0:
                        next_states = self.states[1:]
                        actions = self.velocities[1:]
                        terminals = np.concatenate([np.asarray(states[1:])[:, -1] == 0.0, [True]])

                        train_states_to_save.extend(states)
                        train_actions_to_save.extend(actions)
                        train_next_states_to_save.extend(next_states)
                        train_terminals_to_save.extend(terminals)


        train_dict = {'actions': np.asarray(train_actions_to_save, dtype=np.float32),
                'next:observations': np.asarray(train_next_states_to_save, dtype=np.float32),
                'observations': np.asarray(train_states_to_save, dtype=np.float32),
                'rewards': np.ones((len(train_actions_to_save), 1)),
                'terminals': np.array(train_terminals_to_save),
                'timeouts': np.zeros((len(train_actions_to_save), 1)),
                }

        self._train_set = train_dict

        for [path, files] in np.asarray(traj_files)[test_indices]:
            print(path)
            if len(files) > 0 and files[0].endswith('.json'):
                self.states, self.velocities, self.angles, self.frames, self.params, self.comp_velocities, self.comp_angl, self.comp_params = self._read_trajectory(
                    path, test=True)

                states = self.states[:-1]
                next_states = self.states[1:]
                actions = self.velocities[1:]
                terminals = np.concatenate([np.asarray(states[1:])[:, -1] == 0.0, [True]])

                test_states_to_save.extend(states)
                test_actions_to_save.extend(actions)
                test_next_states_to_save.extend(next_states)
                test_terminals_to_save.extend(terminals)

        test_dict = {'actions': np.asarray(test_actions_to_save, dtype=np.float32),
                # 'infos/action_log_probs': np.ones((states.shape[0], 1),
                # 'infos/qpos': np.ones((states.shape[0], 1),
                # 'infos/qvel': np.ones((states.shape[0], 1),
                'next:observations': np.asarray(test_next_states_to_save, dtype=np.float32),
                'observations': np.asarray(test_states_to_save, dtype=np.float32),
                'rewards': np.ones((len(test_actions_to_save), 1)),
                'terminals': np.asarray(test_terminals_to_save),
                'timeouts': np.zeros((len(test_actions_to_save), 1)),
                }

        self._test_set = test_dict

        global GET_TRAIN_SET
        if GET_TRAIN_SET:
            dict = self._train_set
            GET_TRAIN_SET = False
        else:
            dict = self._test_set
            GET_TRAIN_SET = True
        return dict

    def get_dataset_trajectories(self):
        n_augmentation = 1
        train_states_to_save = []
        train_actions_to_save = []
        train_next_states_to_save = []
        train_terminals_to_save = []
        test_states_to_save = []
        test_actions_to_save = []
        test_next_states_to_save = []
        test_terminals_to_save = []


        random.seed(1)
        np.random.seed(1)



        traj_files = []
        for path, directories, files in os.walk(self.data_file):
            print(path)
            if len(files) > 0 and files[0].endswith('.json'):
                traj_files.append([path, files])

        num_to_choose = int(len(traj_files) * 0.2)
        test_indices = np.random.choice(range(len(traj_files)), size=num_to_choose, replace=False)
        train_indices = np.setdiff1d(range(len(traj_files)), test_indices)
        for i in range(n_augmentation):
            for [path, files] in np.asarray(traj_files)[train_indices]:
                print(path)
                if len(files) > 0 and files[0].endswith('.json'):
                    self.states, self.velocities, self.angles, self.frames, self.params, self.comp_velocities, self.comp_angl, self.comp_params = self._read_trajectory(
                        path)
                    states = self.states[:-1]
                    if len(states) > 0:
                        next_states = self.states[1:]
                        actions = self.velocities[1:]
                        terminals = np.concatenate([np.asarray(states[1:])[:, -1] == 0.0, [True]])

                        train_states_to_save.append(states)
                        train_actions_to_save.append(actions)
                        train_next_states_to_save.append(next_states)
                        train_terminals_to_save.append(terminals)


        train_dict = {'actions': np.asarray(train_actions_to_save, dtype=np.float32),
                'next:observations': np.asarray(train_next_states_to_save, dtype=np.float32),
                'observations': np.asarray(train_states_to_save, dtype=np.float32),
                'rewards': np.ones((len(train_actions_to_save), 1)),
                'terminals': np.array(train_terminals_to_save),
                'timeouts': np.zeros((len(train_actions_to_save), 1)),
                }

        self._train_set = train_dict

        for [path, files] in np.asarray(traj_files)[test_indices]:
            print(path)
            if len(files) > 0 and files[0].endswith('.json'):
                self.states, self.velocities, self.angles, self.frames, self.params, self.comp_velocities, self.comp_angl, self.comp_params = self._read_trajectory(
                    path, test=True)

                states = self.states[:-1]
                next_states = self.states[1:]
                actions = self.velocities[1:]
                terminals = np.concatenate([np.asarray(states[1:])[:, -1] == 0.0, [True]])

                test_states_to_save.append(np.asarray(states, dtype=np.float32))
                test_actions_to_save.append(np.asarray(actions, dtype=np.float32))
                test_next_states_to_save.append(np.asarray(next_states, dtype=np.float32))
                test_terminals_to_save.append(np.asarray(terminals, dtype=np.float32))

        test_dict = {'actions': test_actions_to_save,
                # 'infos/action_log_probs': np.ones((states.shape[0], 1),
                # 'infos/qpos': np.ones((states.shape[0], 1),
                # 'infos/qvel': np.ones((states.shape[0], 1),
                'next:observations': test_next_states_to_save,
                'observations': test_states_to_save,
                'rewards': np.ones((len(test_actions_to_save), 1)),
                'terminals': test_terminals_to_save,
                'timeouts': np.zeros((len(test_actions_to_save), 1)),
                }

        self._test_set = test_dict

        global GET_TRAIN_SET
        if GET_TRAIN_SET:
            dict = self._train_set
            GET_TRAIN_SET = False
        else:
            dict = self._test_set
            GET_TRAIN_SET = True
        return dict

class SingleSpermBezierIncrementsSynth(SingleSpermBezierIncrementsDataAug):
    def __init__(self, render_mode='rgb_array', data_file = 'diffuser/datasets/synthdata/2023-09-25:15-13'):
        super().__init__(render_mode=render_mode, data_file = data_file)

    def _read_trajectory(self, spline_path, frames=None, num=25, test=False):
        def displacement(coords, x_rand, y_rand):
            x_disp = x_rand - coords[0][0]
            y_disp = y_rand - coords[0][1]
            coords = np.asarray(coords)

            coords[:, 0] += x_disp
            coords[:, 1] += y_disp
            # coords[:, 0] = coords[:, 0]*2-1
            # coords[:, 1] = coords[:, 1] * 2 - 1
            return coords[-1, :]
        states = []
        coords = []
        heads = []
        angles = []
        params = []


        for path, directories, files in os.walk(spline_path):
            files.sort()

            first_iter = True
            old_head = [0.0, 0.0]

            rand_rot = random.random() * 2 - 1
            x_rand = ((random.random()) * 1280) #* 0.9 + 0.05
            y_rand = ((random.random()) * 1024) #* 0.9 + 0.05

            head_list = []
            real_head_list = []
            temp_states = []
            temp_coords = []
            temp_heads = []
            temp_angles = []
            temp_params = []
            for f in files:
                if f.endswith('.json'):
                    json_data = read_json(os.path.join(path, f))
                    point_pairs = np.asarray(json_data['spline_line_space'])
                    parameters = np.asarray(json_data['spline_params'])/70.-1

                    # point_pairs = np.concatenate([np.expand_dims(curve[:, 0], axis=-1), np.expand_dims(curve[:, 1], axis=-1)], axis=-1)
                    point_pairs[:, 0] = point_pairs[:, 0]/70. - 1
                    point_pairs[:, 1] = point_pairs[:, 1]/70. - 1

                    ravel_point_pairs = np.ravel(point_pairs)

                    angle = (((json_data['correction_angle'] / 180) + 1) % 2) - 1.

                    head = json_data['head_coordinates']
                    head = [head[0], 1024-head[1], head[2]]
                    real_head_list.append(head)

                    if first_iter:
                        old_head[0] = head[0]
                        old_head[1] = head[1]
                        first_iter = False
                    head_displacement = np.clip([float(head[0]-old_head[0])/20., float(head[1]-old_head[1])/20.], -1.0, 1.0)
                    old_head[0] = head[0]
                    old_head[1] = head[1]
                    head[0] = (head[0] / 1280)*2 - 1
                    head[1] = (head[1] / 1024)*2 - 1


                    temp_states.append(np.concatenate([ravel_point_pairs, np.ravel(parameters), head_displacement, [angle], head], axis=0))
                    temp_coords.append(point_pairs)
                    temp_heads.append(head)
                    temp_angles.append(angle)
                    temp_params.append(parameters)

                    # states.append(np.concatenate([ravel_point_pairs, np.ravel(parameters), head_displacement, [angle], head], axis=0))
                    # coords.append(point_pairs)
                    # heads.append(head)
                    # angles.append(angle)
                    # params.append(parameters)

            # if not np.any(np.array(temp_heads)[:, :-1] > 1) and not np.any(np.array(temp_heads)[:, :-1] < -1):
            states.extend(temp_states)
            coords.extend(temp_coords)
            heads.extend(temp_heads)
            angles.extend(temp_angles)
            params.extend(temp_params)

        new_states = []
        new_coords = []
        new_heads = []
        new_angles = []
        new_frames = []
        new_params = []
        for i in range(len(states)):
            new_states.append(states[i][:-1])  #np.concatenate([states[i][:-1], states[j][:-1]]))
            new_coords.append(coords[i])
            new_heads.append(heads[i])
            new_angles.append(angles[i])
            if frames is not None:
                new_frames.append(frames[i])
            new_params.append(params[i])
        return new_states, new_coords, new_heads, new_angles, new_frames, new_params, coords, heads, angles, params

    def get_dataset(self):
        n_augmentation = 1
        train_states_to_save = []
        train_actions_to_save = []
        train_next_states_to_save = []
        train_terminals_to_save = []
        test_states_to_save = []
        test_actions_to_save = []
        test_next_states_to_save = []
        test_terminals_to_save = []


        random.seed(1)
        np.random.seed(1)



        traj_files = []
        for path, directories, files in os.walk(self.data_file):
            print(path)
            if len(files) > 0 and files[0].endswith('.json'):
                traj_files.append([path, files])

        num_to_choose = int(len(traj_files) * 0.2)
        test_indices = np.random.choice(range(len(traj_files)), size=num_to_choose, replace=False)
        train_indices = np.setdiff1d(range(len(traj_files)), test_indices)
        for i in range(n_augmentation):
            for [path, files] in np.asarray(traj_files)[train_indices]:
                print(path)
                if len(files) > 0 and files[0].endswith('.json'):
                    self.states, self.coords, self.heads, self.angles, self.frames, self.params, self.comp_coords, self.comp_heads, self.comp_angl, self.comp_params = self._read_trajectory(
                        path)
                    states = self.states[:-1]
                    if len(states) > 0:
                        next_states = self.states[1:]
                        actions = [np.concatenate([np.ravel(p), [a], [(h[0]-p_h[0])*102.4, (h[1]-p_h[1])*128]], axis=0) for p, a, h, p_h in
                                   zip(self.params[1:], self.angles[1:], self.heads[1:], self.heads[:-1])]
                        terminals = np.concatenate([np.asarray(states[1:])[:, -1] == 0.0, [True]])

                        train_states_to_save.extend(states)
                        train_actions_to_save.extend(actions)
                        train_next_states_to_save.extend(next_states)
                        train_terminals_to_save.extend(terminals)


        train_dict = {'actions': np.asarray(train_actions_to_save, dtype=np.float32),
                # 'infos/action_log_probs': np.ones((states.shape[0], 1),
                # 'infos/qpos': np.ones((states.shape[0], 1),
                # 'infos/qvel': np.ones((states.shape[0], 1),
                'next:observations': np.asarray(train_next_states_to_save, dtype=np.float32),
                'observations': np.asarray(train_states_to_save, dtype=np.float32),
                'rewards': np.ones((len(train_actions_to_save), 1)),
                'terminals': np.array(train_terminals_to_save),
                'timeouts': np.zeros((len(train_actions_to_save), 1)),
                }

        self._train_set = train_dict

        for [path, files] in np.asarray(traj_files)[test_indices]:
            print(path)
            if len(files) > 0 and files[0].endswith('.json'):
                self.states, self.coords, self.heads, self.angles, self.frames, self.params, self.comp_coords, self.comp_heads, self.comp_angl, self.comp_params = self._read_trajectory(
                    path, test=True)

                if len(self.states) > 0:
                    states = self.states[:-1]
                    next_states = self.states[1:]
                    actions = [np.concatenate([np.ravel(p), [a], [(h[0] - p_h[0]) * 102.4, (h[1] - p_h[1]) * 128]], axis=0)
                               for p, a, h, p_h in
                               zip(self.params[1:], self.angles[1:], self.heads[1:], self.heads[:-1])]

                    try:
                        terminals = np.concatenate([np.asarray(states[1:])[:, -1] == 0.0, [True]])
                    except:
                        pass

                    test_states_to_save.extend(states)
                    test_actions_to_save.extend(actions)
                    test_next_states_to_save.extend(next_states)
                    test_terminals_to_save.extend(terminals)

        test_dict = {'actions': np.asarray(test_actions_to_save, dtype=np.float32),
                # 'infos/action_log_probs': np.ones((states.shape[0], 1),
                # 'infos/qpos': np.ones((states.shape[0], 1),
                # 'infos/qvel': np.ones((states.shape[0], 1),
                'next:observations': np.asarray(test_next_states_to_save, dtype=np.float32),
                'observations': np.asarray(test_states_to_save, dtype=np.float32),
                'rewards': np.ones((len(test_actions_to_save), 1)),
                'terminals': np.asarray(test_terminals_to_save),
                'timeouts': np.zeros((len(test_actions_to_save), 1)),
                }

        self._test_set = test_dict

        global GET_TRAIN_SET
        if GET_TRAIN_SET:
            dict = self._train_set
            GET_TRAIN_SET = False
        else:
            dict = self._test_set
            GET_TRAIN_SET = True
        return dict

    def get_dataset_trajectories(self):
        n_augmentation = 1
        train_states_to_save = []
        train_actions_to_save = []
        train_next_states_to_save = []
        train_terminals_to_save = []
        test_states_to_save = []
        test_actions_to_save = []
        test_next_states_to_save = []
        test_terminals_to_save = []


        random.seed(1)
        np.random.seed(1)



        traj_files = []
        for path, directories, files in os.walk(self.data_file):
            print(path)
            if len(files) > 0 and files[0].endswith('.json'):
                traj_files.append([path, files])

        num_to_choose = int(len(traj_files) * 0.0)
        test_indices = np.random.choice(range(len(traj_files)), size=num_to_choose, replace=False)
        train_indices = np.setdiff1d(range(len(traj_files)), test_indices)
        for i in range(n_augmentation):
            for [path, files] in np.asarray(traj_files)[train_indices]:
                print(path)
                if len(files) > 0 and files[0].endswith('.json'):
                    self.states, self.coords, self.heads, self.angles, self.frames, self.params, self.comp_coords, self.comp_heads, self.comp_angl, self.comp_params = self._read_trajectory(
                        path)
                    states = self.states[:-1]
                    if len(states) > 0:
                        next_states = self.states[1:]
                        actions = [np.concatenate([np.ravel(p), [a], [(h[0]-p_h[0])*102.4, (h[1]-p_h[1])*128]], axis=0) for p, a, h, p_h in
                                   zip(self.params[1:], self.angles[1:], self.heads[1:], self.heads[:-1])]
                        terminals = np.concatenate([np.asarray(states[1:])[:, -1] == 0.0, [True]])

                        train_states_to_save.append(np.asarray(states, dtype=np.float32))
                        train_actions_to_save.append(np.asarray(actions, dtype=np.float32))
                        train_next_states_to_save.append(np.asarray(next_states, dtype=np.float32))
                        train_terminals_to_save.append(np.asarray(terminals, dtype=np.float32))


        train_dict = {'actions': train_actions_to_save,
                # 'infos/action_log_probs': np.ones((states.shape[0], 1),
                # 'infos/qpos': np.ones((states.shape[0], 1),
                # 'infos/qvel': np.ones((states.shape[0], 1),
                'next:observations': train_next_states_to_save,
                'observations': train_states_to_save,
                'rewards': np.ones((len(train_actions_to_save), 1)),
                'terminals': np.array(train_terminals_to_save),
                'timeouts': np.zeros((len(train_actions_to_save), 1)),
                }

        self._train_set = train_dict

        for [path, files] in np.asarray(traj_files)[test_indices]:
            print(path)
            if len(files) > 0 and files[0].endswith('.json'):
                self.states, self.coords, self.heads, self.angles, self.frames, self.params, self.comp_coords, self.comp_heads, self.comp_angl, self.comp_params = self._read_trajectory(
                    path, test=True)

                if len(self.states) > 0:
                    states = self.states[:-1]
                    next_states = self.states[1:]
                    actions = [np.concatenate([np.ravel(p), [a], [(h[0] - p_h[0]) * 102.4, (h[1] - p_h[1]) * 128]], axis=0)
                               for p, a, h, p_h in
                               zip(self.params[1:], self.angles[1:], self.heads[1:], self.heads[:-1])]

                    try:
                        terminals = np.concatenate([np.asarray(states[1:])[:, -1] == 0.0, [True]])
                    except:
                        pass

                    test_states_to_save.append(np.asarray(states, dtype=np.float32))
                    test_actions_to_save.append(np.asarray(actions, dtype=np.float32))
                    test_next_states_to_save.append(np.asarray(next_states, dtype=np.float32))
                    test_terminals_to_save.append(np.asarray(terminals, dtype=np.float32))

        test_dict = {'actions': test_actions_to_save,
                # 'infos/action_log_probs': np.ones((states.shape[0], 1),
                # 'infos/qpos': np.ones((states.shape[0], 1),
                # 'infos/qvel': np.ones((states.shape[0], 1),
                'next:observations': test_next_states_to_save,
                'observations': test_states_to_save,
                'rewards': np.ones((len(test_actions_to_save), 1)),
                'terminals': test_terminals_to_save,
                'timeouts': np.zeros((len(test_actions_to_save), 1)),
                }

        self._test_set = test_dict

        global GET_TRAIN_SET
        if GET_TRAIN_SET:
            dict = self._train_set
            GET_TRAIN_SET = False
        else:
            dict = self._test_set
            GET_TRAIN_SET = True
        return dict

class SingleSpermBezierIncrementsDataAugMulticlass(SingleSpermBezierIncrementsDataAug):
    def __init__(self,
                 render_mode='rgb_array',
                 data_file=['diffuser/datasets/BezierSplinesData/moving', 'diffuser/datasets/BezierSplinesData/slow', 'diffuser/datasets/BezierSplinesData/stopped']):
        self.timesteps = 0
        self.n_cat = len(data_file)
        self.action_space = spaces.Box(low=-1, high=1., shape=(13,), dtype=np.float32)
        high = np.array([10. for i in range(108)]).astype(np.float32)
        low = np.array([-10. for i in range(108)]).astype(np.float32)
        self.observation_space = spaces.Box(low, high)

        self.data_file = data_file

        video = 'diffuser/datasets/test-1-29-field_1_30/frames'

        self.frames = self._read_frames(video)

        self.states = []
        self.coords = []
        self.heads = []
        self.angles = []
        self.frames = []
        self.params = []
        self.comp_coords = []
        self.comp_heads = []
        self.comp_angl = []
        self.comp_params = []

        for data_files, cat in zip(self.data_file, range(len(self.data_file))):
            for path, directories, files in os.walk(data_files):
                print(path)
                if len(files) > 0 and files[0].endswith('.json'):
                    states, coords, heads, angles, frames, params, comp_coords, comp_heads, comp_angl, comp_params = self._read_trajectory(path, cat)
                    self.states.extend(states)
                    self.coords.extend(coords)
                    self.heads.extend(heads)
                    self.angles.extend(angles)
                    self.frames.extend(frames)
                    self.params.extend(params)
                    self.comp_coords.extend(comp_coords)
                    self.comp_heads.extend(comp_heads)
                    self.comp_angl.extend(comp_angl)
                    self.comp_params.extend(comp_params)
                    break

        self._max_episode_steps = len(self.states)-1
        self.max_timesteps = 15
        self.min_timesteps = 5
        self.step_counter = 0

        self.render_mode = render_mode

        self.current_coords = None
        self.current_coords1 = None
        self.current_coords2 = None
        self.current_head = None
        self.current_angle = None
        self.parameters = None
        self.state0 = None
        self.state1 = None
        self.state2 = None

        self.accum_tail_error = 0.
        self.accum_tail_mom_error = 0.

        self._train_set = None
        self._test_set = None

    def _read_trajectory(self, spline_path, category, frames=None, num=25, test=False):
        def displacement(coords, x_rand, y_rand):
            x_disp = x_rand - coords[0][0]
            y_disp = y_rand - coords[0][1]
            coords = np.asarray(coords)

            coords[:, 0] += x_disp
            coords[:, 1] += y_disp
            # coords[:, 0] = coords[:, 0]*2-1
            # coords[:, 1] = coords[:, 1] * 2 - 1
            return coords[-1, :]
        states = []
        coords = []
        heads = []
        angles = []
        params = []


        for path, directories, files in os.walk(spline_path):
            files.sort()

            first_iter = True
            old_head = [0.0, 0.0]

            rand_rot = random.random() * 2 - 1
            x_rand = ((random.random()) * 1280) #* 0.9 + 0.05
            y_rand = ((random.random()) * 1024) #* 0.9 + 0.05

            head_list = []
            real_head_list = []
            temp_states = []
            temp_coords = []
            temp_heads = []
            temp_angles = []
            temp_params = []
            for f in files:
                if f.endswith('.json'):
                    json_data = read_json(os.path.join(path, f))
                    point_pairs = np.asarray(json_data['spline_line_space'])
                    parameters = np.asarray(json_data['spline_params'])/70.-1

                    # point_pairs = np.concatenate([np.expand_dims(curve[:, 0], axis=-1), np.expand_dims(curve[:, 1], axis=-1)], axis=-1)
                    point_pairs[:, 0] = point_pairs[:, 0]/70. - 1
                    point_pairs[:, 1] = point_pairs[:, 1]/70. - 1

                    ravel_point_pairs = np.ravel(point_pairs)

                    if not test:
                        angle = (((json_data['correction_angle']/180 + rand_rot)+1) % 2) -1.

                        head = json_data['head_coordinates']
                        head = [head[0], 1024-head[1], head[2]]
                        real_head_list.append(head)
                        rot = mpl.transforms.Affine2D().rotate_deg(rand_rot * 180)
                        rot_heads = rot.transform(np.asarray(head[:-1]))
                        head_list.append(rot_heads)

                        head = [*displacement(np.asarray(head_list), x_rand, y_rand), head[-1]]
                    else:
                        angle = (((json_data['correction_angle'] / 180) + 1) % 2) - 1.

                        head = json_data['head_coordinates']
                        head = [head[0], 1024-head[1], head[2]]
                        real_head_list.append(head)

                    if first_iter:
                        old_head[0] = head[0]
                        old_head[1] = head[1]
                        first_iter = False
                    head_displacement = np.clip([float(head[0]-old_head[0])/20., float(head[1]-old_head[1])/20.], -1.0, 1.0)
                    old_head[0] = head[0]
                    old_head[1] = head[1]
                    head[0] = (head[0] / 1280)*2 - 1
                    head[1] = (head[1] / 1024)*2 - 1


                    temp_states.append(np.concatenate([ravel_point_pairs, np.ravel(parameters), [category/self.n_cat], head_displacement, [angle], head], axis=0))
                    temp_coords.append(point_pairs)
                    temp_heads.append(head)
                    temp_angles.append(angle)
                    temp_params.append(parameters)

                    # states.append(np.concatenate([ravel_point_pairs, np.ravel(parameters), head_displacement, [angle], head], axis=0))
                    # coords.append(point_pairs)
                    # heads.append(head)
                    # angles.append(angle)
                    # params.append(parameters)

            if not np.any(np.array(temp_heads)[:, :-1] > 1) and not np.any(np.array(temp_heads)[:, :-1] < -1):
                states.extend(temp_states)
                coords.extend(temp_coords)
                heads.extend(temp_heads)
                angles.extend(temp_angles)
                params.extend(temp_params)

        new_states = []
        new_coords = []
        new_heads = []
        new_angles = []
        new_frames = []
        new_params = []
        for i in range(len(states)):
            new_states.append(states[i][:-1])  #np.concatenate([states[i][:-1], states[j][:-1]]))
            new_coords.append(coords[i])
            new_heads.append(heads[i])
            new_angles.append(angles[i])
            if frames is not None:
                new_frames.append(frames[i])
            new_params.append(params[i])
        return new_states, new_coords, new_heads, new_angles, new_frames, new_params, coords, heads, angles, params

    def get_dataset(self):
        n_augmentation = 25
        train_states_to_save = []
        train_actions_to_save = []
        train_next_states_to_save = []
        train_terminals_to_save = []
        test_states_to_save = []
        test_actions_to_save = []
        test_next_states_to_save = []
        test_terminals_to_save = []


        random.seed(1)
        np.random.seed(1)



        traj_files = []
        for data_files, cat in zip(self.data_file, range(len(self.data_file))):
            for path, directories, files in os.walk(data_files):
                print(path)
                if len(files) > 0 and files[0].endswith('.json'):
                    traj_files.append([path, files, cat])

        num_to_choose = int(len(traj_files) * 0.2)
        test_indices = np.random.choice(range(len(traj_files)), size=num_to_choose, replace=False)
        train_indices = np.setdiff1d(range(len(traj_files)), test_indices)
        for i in range(n_augmentation):
            for [path, files, cat] in np.asarray(traj_files)[train_indices]:
                print(path)
                if len(files) > 0 and files[0].endswith('.json'):

                    self.states, self.coords, self.heads, self.angles, self.frames, self.params, self.comp_coords, self.comp_heads, self.comp_angl, self.comp_params = self._read_trajectory(
                        path, cat)
                    states = self.states[:-1]
                    if len(states) > 0:
                        next_states = self.states[1:]
                        actions = [np.concatenate([np.ravel(p), [a], [(h[0]-p_h[0])*102.4, (h[1]-p_h[1])*128]], axis=0) for p, a, h, p_h in
                                   zip(self.params[1:], self.angles[1:], self.heads[1:], self.heads[:-1])]
                        terminals = np.concatenate([np.asarray(states[1:])[:, -1] == 0.0, [True]])

                        train_states_to_save.extend(states)
                        train_actions_to_save.extend(actions)
                        train_next_states_to_save.extend(next_states)
                        train_terminals_to_save.extend(terminals)


        train_dict = {'actions': np.asarray(train_actions_to_save, dtype=np.float32),
                # 'infos/action_log_probs': np.ones((states.shape[0], 1),
                # 'infos/qpos': np.ones((states.shape[0], 1),
                # 'infos/qvel': np.ones((states.shape[0], 1),
                'next:observations': np.asarray(train_next_states_to_save, dtype=np.float32),
                'observations': np.asarray(train_states_to_save, dtype=np.float32),
                'rewards': np.ones((len(train_actions_to_save), 1)),
                'terminals': np.array(train_terminals_to_save),
                'timeouts': np.zeros((len(train_actions_to_save), 1)),
                }

        self._train_set = train_dict

        for [path, files, cat] in np.asarray(traj_files)[test_indices]:
            print(path)
            if len(files) > 0 and files[0].endswith('.json'):
                self.states, self.coords, self.heads, self.angles, self.frames, self.params, self.comp_coords, self.comp_heads, self.comp_angl, self.comp_params = self._read_trajectory(
                    path, cat, test=True)

                states = self.states[:-1]
                next_states = self.states[1:]
                actions = [np.concatenate([np.ravel(p), [a], [(h[0] - p_h[0]) * 102.4, (h[1] - p_h[1]) * 128]], axis=0)
                           for p, a, h, p_h in
                           zip(self.params[1:], self.angles[1:], self.heads[1:], self.heads[:-1])]
                terminals = np.concatenate([np.asarray(states[1:])[:, -1] == 0.0, [True]])

                test_states_to_save.extend(states)
                test_actions_to_save.extend(actions)
                test_next_states_to_save.extend(next_states)
                test_terminals_to_save.extend(terminals)

        test_dict = {'actions': np.asarray(test_actions_to_save, dtype=np.float32),
                # 'infos/action_log_probs': np.ones((states.shape[0], 1),
                # 'infos/qpos': np.ones((states.shape[0], 1),
                # 'infos/qvel': np.ones((states.shape[0], 1),
                'next:observations': np.asarray(test_next_states_to_save, dtype=np.float32),
                'observations': np.asarray(test_states_to_save, dtype=np.float32),
                'rewards': np.ones((len(test_actions_to_save), 1)),
                'terminals': np.asarray(test_terminals_to_save),
                'timeouts': np.zeros((len(test_actions_to_save), 1)),
                }

        self._test_set = test_dict

        global GET_TRAIN_SET
        if GET_TRAIN_SET:
            dict = self._train_set
            GET_TRAIN_SET = False
        else:
            dict = self._test_set
            GET_TRAIN_SET = True
        return dict

def read_json(path):
    with open(path, 'r') as openfile:
        # Reading from json file
        json_object = json.load(openfile)
    json_object = ast.literal_eval(json_object)
    return json_object