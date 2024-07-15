import os
import numpy as np
import einops
import imageio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import gym
import mujoco_py as mjc
import warnings
import pdb

from .arrays import to_np
from .video import save_video, save_videos

from diffuser.datasets.d4rl import load_environment
from ..environments.utils.sperm_rendering import *


#-----------------------------------------------------------------------------#
#------------------------------- helper structs ------------------------------#
#-----------------------------------------------------------------------------#

def env_map(env_name):
    '''
        map D4RL dataset names to custom fully-observed
        variants for rendering
    '''
    if 'halfcheetah' in env_name:
        return 'HalfCheetahFullObs-v2'
    elif 'hopper' in env_name:
        return 'HopperFullObs-v2'
    elif 'walker2d' in env_name:
        return 'Walker2dFullObs-v2'
    else:
        return env_name

#-----------------------------------------------------------------------------#
#------------------------------ helper functions -----------------------------#
#-----------------------------------------------------------------------------#

def get_image_mask(img):
    background = (img == 255).all(axis=-1, keepdims=True)
    mask = ~background.repeat(3, axis=-1)
    return mask

def atmost_2d(x):
    while x.ndim > 2:
        x = x.squeeze(0)
    return x

#-----------------------------------------------------------------------------#
#---------------------------------- renderers --------------------------------#
#-----------------------------------------------------------------------------#

class MuJoCoRenderer:
    '''
        default mujoco renderer
    '''

    def __init__(self, env):
        if type(env) is str:
            env = env_map(env)
            self.env = gym.make(env)
        else:
            self.env = env
        ## - 1 because the envs in renderer are fully-observed
        self.observation_dim = np.prod(self.env.observation_space.shape) - 1
        self.action_dim = np.prod(self.env.action_space.shape)
        print('[ utils/rendering ] Warning: could not initialize offscreen renderer')
        self.viewer = None

    def pad_observation(self, observation):
        state = np.concatenate([
            np.zeros(1),
            observation,
        ])
        return state

    def pad_observations(self, observations):
        qpos_dim = self.env.sim.data.qpos.size
        ## xpos is hidden
        xvel_dim = qpos_dim - 1
        xvel = observations[:, xvel_dim]
        xpos = np.cumsum(xvel) * self.env.dt
        states = np.concatenate([
            xpos[:,None],
            observations,
        ], axis=-1)
        return states

    def render(self, observation, dim=256, partial=False, qvel=True, render_kwargs=None, conditions=None):

        if type(dim) == int:
            dim = (dim, dim)

        if self.viewer is None:
            return np.zeros((*dim, 3), np.uint8)

        if render_kwargs is None:
            xpos = observation[0] if not partial else 0
            render_kwargs = {
                'trackbodyid': 2,
                'distance': 3,
                'lookat': [xpos, -0.5, 1],
                'elevation': -20
            }

        for key, val in render_kwargs.items():
            if key == 'lookat':
                self.viewer.cam.lookat[:] = val[:]
            else:
                setattr(self.viewer.cam, key, val)

        if partial:
            state = self.pad_observation(observation)
        else:
            state = observation

        qpos_dim = self.env.sim.data.qpos.size
        if not qvel or state.shape[-1] == qpos_dim:
            qvel_dim = self.env.sim.data.qvel.size
            state = np.concatenate([state, np.zeros(qvel_dim)])

        set_state(self.env, state)

        self.viewer.render(*dim)
        data = self.viewer.read_pixels(*dim, depth=False)
        data = data[::-1, :, :]
        return data

    def _renders(self, observations, **kwargs):
        images = []
        for observation in observations:
            img = self.render(observation, **kwargs)
            images.append(img)
        return np.stack(images, axis=0)

    def renders(self, samples, partial=False, **kwargs):
        if partial:
            samples = self.pad_observations(samples)
            partial = False

        sample_images = self._renders(samples, partial=partial, **kwargs)

        composite = np.ones_like(sample_images[0]) * 255

        for img in sample_images:
            mask = get_image_mask(img)
            composite[mask] = img[mask]

        return composite

    def composite(self, savepath, paths, real_paths=None, dim=(1024, 256), **kwargs):

        render_kwargs = {
            'trackbodyid': 2,
            'distance': 10,
            'lookat': [5, 2, 0.5],
            'elevation': 0
        }
        images = []
        for path in paths:
            ## [ H x obs_dim ]
            path = atmost_2d(path)
            # TODO: # comprobar como sustituir esto por mi modelo de splines
            try:
                if path.shape[1] == 14:  # If len(state)=12 -> SingleSpermBezierIncrementsDataAugSimplified
                    img = sperm_traj_simp_render(to_np(path))

                    if real_paths is not None:
                        real_paths = atmost_2d(real_paths)
                        real_img = sperm_traj_simp_render(to_np(real_paths))
                elif path.shape[1] == 12:  # If len(state)=12 -> SingleSpermBezierIncrementsDataAugSimplified2
                    img = sperm_traj_simp_render2(to_np(path))

                    if real_paths is not None:
                        real_paths = atmost_2d(real_paths)
                        real_img = sperm_traj_simp_render2(to_np(real_paths))
                else:
                    img = sperm_traj_render(to_np(path))

                    if real_paths is not None:
                        real_paths = atmost_2d(real_paths)
                        real_img = sperm_traj_render(to_np(real_paths))
            except:
                img = self.renders(to_np(path), dim=dim, partial=True, qvel=True, render_kwargs=render_kwargs, **kwargs)

            if real_paths is not None:
                images.append(np.concatenate([real_img, img], axis=0))
            else:
                images.append(img)

        images = np.concatenate(images, axis=0)

        if savepath is not None:
            imageio.imsave(savepath, images)
            print(f'Saved {len(paths)} samples to: {savepath}')

        return images

    def composite_w_displacement(self, savepath, paths, real_paths=None, dim=(1024, 256), **kwargs):
        images = []
        for path in paths:
            ## [ H x obs_dim ]
            path = atmost_2d(path)
            # TODO: # comprobar como sustituir esto por mi modelo de splines
            # try:
            img = sperm_traj_render_w_displacement(to_np(path))

            if real_paths is not None:
                real_paths = atmost_2d(real_paths)
                real_img = sperm_traj_render_w_displacement(to_np(real_paths))

            # except:
            #     img = self.renders(to_np(path), dim=dim, partial=True, qvel=True, render_kwargs=render_kwargs, **kwargs)
            if real_paths is not None:
                images.append(np.concatenate([real_img, img], axis=0))
            else:
                images.append(img)

        images = np.concatenate(images, axis=0)

        if savepath is not None:
            imageio.imsave(savepath, images)
            print(f'Saved {len(paths)} samples to: {savepath}')

        return images


    def composite_full_image(self, savepath, paths, dim=(1024, 1280), resize=(512, 640), ext='.png', **kwargs):
        images = []
        images_debug = []
        recorded_positions = [[] for _ in range(to_np(paths[:, 0].shape[0]))]
        img_debug, position1, recorded_positions = sperm_traj_render_full_image_debug(to_np(paths[:, 0]), dim=dim, resize=resize, recorded_positions=recorded_positions)
        img, position2 = sperm_traj_render_full_image(to_np(paths[:, 0]), dim=dim, resize=resize)

        images_debug.append(img_debug)
        images.append(img)

        for i in range(1, paths.shape[1]):
            img_debug, position1, recorded_positions = sperm_traj_render_full_image_debug(to_np(paths[:, i]), dim=dim, resize=resize, old_coords=position1, recorded_positions=recorded_positions)
            img, position2 = sperm_traj_render_full_image(to_np(paths[:, i]), dim=dim, resize=resize, old_coords=position2)

            images_debug.append(img_debug)
            images.append(img)

        if savepath is not None:
            images_debug_aux = np.concatenate(images_debug, axis=1)
            images_aux = np.concatenate(images, axis=1)

            imageio.imsave(savepath + ext, images_aux)
            print(f'Saved {len(paths)} samples to: ', savepath + ext)

            imageio.imsave(savepath + '_debug' + ext, images_debug_aux)
            print(f'Saved {len(paths)} samples to: ', savepath + '_debug' + ext)

        return images, images_debug


    def render_rollout(self, savepath, states, **video_kwargs):
        if type(states) is list: states = np.array(states)
        images = self._renders(states, partial=True)
        save_video(savepath, images, **video_kwargs)

    def render_plan(self, savepath, actions, observations_pred, state, fps=30):
        ## [ batch_size x horizon x observation_dim ]
        observations_real = rollouts_from_state(self.env, state, actions)

        ## there will be one more state in `observations_real`
        ## than in `observations_pred` because the last action
        ## does not have an associated next_state in the sampled trajectory
        observations_real = observations_real[:,:-1]

        images_pred = np.stack([
            self._renders(obs_pred, partial=True)
            for obs_pred in observations_pred
        ])

        images_real = np.stack([
            self._renders(obs_real, partial=False)
            for obs_real in observations_real
        ])

        ## [ batch_size x horizon x H x W x C ]
        images = np.concatenate([images_pred, images_real], axis=-2)
        save_videos(savepath, *images)

    def render_diffusion(self, savepath, diffusion_path, **video_kwargs):
        '''
            diffusion_path : [ n_diffusion_steps x batch_size x 1 x horizon x joined_dim ]
        '''
        render_kwargs = {
            'trackbodyid': 2,
            'distance': 10,
            'lookat': [10, 2, 0.5],
            'elevation': 0,
        }

        diffusion_path = to_np(diffusion_path)

        n_diffusion_steps, batch_size, _, horizon, joined_dim = diffusion_path.shape

        frames = []
        for t in reversed(range(n_diffusion_steps)):
            print(f'[ utils/renderer ] Diffusion: {t} / {n_diffusion_steps}')

            ## [ batch_size x horizon x observation_dim ]
            states_l = diffusion_path[t].reshape(batch_size, horizon, joined_dim)[:, :, :self.observation_dim]

            frame = []
            for states in states_l:
                img = self.composite(None, states, dim=(1024, 256), partial=True, qvel=True, render_kwargs=render_kwargs)
                frame.append(img)
            frame = np.concatenate(frame, axis=0)

            frames.append(frame)

        save_video(savepath, frames, **video_kwargs)

    def __call__(self, *args, **kwargs):
        return self.renders(*args, **kwargs)

class EMARenderer(MuJoCoRenderer):

    def composite_w_displacement(self, savepath, paths, real_paths=None, dim=(1024, 256), **kwargs):
        images = []
        for path in paths:
            ## [ H x obs_dim ]
            path = atmost_2d(path)
            # TODO: # comprobar como sustituir esto por mi modelo de splines
            # try:
            if path.shape[1] == 14:  # If len(state)=12 -> SingleSpermBezierIncrementsDataAugSimplified
                img = sperm_traj_simp_render(to_np(path))

                if real_paths is not None:
                    real_paths = atmost_2d(real_paths)
                    real_img = sperm_traj_simp_render(to_np(real_paths))
            elif path.shape[1] == 12:  # If len(state)=12 -> SingleSpermBezierIncrementsDataAugSimplified2
                img = sperm_traj_simp_render2(to_np(path))

                if real_paths is not None:
                    real_paths = atmost_2d(real_paths)
                    real_img = sperm_traj_simp_render2(to_np(real_paths))
            else:
                img = sperm_traj_render_w_displacementEMA(to_np(path))

                if real_paths is not None:
                    real_paths = atmost_2d(real_paths)
                    real_img = sperm_traj_render_w_displacementEMA(to_np(real_paths))

            # except:
            #     img = self.renders(to_np(path), dim=dim, partial=True, qvel=True, render_kwargs=render_kwargs, **kwargs)
            if real_paths is not None:
                images.append(np.concatenate([real_img, img], axis=0))
            else:
                images.append(img)

        images = np.concatenate(images, axis=0)

        if savepath is not None:
            imageio.imsave(savepath, images)
            print(f'Saved {len(paths)} samples to: {savepath}')

        return images

    def _ema(self, pred):
        alpha = 0.4
        ema = np.zeros((pred.shape[0], pred.shape[1]), dtype=np.float32)

        ema[:, 0] = pred[:, 0]

        for i in range(1, pred.shape[1]):
            ema[:, i] = (alpha * pred[:, i] + (1 - alpha) * ema[:, i - 1])

        return ema

    def composite_full_image(self, savepath, paths, dim=(1024, 1280), resize=(512, 640), ext='.png', use_ema=True, **kwargs):
        images = []
        images_debug = []
        recorded_positions = [[] for _ in range(to_np(paths[:, 0].shape[0]))]

        np_paths = to_np(paths)

        if use_ema:
            ema_angles = self._ema(to_np(paths[:, :, -3]))
            np_paths[:, :, -3] = ema_angles

        img_debug, position1, recorded_positions = sperm_traj_render_full_image_debug(np_paths[:, 0], dim=dim, resize=resize, recorded_positions=recorded_positions)
        img, position2 = sperm_traj_render_full_image(np_paths[:, 0], dim=dim, resize=resize)
        # font

        images_debug.append(img_debug)
        images.append(img)

        for i in range(1, paths.shape[1]):
            img_debug, position1, recorded_positions = sperm_traj_render_full_image_debug(np_paths[:, i], dim=dim, resize=resize, old_coords=position1, recorded_positions=recorded_positions)
            img, position2 = sperm_traj_render_full_image(np_paths[:, i], dim=dim, resize=resize, old_coords=position2)

            images_debug.append(img_debug)
            images.append(img)

        if savepath is not None:
            images_debug_aux = np.concatenate(images_debug, axis=1)
            images_aux = np.concatenate(images, axis=1)

            imageio.imsave(savepath + ext, images_aux)
            print(f'Saved {len(paths)} samples to: ', savepath + ext)

            imageio.imsave(savepath + '_debug' + ext, images_debug_aux)
            print(f'Saved {len(paths)} samples to: ', savepath + '_debug' + ext)

        return images, images_debug


    # def composite_full_image_simplified(self, savepath, paths, dim=(1024, 1280), resize=(512, 640), ext='.png', use_ema=True, **kwargs):
    #     images = []
    #     images_debug = []
    #     recorded_positions = [[] for _ in range(to_np(paths[:, 0].shape[0]))]
    #
    #     np_paths = to_np(paths)
    #     if use_ema:
    #         ema_angles = self._ema(to_np(paths[:, :, -3]))
    #         np_paths[:, :, -3] = ema_angles
    #     img_debug, position1, recorded_positions = sperm_traj_render_full_image_debug(np_paths[:, 0], dim=dim, resize=resize, recorded_positions=recorded_positions)
    #     img, position2 = sperm_traj_render_full_image(np_paths[:, 0], dim=dim, resize=resize)
    #
    #     images_debug.append(img_debug)
    #     images.append(img)
    #
    #     for i in range(1, paths.shape[1]):
    #         img_debug, position1, recorded_positions = sperm_traj_render_full_image_debug(np_paths[:, i], dim=dim, resize=resize, old_coords=position1, recorded_positions=recorded_positions)
    #         img, position2 = sperm_traj_render_full_image(np_paths[:, i], dim=dim, resize=resize, old_coords=position2)
    #
    #         images_debug.append(img_debug)
    #         images.append(img)
    #
    #     if savepath is not None:
    #         images_debug_aux = np.concatenate(images_debug, axis=1)
    #         images_aux = np.concatenate(images, axis=1)
    #
    #         imageio.imsave(savepath + ext, images_aux)
    #         print(f'Saved {len(paths)} samples to: ', savepath + ext)
    #
    #         imageio.imsave(savepath + '_debug' + ext, images_debug_aux)
    #         print(f'Saved {len(paths)} samples to: ', savepath + '_debug' + ext)
    #
    #     return images, images_debug


#-----------------------------------------------------------------------------#
#---------------------------------- rollouts ---------------------------------#
#-----------------------------------------------------------------------------#

def set_state(env, state):
    qpos_dim = env.sim.data.qpos.size
    qvel_dim = env.sim.data.qvel.size
    if not state.size == qpos_dim + qvel_dim:
        warnings.warn(
            f'[ utils/rendering ] Expected state of size {qpos_dim + qvel_dim}, '
            f'but got state of size {state.size}')
        state = state[:qpos_dim + qvel_dim]

    env.set_state(state[:qpos_dim], state[qpos_dim:])

def rollouts_from_state(env, state, actions_l):
    rollouts = np.stack([
        rollout_from_state(env, state, actions)
        for actions in actions_l
    ])
    return rollouts

def rollout_from_state(env, state, actions):
    qpos_dim = env.sim.data.qpos.size
    env.set_state(state[:qpos_dim], state[qpos_dim:])
    observations = [env._get_obs()]
    for act in actions:
        obs, rew, term, _ = env.step(act)
        observations.append(obs)
        if term:
            break
    for i in range(len(observations), len(actions)+1):
        ## if terminated early, pad with zeros
        observations.append( np.zeros(obs.size) )
    return np.stack(observations)
