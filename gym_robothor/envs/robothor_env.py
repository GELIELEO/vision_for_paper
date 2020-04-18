"""
Base class implementation for ai2thor environments wrapper, which adds an openAI gym interface for
inheriting the predefined methods and can be extended for particular tasks.
"""

import os

import ai2thor.controller
import numpy as np
from skimage import transform
from collections import defaultdict

import gym
from gym import error, spaces
from gym.utils import seeding
from gym_robothor.image_processing import rgb2gray
from gym_robothor.utils import read_config
import gym_robothor.tasks

import torch
from gym_robothor.visualpriors.transforms import VisualPrior 
import json
import cv2
import math
import random

# from baselines.common.vec_env import SubprocVecEnv, DummyVecEnv

# ALL_POSSIBLE_ACTIONS = ['LookUp', 'RotateLeft', 'MoveAhead', 'MoveBack', 'RotateRight', 'LookDown']
# ALL_POSSIBLE_ACTIONS = ['LookUp', 'RotateLeft', 'MoveAhead', 'RotateRight', 'LookDown']
ALL_POSSIBLE_ACTIONS = ['RotateLeft', 'MoveAhead', 'RotateRight']


class RoboThorEnv(gym.Env):
    """
    Wrapper base class
    """
    def __init__(self, seed=None, config_file='config_files/NavTaskTrain.json', config_dict=None, device=None):
        """
        :param seed:         (int)   Random seed
        :param config_file:  (str)   Path to environment configuration file. Either absolute or
                                     relative path to the root of this repository.
        :param: config_dict: (dict)  Overrides specific fields from the input configuration file.
        """

        # Loads config settings from file
        self.config = read_config(config_file, config_dict)

        # Randomness settings
        self.np_random = None
        if seed:
            self.seed(seed)

        # priors vision settings
        self.use_priors = self.config['use_priors']
        if self.use_priors: 
            self.vp = VisualPrior(mode=self.config['mode'], m=self.config['m'], k=self.config['k'])
        
    
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else: 
            self.device = torch.device(device)
        print('Ai2thor device (but not used):', self.device)

        # Action settings
        self.action_names = tuple(ALL_POSSIBLE_ACTIONS.copy())
        self.action_space = spaces.Discrete(len(self.action_names))

        # Create task from config
        try:
            self.task = getattr(gym_robothor.tasks, self.config['task']['task_name'])(**self.config)
        except Exception as e:
            raise ValueError('Error occurred while creating task. Exception: {}'.format(e))
        
        # Start ai2thor
        '''
        params stored to env.controller.initialization_parameters >>>>>>>>>>>>>>>>>>>>
        'cameraY': 0.675, 'fieldOfView': 90.0, 'rotateStepDegrees': 30, 'visibilityDistance': 1.0, 'gridSize': 0.25, 
        'agentType': 'stochastic', 'agentMode': 'bot', 'continuousMode': True, 'snapToGrid': False, 
        'applyActionNoise': True, 'renderDepthImage': False, 'renderClassImage': False, 'renderObjectImage': False
        '''
        self.controller = ai2thor.controller.Controller(width=self.config['width'], height=self.config['height'], **self.config['initialize'])
        
        # TOBE params
        self.event = None
        self.scene = self.config['scene']
        self.init_pos = None
        self.init_ori = None
        self.task.target_id = None
        self.observation_space = None


    def step(self, action, verbose=True, return_event=False):
        if not self.action_space.contains(action):
            raise error.InvalidAction('Action must be an integer between '
                                      '0 and {}!'.format(self.action_space.n))
        action_str = self.action_names[action]

        # visible_objects = [obj for obj in self.event.metadata['objects'] if obj['visible']]

        # if/else statements below for dealing with up to 13 actions
        if action_str.startswith('Rotate'):
            self.event = self.controller.step(dict(action=action_str))
        
        elif action_str.startswith('Move') or action_str.startswith('Look'):
            # Move and Look actions
            self.event = self.controller.step(dict(action=action_str))
        
        elif action_str == 'Stop':
            self.event = self.controller.step(dict(action=action_str))
        
        else:
            raise NotImplementedError('action_str: {} is not implemented'.format(action_str))
        
        
        target_obj = self.event.get_object(self.task.target_id)
        cur_pos = self.event.metadata['agent']['position']
        cur_ori = self.event.metadata['agent']['rotation']
        tgt_pos = target_obj['position']

        if self.config['initialize']['renderDepthImage']:
            state = self.preprocess(self.event.frame, cur_pos, cur_ori, tgt_pos, self.event.depth_frame)
        else: 
            state = self.preprocess(self.event.frame, cur_pos, cur_ori, tgt_pos)

        self.reward, self.done = self.task.transition_reward(self.event)
        
        info = {}
        if return_event:
            info['event'] = self.event
        
        return state, self.reward, self.done, info

    
    def preprocess(self, img, cur_pos, cur_ori, tgt_pos, depth=None):
        """
        Compute image operations to generate state representation
        """
        # TODO: replace scikit image with opencv
        # input shape: width,  height, 3
        img = transform.resize(img, self.config['resolution'], mode='reflect')
        img = img.astype(np.float32)
        # cv2.imshow('img',img)
        # cv2.waitKey(1)

        if self.config['grayscale']:
            img = rgb2gray(img) 
            img = np.moveaxis(img, 2, 0)
        elif self.use_priors:
            img = np.moveaxis(img, 2, 0)
            img = torch.Tensor(img).unsqueeze(0)
            img = self.vp.to_representation(img)
            img = img.squeeze(0).cpu().numpy()# 3 dims, tensor      
        else:
            img = np.moveaxis(img, 2, 0)

        if depth is not None:
            depth = transform.resize(depth, self.config['resolution'], mode='reflect')
            depth = np.expand_dims(depth, axis=0)
            img = np.concatenate([img, depth], 0)   
        
        # all to be tensor.
        cur_pos = np.array([cur_pos['x'], cur_pos['z']])
        tgt_pos = np.array([tgt_pos['x'], tgt_pos['z']])
        vector = tgt_pos - cur_pos
        dir = math.atan2(vector[1], vector[0])/math.pi*180
        
        cur_ori = np.array(cur_ori['y'])

        # -180 ~ 180
        cur_ori = 360 - cur_ori
        splm_ori = cur_ori + 180 if cur_ori<180 else cur_ori-180
        dir = 360+dir if dir < 0 else dir
        if splm_ori > cur_ori:
            if dir > cur_ori and dir < splm_ori:# left
                angle = dir-cur_ori
            elif dir < cur_ori: # right
                angle = dir-cur_ori
            else:
                angle = -(360-dir+cur_ori)
        else:
            if dir > splm_ori and dir < cur_ori:# right
                angle = dir-cur_ori
            elif dir < splm_ori:#left
                angle = 360-cur_ori+dir
            else:
                angle= dir-cur_ori
        
        #  to 0 ~ 360
        # angle = angle+360.0 if angle < 0 else angle

        # to rad
        angle = angle / 180.0 * math.pi

        dis = np.sqrt(np.sum(np.square(vector)))
        bear = np.stack([dis*math.cos(angle), dis*math.sin(angle)])/8.0

        bear = bear.astype(np.float32)

        # fake_img = torch.stack([i.expand_as(img[0]) for i in bear])
        # fake_img = bear.unsqueeze(0).unsqueeze(0)
        # fake_img = fake_img.repeat(1, img.shape[1]//fake_img.shape[1], img.shape[2]//fake_img.shape[2])
        # print('fake', fake_img) # 3 dims
        # img = torch.cat([img, fake_img], axis=0).to(self.device) #  cuda if have
        # print(img.shape)

        return dict(rgb=img, compass=bear)

    def reset(self):
        print('Resetting environment and starting new episode')
        # resetting scene
        # self.event = self.controller.reset(scene=self.scene)

        #resetting pos & ori
        assert self.init_pos != None and self.init_ori != None
        teleport_action = dict(action='TeleportFull')
        teleport_action.update(self.init_pos)
        self.controller.step(action=teleport_action)
        self.controller.step(action=dict(action='Rotate', rotation=dict(y=self.init_ori, horizon=0.0)))
        
        # initialize action must be evolved after changing the pos and ori, cannot reset anymore here!!!
        self.event = self.controller.step(dict(action='Initialize', **self.config['initialize']))
        
        # resetting task
        self.task.reset()

        # state preprocessing
        assert self.task.target_id != None
        target_obj = self.event.get_object(self.task.target_id)
        cur_pos = self.event.metadata['agent']['position']
        cur_ori = self.event.metadata['agent']['rotation']
        tgt_pos = target_obj['position']
        self.task.pre_distance = np.sqrt(np.sum(np.square(np.array([cur_pos['x'], cur_pos['z']])-np.array([tgt_pos['x'], tgt_pos['z']]))))
        
        if self.config['initialize']['renderDepthImage']:
            state = self.preprocess(self.event.frame, cur_pos, cur_ori, tgt_pos, self.event.depth_frame)
        else: 
            state = self.preprocess(self.event.frame, cur_pos, cur_ori, tgt_pos)
        
        self.observation_space = dict(rgb = spaces.Box(low=0, high=1, shape=(state['rgb'].shape[0], state['rgb'].shape[1], state['rgb'].shape[2]), dtype=np.uint8),
                                      compass = spaces.Box(low=0, high=1, shape=(state['compass'].shape[0],), dtype=np.uint8)
                                      )
        return state

    def render(self, mode='human'):
        # raise NotImplementedError
        self.controller.step(dict(action='ToggleMapView'))

    def seed(self, seed=None):
        self.np_random, seed_new = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        return seed_new

    def close(self):
        self.controller.stop()


def env_generator(split:str, config_dict=None, device=None):
    split_path = os.path.join(os.path.dirname('gym_robothor/data_split/'), split + ".json")
    with open(split_path) as f:
        episodes = json.loads(f.read())
    
    env = RoboThorEnv(config_file='config_files/NavTaskTrain.json', config_dict=config_dict, device=device)
    
    while True:
    # for e in list(filter(lambda x:x['difficulty']=='easy', episodes)):
        e = random.choice(episodes)
        # e = random.choice(list(filter(lambda x:x['difficulty']=='medium', episodes)))
        env.controller.initialization_parameters['robothorChallengeEpisodeId'] = e['id']
        env.controller.initialization_parameters['shortest_path'] = e['shortest_path']
        
        env.scene = e['scene']
        env.controller.reset(scene=env.scene)
        env.init_pos = e['initial_position']
        env.init_ori = e['initial_orientation']
        env.task.target_id = e['object_id']

        print('>>>>>>> Using scene {}, {} LEVEL'.format(env.controller.initialization_parameters['robothorChallengeEpisodeId'], e['difficulty']))

        yield env


def make_parallel_env(split, n_rollout_threads, seed, device=None):
    def get_env_fn(rank):
        def init_env():
            env = make_env(split, device)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def make_env(split:str, device=None):
    split_path = os.path.join(os.path.dirname('gym_robothor/data_split/'), split + ".json")
    with open(split_path) as f:
        episodes = json.loads(f.read())

    e = random.choice(list(filter(lambda x:x['difficulty']=='easy', episodes)))

    env = RoboThorEnv(config_file='config_files/NavTaskTrain.json', device=device)
    env.controller.initialization_parameters['robothorChallengeEpisodeId'] = e['id']
    env.controller.initialization_parameters['shortest_path'] = e['shortest_path']

    env.scene = e['scene']
    env.controller.reset(scene=env.scene)
    env.init_pos = e['initial_position']
    env.init_ori = e['initial_orientation']
    env.task.target_id = e['object_id']

    return env



if __name__ == '__main__':
    # for i in env_generator('train'):
    #     obs = i.reset() # render() always at the last
    #     print(obs)

    envs = make_parallel_env('train_small', 2, 10, 'cpu')
    envs.reset()
    while True:
        envs.step([0,0])