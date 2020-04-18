import os 
from gym_robothor.envs.robothor_env import RoboThorEnv, env_generator
import torch
from algorithms.ppo.core import ActorCritic

hidden_state = 512
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

all_files = os.walk('./pt')


for root, dirs, files in all_files:
        for f in files:
            name = f.split('.')[0]
            env = RoboThorEnv(config_file="config_files/NavTaskTrain.json", config_dict=dict(m=[name]), device='cpu')
            env.init_pos = {'x':0, 'y':0, 'z':0}
            env.init_ori = {'x':0, 'y':0, 'z':0}
            env.task.target_id = 'Apple|+01.98|+00.77|-01.75'
            env.reset()
            obs_dim = env.observation_space['rgb'].shape
            # Share information about action space with policy architecture
            ac_kwargs = dict()
            ac_kwargs['action_space'] = env.action_space
            ac_kwargs['state_size'] = hidden_state
            ac_kwargs['attention'] = None
            ac_kwargs['priors'] = True
            env.close()  
    

            print("Initialize Model...")
            model = ActorCritic(obs_shape=obs_dim, **ac_kwargs)
            model.to(device)
            model.load_state_dict(torch.load(os.path.join(root, f), map_location=torch.device('cpu')))
            print('loaded model completely')
