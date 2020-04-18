import sys
import torch
from algorithms.ppo.worker import tester
from algorithms.ppo.core import ActorCritic
from  gym_ai2thor.envs.ai2thor_env import AI2ThorEnv

if __name__ == '__main__':

    gpu_id = 0
    device = torch.device(f'cuda:{gpu_id}')
    env = AI2ThorEnv()
    obs_dim = env.observation_space.shape
    # Share information about action space with policy architecture
    rnn_size= 128
    ac_kwargs = dict()
    ac_kwargs['action_space'] = env.action_space
    ac_kwargs['state_size'] = rnn_size
    env.close()
    # Construct Model
    ac_model = ActorCritic(obs_shape=obs_dim, **ac_kwargs).to(device)

    for name in sys.argv[1:]:
        print("test with model:",name)
        state_dict = torch.load(name)
        # load params
        ac_model.load_state_dict(state_dict)
        tester(ac_model,device,task_config_file="config_files/multiMugTaskTest.json")
    
    print(f"Tester finished job")
