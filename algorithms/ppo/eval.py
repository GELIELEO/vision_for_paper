import numpy as np
import torch
from torch.autograd import Variable

from gym_robothor.envs.robothor_env import RoboThorEnv, env_generator
from algorithms.ppo.core import ActorCritic
import ai2thor.util.metrics
import torch
import argparse


def reset(env, state_size, device):
    o = env.reset()
    mask_t = torch.tensor(0., dtype=torch.float32).to(device)
    prev_a = torch.tensor(0, dtype=torch.long).to(device)
    obs_t = torch.tensor(o['rgb']).to(device)
    bear_t = torch.tensor(o['compass']).to(device)
    state_t = torch.zeros(state_size, dtype=torch.float32).to(device)
    x = {"observation": obs_t, 'bear':bear_t,
         "memory": {
             "state": state_t,
             "mask": mask_t,
             "action": prev_a
          }
         }
    return x

def evaluate(args):
    hidden_state = 512
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    env = RoboThorEnv(config_file="config_files/NavTaskTrain.json", device='cpu')
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
    model.load_state_dict(torch.load(args.model_path))
    print('loaded model completely')

    max_num_env = 20

    col_record = {}
    spl_record = {}

    for i, env in enumerate(env_generator('test_valid_', device=device.type)):
        if i < max_num_env:
            episode_result = dict(shortest_path=env.controller.initialization_parameters['shortest_path'], success=False, path=[])
            episode_result['path'].append(env.controller.last_event.metadata['agent']['position'])

            inputs = reset(env, hidden_state, device)
            done = False

            while not done:
                a_t, logp_t, _, v_t, state_t = model(inputs)
                
                with torch.no_grad():
                    state, reward, done, _ = env.step(a_t.item()) # if the data is in cuda, use item to extract it.
                
                episode_result['path'].append(env.controller.last_event.metadata['agent']['position'])
                
                inputs["observation"] = torch.tensor(state['rgb']).to(device)
                inputs["bear"] = torch.tensor(state['compass']).to(device)
                inputs["memory"]["state"] = state_t
                inputs["memory"]["mask"] = torch.tensor((done+1)%2, dtype=torch.float32).to(device)
                inputs["memory"]["action"] = a_t

                if done:
                    target_obj = env.controller.last_event.get_object(env.task.target_id)
                    episode_result['success'] = target_obj['visible']
                    spl = ai2thor.util.metrics.compute_spl([episode_result])
                    print(spl, env.task.num_collision, env.scene)
                    
                    if env.scene in col_record:
                        col_record[env.scene].append(env.task.num_collision)
                        spl_record[env.scene].append(spl)
                    else:
                        col_record[env.scene] = [env.task.num_collision]
                        spl_record[env.scene] = [spl]
                    
                    break
        else:
            break
    
    print(col_record, spl_record)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default=None)
    args = parser.parse_args()
    evaluate(args)




'''
 {
        "difficulty": "medium",
        "id": "Train_1_1_Apple_17",
        "initial_orientation": 90,
        "initial_position": {
            "x": 1.75,
            "y": 0.9009997,
            "z": -4.25
        },
        "object_id": "Apple|+01.98|+00.77|-01.75",
        "object_type": "Apple",
        "scene": "FloorPlan_Train1_1",
        "shortest_path": [
            {
                "x": 1.75,
                "y": 0.0103442669,
                "z": -4.25
            },
            {
                "x": 2.85833335,
                "y": 0.0103442669,
                "z": -3.208334
            },
            {
                "x": 4.025,
                "y": 0.0103442669,
                "z": -2.68333435
            },
            {
                "x": 4.141667,
                "y": 0.0103442669,
                "z": -2.56666756
            },
            {
                "x": 4.025,
                "y": 0.0103442669,
                "z": -2.27500057
            },
            {
                "x": 3.0,
                "y": 0.0103442669,
                "z": -2.0
            }
        ],
        "shortest_path_length": 4.340735893219212,
        "target_position": {
            "x": 1.979,
            "y": 0.7714,
            "z": -1.753
        }
    },
    '''