import os
import numpy as np
import torch
from gym_ai2thor.envs.ai2thor_env import AI2ThorEnv
from gym_robothor.envs.robothor_env import RoboThorEnv, env_generator


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


def worker(worker_id,
           policy,
           storage,
           ready_to_work,
           queue,
           exit_flag,
           task_config_file="config_files/config_example.json"):
    '''
    Worker function to collect experience based on policy and store the experience in storage
    :param worker_id: id used for store the experience in storage
    :param policy: function/actor-critic
    :param storage:
    :param ready_to_work: condition to synchronize work and training
    :param queue: message queue to send episode reward to learner
    :param exit_flag: flag set by leaner to exit the job
    :param task_config_file: the task configuration file
    :return:
    '''

    print(f"Worker with Id:{worker_id} pid ({os.getpid()}) starts ...")

    steps_per_epoch = storage.block_size
    state_size = storage.h_buf.shape[1]
    device = storage.device
    
    episode_rewards, episode_steps = [], []
    r_sum, step_sum = 0., 0

    policy.eval()

    # Wait for start job
    ready_to_work.wait()

    for env in env_generator('train_small', device=device.type):
        if exit_flag.value != 1:
            x = reset(env, state_size, device)
            for i in range(steps_per_epoch):
                with torch.no_grad():
                    a_t, logp_t, _, v_t, state_t = policy(x)
                # interact with environment
                o, r, d, _ = env.step(a_t.item())
                r_sum += r  # accumulate reward within one rollout.
                step_sum += 1
                r_t = torch.tensor(r, dtype=torch.float32).to(device)
                # save experience
                storage.store(worker_id,
                                x["observation"],
                                x['bear'],
                                a_t,
                                r_t,
                                v_t,
                                logp_t,
                                x["memory"]["state"],
                                x["memory"]["mask"])
                # prepare inputs for next step
                x["observation"] = torch.tensor(o['rgb']).to(device) # 128x128 -> 1x128x128
                x['bear'] = torch.tensor(o['compass']).to(device)
                x["memory"]["state"] = state_t
                x["memory"]["mask"] = torch.tensor([(d+1)%2], dtype=torch.float32).to(device)
                x["memory"]["action"] = a_t
                # check terminal state
                if d: # calculate the returns and GAE and reset environment
                    storage.finish_path(worker_id, 0)
                    # print(f"Worker:{worker_id} {device} pid:{os.getpid()} finishes goal at steps :{i}")
                    episode_rewards.append(r_sum)
                    episode_steps.append(step_sum)
                    x = reset(env, state_size, device)
                    r_sum, step_sum = 0., 0
            # env does not reaches end
            if not d:
                _, _, _, last_val, _ = policy(x)
                storage.finish_path(worker_id,last_val)
            # print(f"Worker:{worker_id} {device} pid:{os.getpid()} begins to notify Learner Episode done")
            queue.put((episode_rewards,episode_steps, worker_id))
            # print(f"Worker:{worker_id} waits for next episode")
            episode_rewards, episode_steps = [], []
            # x = reset(env, state_size)
            # r_sum, step_sum = 0., 0
            # Wait for next job
            ready_to_work.clear()
            ready_to_work.wait()
            # print(f"Worker:{worker_id} {device} pid:{os.getpid()} starts new episode")

    env.close()
    print(f"Worker with pid ({os.getpid()})  finished job")


def tester(model, device, n=5, task_config_file="config_files/config_example.json"):
    episode_reward = []
    rnn_size = 128
    env = AI2ThorEnv(config_file=task_config_file)
    for _ in range(n):
        # Wait for trainer to inform next job
        total_r = 0.
        d = False
        x = reset(env, rnn_size, device)
        while not d:
            with torch.no_grad():
                a_t, _, _, _, state_t = model(x)
                # interact with environment
                o, r, d, _ = env.step(a_t.data.item())
                total_r += r  # accumulate reward within one rollout.
                # prepare inputs for next step
                x["observation"] = torch.Tensor(o / 255.).to(device)
                x["memory"]["state"] = state_t
                x["memory"]["mask"] = torch.tensor((d + 1) % 2, dtype=torch.float32).to(device)
                x["memory"]["action"] = a_t

        episode_reward.append(total_r)
        print("Episode reward:", total_r)

    env.close()
    print(f"Average eposide reward ({np.mean(episode_reward)})")