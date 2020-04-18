import os, time
import torch
import numpy as np
from algorithms.ppo.ppo import PPO

import torch.distributed as dist
from tensorboardX import SummaryWriter


def dist_sum(x):
    dist.all_reduce(x, dist.ReduceOp.SUM)
    return x


def dist_mean(x):
    return dist_sum(x) / dist.get_world_size()


class TB_logger:
    def __init__(self, name, rank=0):
        self.rank = rank
        if self.rank == 0:
            self.writer = SummaryWriter(comment=name)

    def add_scalar(self, name, val, steps):
        if self.rank == 0:
            self.writer.add_scalar(name, val, steps)
    
    def log_info(self, info):
        if self.rank ==0:
            print(info)

def learner(model, rollout_storage, train_params, ppo_params, ready_to_works, queue, sync_flag, rank=0, distributed=False, b=None):
    '''
    learner use ppo algorithm to train model with experience from storage
    :param model:
    :param storage:
    :param params:
    :param ready_to_works:
    :param queue:
    :param sync_flag:
    :param rank:
    :return:
    '''

    print(f"learner with pid ({os.getpid()})  starts job")
    logger = TB_logger("ppo_ai2thor",rank)
    agent = PPO(actor_critic = model, **ppo_params)
    device = rollout_storage.device
    if distributed:
        world_size = dist.get_world_size()
    else:
        world_size = 1
    
    epochs = train_params["epochs"]
    min_clip_param = 0.001
    min_kl = 0.001
    # start workers for next epoch
    _ = [e.set() for e in ready_to_works]
    # Training policy
    start_time = time.time()
    for epoch in range(epochs):
        agent.clip_param = (ppo_params['clip_param'] - min_clip_param)*(epochs -epoch)/epochs + min_clip_param
        agent.max_kl = (ppo_params['max_kl'] -  min_kl)*(epochs-epoch)/epochs + min_kl
        rollout_ret = []
        rollout_steps = []
        # wait until all workers finish a epoch
        for i in range(train_params["num_workers"]):
            rewards, steps, id = queue.get()
            print(f'Leaner rank:{rank} recieve worker:{id} done signal and reaches {i}th wokers')
            rollout_ret.extend(rewards)
            rollout_steps.extend(steps)

        if b:
            print(f'Learner rank:{rank} wait')
            b.wait()
        print("Start training")
        # normalize advantage
        # if distributed:
        #     mean = rollout_storage.adv_buf.mean()
        #     var = rollout_storage.adv_buf.var()
        #     mean = dist_mean(mean)
        #     var = dist_mean(var)
        #     rollout_storage.normalize_adv(mean_std=(mean, torch.sqrt(var)))
        # else:
        #     rollout_storage.normalize_adv()

        # train with batch
        model.train()
        pi_loss, v_loss, kl, entropy = agent.update(rollout_storage, distributed)
        v_mean = rollout_storage.val_buf.mean()
        model.eval()
        print("Finishes training")
        # start workers for next epoch
        if epoch == train_params["epochs"] -1:
            # set exit flag to 1, and notify workers to exit
            sync_flag.value = 1
        _ = [e.set() for e in ready_to_works]

        # log statistics with TensorBoard
        ret_sum = np.sum(rollout_ret)
        steps_sum = np.sum(rollout_steps)
        episode_count = len(rollout_ret)

        if distributed:
            pi_loss = dist_mean(pi_loss)
            v_loss = dist_mean(v_loss)
            kl = dist_mean(kl)
            entropy = dist_mean(entropy)
            v_mean = dist_mean(v_mean)
            ret_sum = dist_sum(torch.tensor(ret_sum).to(device))
            steps_sum = dist_sum(torch.tensor(steps_sum).to(device))
            episode_count = dist_sum(torch.tensor(episode_count).to(device))
        # Log info about epoch
        global_steps = (epoch + 1) * train_params["steps"] * train_params["world_size"]
        fps = global_steps / (time.time() - start_time)
        logger.log_info(f"Epoch [{epoch}] avg. FPS:[{fps:.2f}]")

        logger.add_scalar("KL", kl, global_steps)
        logger.add_scalar("Entropy", entropy, global_steps)
        logger.add_scalar("p_loss", pi_loss, global_steps)
        logger.add_scalar("v_loss", v_loss, global_steps)
        logger.add_scalar("v_mean", v_mean, global_steps)
        
        # print(agent.clip_param,agent.max_kl)
        logger.add_scalar("clip_ration", agent.clip_param, global_steps)
        logger.add_scalar("max_kl", agent.max_kl, global_steps)
        
        if episode_count > 0:
            ret_per_1000 = (ret_sum / steps_sum) * 1000
            logger.add_scalar("Return1000", ret_per_1000, global_steps)
            logger.log_info(f"Epoch [{epoch}] Steps {global_steps}: "
                  f"return:({ret_per_1000:.1f}), sum:{ret_sum}, step_sum:{steps_sum}")
        else:
            logger.log_info(f"Epoch [{epoch}] Steps {global_steps}: "
                  f"Goal is not reached in this epoch")
        
        if (epoch + 1) % 20 == 0 and rank == 0:
            if distributed:
                torch.save(model.module.state_dict(), f'model{epoch+1}.pt')
            else:
                torch.save(model.state_dict(), f'model{epoch+1}.pt')
        print("finish statistics")
        
    print(f"learner with pid ({os.getpid()})  finished job")