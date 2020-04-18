import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR

# Proximal Policy Optimization (by clipping),
# with early stopping based on approximate KL

def linear_decay(epoch: int, total_num_updates: int) -> float:
    r"""Returns a multiplicative factor for linear value decay

    Args:
        epoch: current epoch number
        total_num_updates: total number of epochs

    Returns:
        multiplicative factor that decreases param value linearly
    """
    return 1 - (epoch / float(total_num_updates))

class PPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 train_iters,
                 mini_batch_size,
                 value_loss_coef,
                 entropy_coef,
                 rnn_steps,
                 lr=0.001,
                 eps=0.001,
                 max_kl=0.15
                 ):

        self.actor_critic = actor_critic
        self.clip_param = clip_param
        self.train_iters = train_iters
        self.mini_batch_size = mini_batch_size

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.rnn_steps = rnn_steps
        self.max_kl = max_kl
        self.optimizer = torch.optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        self.lr_scheduler = LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lambda x: linear_decay(x, 10000),
        )

    def update(self, rollouts, distributed=False):
        device = rollouts.device
        for i in range(self.train_iters):
            batch_gen = rollouts.batch_generator(self.mini_batch_size)
            kl_sum, ent_sum, pi_loss_sum, v_loss_sum = [torch.tensor(0.0).to(device) for _ in range(4)]

            for batch in batch_gen:
                obs, bear, act, adv, ret, logp_old, state, mask, pre_action = batch
                x = {"observation":obs, 'bear':bear,
                     "memory":{
                         "state":state,
                         "mask":mask,
                         "action":pre_action
                     }}

                _, logp_a, ent, v, _ = self.actor_critic(x, action=act, rnn_step_size=self.rnn_steps)
                # PPO policy objective
                ratio = (logp_a - logp_old).exp()
                # min_adv = torch.where(adv > 0, (1 + self.clip_param ) * adv, (1 - self.clip_param ) * adv)
                min_adv = (
                    torch.clamp(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                    )
                    * adv
                )
                pi_loss = -(torch.min(ratio * adv, min_adv)).mean()
                # PPO value objective
                v_loss = F.mse_loss(v, ret)
                # PPO entropy objective
                ent_mean = ent.mean()
                # Policy gradient step
                self.optimizer.zero_grad()
                (pi_loss + v_loss * self.value_loss_coef - ent_mean * self.entropy_coef).backward()
                self.optimizer.step()
                with torch.no_grad():
                    batch_size = len(act)
                    kl_sum += (logp_old - logp_a).sum()
                    ent_sum += ent_mean * batch_size
                    pi_loss_sum += pi_loss * batch_size
                    v_loss_sum += v_loss * batch_size

            kl_mean = kl_sum / rollouts.max_size
            if distributed:
                dist.all_reduce(kl_mean, dist.ReduceOp.SUM)
                kl_mean = kl_mean/dist.get_world_size()
            if torch.abs(kl_mean) > self.max_kl:
                print(f'Early stopping at iter ({i} /{self.train_iters}) due to reaching max kl. ({kl_mean:.4f})')
                break

        entropy_mean = ent_sum / rollouts.max_size
        pi_loss_mean = pi_loss_sum / rollouts.max_size
        v_loss_mean = v_loss_sum / rollouts.max_size

        self.lr_scheduler.step()
        print('Current learning rate:',  self.optimizer.param_groups[0]['lr'])

        return pi_loss_mean, v_loss_mean, kl_mean, entropy_mean