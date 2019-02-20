import numpy as np
from collections import deque
from torch.utils import data
import torch
from utils import to_tensor, standardize
from value_estimators import *
import torch.optim as optim
import time

default_dataloader_params={'batch_size': 256, 'shuffle': True,'num_workers': 4}


class MAReacherTrajectories(data.Dataset):
    def __init__(self, trajectories, gamma=1.0, eval_func='gae'):
        self.trajectories=trajectories
        for n, v in trajectories.items():
            if 'last' not in n:
                self.trajectories[n] = torch.cat(v)

        # todo-2, add conditional control for the case of rewards shape for GYM environment:
        self.t_max, self.n_agents = self.trajectories['rewards'].shape  # assume the same t_max for actions, states and rewards
        if eval_func == 'gae':
            # todo-, gae_lambda is not exposed for hyperparam optimization
            self.estimated_values, self.true_values = GAE(self.trajectories['rewards'], self.trajectories['values'],
                                                          self.trajectories['last_dones'], self.trajectories['last_values'],
                                                          gamma=gamma)

        self.estimated_values = standardize(self.estimated_values)
        for n, v in trajectories.items():
            if not('last' in n or n=='rewards'): # rewards are not used for learning, so I fix the shape to keep info not lose
                if len(v.shape) == 3:
                    trajectories[n] = v.reshape([-1, v.shape[-1]])
                else:
                    trajectories[n] = v.reshape([-1])

        ##final shape checking
        # for n, v in trajectories.items():
        #     print(n, 'has shape:', v.shape)
        self.estimated_values = self.estimated_values.reshape([-1])
        self.true_values = self.true_values.reshape([-1])

    def __len__(self):
        return self.t_max*self.n_agents

    def __getitem__(self, item):
        return self.trajectories['states'][item], self.trajectories['actions'][item], \
               self.trajectories['log_probs'][item], self.estimated_values[item], self.true_values[item]

    def mean_trajectory_score(self):
        return self.trajectories['rewards'].sum(0).mean().item()


def unity_rollout_ppo(agent, env, max_t):
    trajectory = {"actions": [],
                  "rewards": [],
                  "states": [],
                  "log_probs": [],
                  "values": [],
                  "last_values": None,
                  "last_dones": None}

    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=True)[brain_name]
    states = env_info.vector_observations

    for t in range(max_t):
        states = to_tensor(states) # states is now tensor type
        actions, log_probs, _, values = agent.forward(states)

        env_info = env.step(actions.cpu().numpy())[brain_name]  # send the action to the environment
        next_states = env_info.vector_observations  # get the next state
        rewards = env_info.rewards  # get the reward
        dones = env_info.local_done  # see if episode has finished

        trajectory['states'].append(states.unsqueeze(0))
        trajectory['rewards'].append(to_tensor(rewards).unsqueeze(0))
        trajectory['actions'].append(actions.detach().unsqueeze(0)) # todo, may not be necessary
        trajectory['log_probs'].append(log_probs.detach().unsqueeze(0))
        if not (values is None):
            trajectory['values'].append(values.detach().unsqueeze(0))

        states = next_states
        if np.any(dones):
            break

    trajectory['last_values'] = agent.state_values(to_tensor(next_states)).detach()
    trajectory['last_dones'] = to_tensor(dones)

    return trajectory


def ppo_unity(policy, env, eval_func, rollout_func,
              n_episodes=1000, t_max=1000,
              gamma=1.0, SGD_epoch=5, grad_clip=0.2,
              epsilon=0.01, beta=0.01, pass_score=30.0,
              print_every=100, dataloader_params=default_dataloader_params):

    optimizer = optim.Adam(policy.parameters(), lr=1e-4)
    scores_deque = deque(maxlen=100)
    scores = []

    t0 = time.time()
    for i_episode in range(1, n_episodes + 1):
        trajectories = rollout_func(policy, env, t_max)
        PPO_Batch = MAReacherTrajectories(trajectories, gamma, eval_func)
        episodic_return = PPO_Batch.mean_trajectory_score()
        PPO_Batch_generator = data.DataLoader(PPO_Batch, **dataloader_params)

        policy.train()
        for e in range(SGD_epoch):
            for i_step, (mb_states, mb_actions, mb_log_probs, mb_estimated_values, mb_returns) in enumerate(PPO_Batch_generator):
                # PPO loss_evaluation
                _, log_probs_new, entropy, mb_v = policy(mb_states, mb_actions)
                ratios = (log_probs_new - mb_log_probs).exp()
                surr1 = ratios * mb_estimated_values
                surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * mb_estimated_values
                L_actor = -torch.min(surr1, surr2).mean() - beta * entropy.mean()
                L_critic = 0.5 * (mb_returns - mb_v).pow(2).mean()
                L = L_actor + L_critic

                # training steps check:
                # print("epoch {}".formate), "step {}".format(i_step), "batch-size {}".format(ratios.shape[0]))

                optimizer.zero_grad()
                L.backward()
                if not(grad_clip is None):
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
                optimizer.step()
                del L

        # episodic value estimation
        scores_deque.append(episodic_return)
        scores.append(episodic_return)
        # clipping and exploration decay
        epsilon *= .999
        beta *= .995

        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque), episodic_return), end="")
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque) >= pass_score:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100, np.mean(scores_deque)))
            break

    t_end = time.time()
    torch.save(policy.state_dict(), './data/ppo_gae/checkpoint.pth')
    print("\nTotal time elapsed: {} second".format(t_end-t0))
    policy.eval()

    return scores