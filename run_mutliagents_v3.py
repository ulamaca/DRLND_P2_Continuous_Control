from unityagents import UnityEnvironment
import numpy as np
from collections import deque
from utils import get_env_spec
from torch.utils import data
import torch
from agent import PPOPolicy, GaussianPolicyNetwork, GaussianActorCriticNetwork
import torch.optim as optim
import time
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def mc_values(discount, rewards):
    '''
    implement it using dynamic programming (bellman equaion: v_t = r_t + \gamma * v_{t+1},
    :param discount: float, discount factor
    :param rewards: tensor/np.array, rewards of the shape: [t_max, n_agent, 1]
    :param normalize:
    :return: MC value estimate
    '''
    t_max, n_agent = rewards.shape
    mc_values = torch.zeros_like(rewards).float() #.to(self.device)
    returns_curr = 0.0  # by setting last value (v_{t+1}) equal to zero
    for t in reversed(range(t_max)):
        r_t = rewards[t]
        returns_curr = r_t + discount*returns_curr
        mc_values[t] = returns_curr

    return mc_values


def GAE(rewards, values, last_dones, last_values):
        n_step, n_agent = rewards.shape

        # Create empty buffer
        GAE = torch.zeros_like(rewards).float()#.to(self.device)
        returns = torch.zeros_like(rewards).float()#.to(self.device)
        gae_lambda = 0.96
        gamma = 0.99

        # Set start values
        GAE_current = torch.zeros(n_agent).float()#.to(self.device)
        returns_current = last_values * last_dones
        values_next = last_values * last_dones

        for irow in reversed(range(n_step)):
            values_current = values[irow]
            rewards_current = rewards[irow]

            # Calculate TD Error
            td_error = rewards_current + gamma * values_next - values_current
            # Update GAE, returns
            GAE_current = td_error + gamma * gae_lambda * GAE_current
            returns_current = rewards_current + gamma * returns_current
            # Set GAE, returns to buffer
            GAE[irow] = GAE_current
            returns[irow] = returns_current

            values_next = values_current

        return GAE, returns


def total_returns(discount, rewards):
    """
    total return for learning in REINFORCE
    :param discount:
    :param rewards:
    :return:
    """
    discounted_rewards = mc_values(discount, rewards)
    return torch.sum(0).repeat(discounted_rewards.shape[0])


class MAReacherTrajectories(data.Dataset):
    def __init__(self, trajectories, gamma=1.0, eval_func=mc_values):
        self.trajectories=trajectories
        for n, v in trajectories.items():
            if 'last' not in n:
                self.trajectories[n] = torch.cat(v)
                #print(n, "shape", self.trajectories[n].shape)
        # todo-2, add conditional control for the case of other rewards shape:
        self.t_max, self.n_agents = self.trajectories['rewards'].shape  # assume the same t_max for actions, states and rewards

        # value estimation
        # self.estimated_values = eval_func(gamma, self.trajectories['rewards'])
        self.estimated_values, self.returns = GAE(self.trajectories['rewards'], self.trajectories['values'],
                                             self.trajectories['last_dones'], self.trajectories['last_values'])

        self.estimated_values = standardize(self.estimated_values)
        # reshape the data after value evaluation
        for n, v in trajectories.items():
            if n!='rewards' and 'last' not in n: # todo-3 condition the 'rewards' parts properly
                if len(v.shape) == 3:
                    trajectories[n] = v.reshape([-1, v.shape[-1]])
                else:
                    trajectories[n] = v.reshape([-1])
        self.estimated_values = self.estimated_values.reshape([-1])
        self.returns = self.returns.reshape([-1])
        #print(self.estimated_values.shape)

    def __len__(self):
        return self.t_max*self.n_agents

    # def to_tensor(self, x, dtype=np.float32):
    #     return torch.from_numpy(np.array(x).astype(dtype)) #.to(self.device)

    def __getitem__(self, item):
        return self.trajectories['states'][item], self.trajectories['actions'][item], \
               self.trajectories['log_probs'][item], self.estimated_values[item], self.returns[item]

    def average_score(self):
        # print(self.trajectories['rewards'].type())
        # print(self.trajectories['rewards'].shape)
        # print(A)
        return self.trajectories['rewards'].sum(0).mean().numpy()


def to_tensor(x, dtype=np.float32):
    return torch.from_numpy(np.array(x).astype(dtype)) #.to(self.device) #todo, devising problem


def unity_rollout_ppo(agent, env, max_t, use_critic=True):
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
        #todo-3, better formatting: states will be transformed into tensor twice
            # put it into an agent class
        if use_critic:
            actions, log_probs, _, values = agent.forward(to_tensor(states))
            trajectory['values'].append(values.detach().unsqueeze(0))
        else:
            actions, log_probs, _, _ = agent.forward(to_tensor(states))

        env_info = env.step(actions.cpu().numpy())[brain_name]  # send the action to the environment
        next_states = env_info.vector_observations  # get the next state
        rewards = env_info.rewards  # get the reward
        dones = env_info.local_done  # see if episode has finished

        trajectory['states'].append(to_tensor(states).unsqueeze(0))
        trajectory['rewards'].append(to_tensor(rewards).unsqueeze(0))
        trajectory['actions'].append(actions.detach().unsqueeze(0))
        trajectory['log_probs'].append(log_probs.detach().unsqueeze(0))

        states = next_states
        if np.any(dones):
            break

    trajectory['last_values'] = agent.state_values(to_tensor(states)).detach()
    trajectory['last_dones'] = to_tensor(dones)

    return trajectory


def clipped_surrogate(policy,
                      log_probs_old,
                      state,
                      action,
                      values, epsilon=0.01, beta=0.01):

    # todo-1: this function should not do processing
    _, log_probs_new, entropy, v_output = policy(state, action)

    # print(log_probs_old)
    ent_loss = entropy.mean()
    # print("shape log_prob_new", log_probs_new.shape)
    # print("shape log_prob_old", log_probs_old.shape)
    # print("values", values.shape)
    ratios = (log_probs_new - log_probs_old).exp()
    # print("shape ratios", ratios.shape)
    # print("ratios", ratios)
    #print(A)
    surr1 = ratios * values
    surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * values
    #print("surr1 shape:", surr1)
    #print("surr2 shape:", surr2)
    Ls = torch.min(surr1, surr2)
    # print("Ls shape:", Ls)
    #print(A)
    return -Ls.mean() - beta * ent_loss, v_output


def ppo_unity(policy, env, eval_func, rollout_func,
        n_episodes=1000, max_t=1000,
        gamma=1.0, SGD_epoch=5, grad_clip=0.2,
        epsilon=0.01, beta=0.01, pass_score=30.0,
        print_every=100):

    optimizer = optim.Adam(policy.parameters(), lr=1e-4) # todo, include it into the agent
    scores_deque = deque(maxlen=100)
    dataloader_params = {'batch_size': 256,
                         'shuffle': True,
                         'num_workers': 4}
    scores = []

    t0 = time.time()
    for i_episode in range(1, n_episodes + 1):
        trajectories = rollout_func(policy, env, max_t)
        PPO_Batch = MAReacherTrajectories(trajectories, gamma, eval_func)
        episodic_return = PPO_Batch.average_score()
        PPO_Batch_generator = data.DataLoader(PPO_Batch, **dataloader_params)

        policy.train()
        for e in range(SGD_epoch):
            for i_step, (mb_states, mb_actions, mb_log_probs, mb_estimated_values, mb_returns) in enumerate(PPO_Batch_generator):
                # loss_evaluation
                _, log_probs_new, entropy, mb_v = policy(mb_states, mb_actions)
                ratios = (log_probs_new - mb_log_probs).exp()
                surr1 = ratios * mb_estimated_values
                surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * mb_estimated_values
                L_actor = -torch.min(surr1, surr2).mean() - beta * entropy.mean()
                L_critic = 0.5 * (mb_returns - mb_v).pow(2).mean()
                L = L_actor + L_critic

                #print("epoch {}".formate), "step {}".format(i_step), "batch-size {}".format(ratios.shape[0]))

                optimizer.zero_grad()
                L.backward()
                if grad_clip > 0.0:
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
    torch.save(policy.state_dict(), 'ppo_checkpoint.pth')
    print("\nTotal time elapsed: {} second".format(t_end-t0))
    policy.eval()

    return scores


if __name__ == "__main__":

    hyperparams = {"max_t": 1000, # maximal possible t_step is only 1000
                   "SGD_epoch": 3,
                   "gamma": 0.99,
                   "n_episodes": 3000,
                   "grad_clip": 0.5,
                   "epsilon": 0.1,
                   "beta": 0.01} # cf. tnakae

    setting = {"multi_env": True,
               "plotting": False}
    if setting["multi_env"]:
        env = UnityEnvironment(file_name='./Reacher_Multi_Linux/Reacher.x86')
    else:
        env = UnityEnvironment(file_name='./Reacher_Linux/Reacher.x86')

    env_spec = get_env_spec(env)
    agent = GaussianActorCriticNetwork(
        state_dim=env_spec['state_size'],
        action_dim=env_spec['action_size'],
        hiddens=[512, 256]).to(device)


    scores = ppo_unity(agent, env,
                       calc_returns,
                       unity_rollout_ppo,
                       **hyperparams)

    if setting["plotting"]:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(1, len(scores) + 1), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()