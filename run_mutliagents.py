from unityagents import UnityEnvironment
import numpy as np
from collections import deque
from utils import get_env_spec
from torch.utils import data
import torch
from agent import PPOPolicy, PPOActorCritic
import torch.optim as optim
import time
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def reward_to_go_evaluation(discount, rewards, normalize=False):
    t_max, n_agents = rewards.shape
    discounts = np.array([discount ** i for i in range(t_max+1)])
    discounts = np.expand_dims(discounts, 1)
    padding = np.zeros((1, n_agents)) # padding for cumsum evaluation
    rewards = np.concatenate((padding, rewards), axis=0)
    # # todo, debug, to remove
    # rewards += np.expand_dims(np.arange(0,rewards.shape[0]),1)
    # ###############
    discounted_rewards_tmp = discounts * rewards
    discounted_r_totals = discounted_rewards_tmp.sum(0)
    # print("tmp",discounted_rewards_tmp)
    # print("r_tot",discounted_r_totals)


    discounted_cumsum = np.cumsum(discounted_rewards_tmp, axis=0)
    discounted_cumsum = discounted_cumsum[:-1, :]
    # print("discounted cumsum: ", discounted_cumsum)
    # print("discounted_r_totals", discounted_r_totals)
    # print("final results", (discounted_r_totals - discounted_cumsum).reshape([t_max * n_agents, 1]))
    if normalize:
        small = 1e-8
        rs_to_go = discounted_r_totals - discounted_cumsum
        sigma = np.std(rs_to_go, axis=1, keepdims=True) # shape=[t_max, 1]
        mu = np.mean(rs_to_go, axis=1, keepdims=True) # shape=[t_max, 1]
        # print("sigma",sigma)
        # print("mu", mu)
        # print(((rs_to_go - mu) / (sigma + small)).reshape([t_max * n_agents, 1]))
        return ((rs_to_go - mu)/(sigma+small)).reshape([t_max * n_agents, 1])
    else:
        return (discounted_r_totals - discounted_cumsum).reshape([t_max * n_agents, 1]) # in format [len(self), 1])


def discounted_rewards_sums(discount, rewards):
    """
    evaluate discounted reward sums for each agent,
    output as a shape of [N_agent*T_max, 1]
    :param discount:
    :param rewards:
    :return:
    """
    t_max = rewards.shape[0]  # trajectory length
    discounts = np.array([discount ** i for i in range(t_max)])
    discounts = np.expand_dims(discounts, 1)
    rewards = np.array(rewards)

    Rs = (rewards * discounts).sum(0) # total rewards for all N agents
    return np.expand_dims(np.repeat(Rs, t_max), 1)


class MAReacherTrajectories(data.Dataset):
    def __init__(self, trajectories, gamma=1.0, eval_func=reward_to_go_evaluation):
        self.statess = np.array(trajectories["states"])
        self.actionss = np.array(trajectories["actions"])
        self.t_max, self.n_agents, _ = self.statess.shape # assume the same t_max for actions, states and rewards
        self.statess = self.statess.reshape([len(self), -1])
        self.actionss = self.actionss.reshape([len(self), -1])
        self.log_porbs_old = torch.cat(trajectories["log_probs"]).reshape([len(self), -1])
        self.rewardss = np.array(trajectories['rewards'])
        self.estimated_values = eval_func(gamma, self.rewardss)  # shape must be [len(self),1])

    def __len__(self):
        return self.t_max*self.n_agents

    def __getitem__(self, item):
        return self.statess[item, :], self.actionss[item, :], self.log_porbs_old[item, :], self.estimated_values[item, :]

    def average_score(self):
        return np.array(self.rewardss).sum(0).mean()


def unity_rollout_ppo(agent, env, max_t):
    trajectory = {"actions": [],
                  "rewards": [],
                  "states": [],
                  'log_probs': []}

    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
    states = env_info.vector_observations
    #agent.reset()
    for t in range(max_t):
        # todo, add PPO-type of action
        actions, log_probs = agent.act(states)
        env_info = env.step(actions)[brain_name]  # send the action to the environment
        next_states = env_info.vector_observations  # get the next state
        rewards = env_info.rewards  # get the reward
        dones = env_info.local_done  # see if episode has finished
        trajectory['states'].append(states)
        trajectory['actions'].append(actions)
        trajectory['rewards'].append(rewards)
        trajectory['log_probs'].append(log_probs.detach())
        states = next_states
        if np.any(dones):
            break

    return trajectory


def test_multiagent_env(env, agent, t_max):
    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
    states = env_info.vector_observations
    agent.reset()

    for t in range(t_max):
        actions = agent.act(states)
        env_info = env.step(actions)[brain_name]  # send the action to the environment
        next_states = env_info.vector_observations  # get the next state
        rewards = env_info.rewards  # get the reward
        dones = env_info.local_done  # see if episode has finished

        print("Actions", actions[0]) # list of np arrays (format determined by me!), udacity-ddpg: (4,)
        print("Rewards", type(rewards[0])) # list of floats
        print("Next States", type(next_states[0])) # list of np arrays

        states = next_states  # roll over the state to next time step
        if np.any(dones):  # exit loop if episode finished
            break


def clipped_surrogate(policy,
                      log_probs_old,
                      states,
                      actions,
                      values, epsilon=0.01, beta=0.01):

    state = states.float().to(device)
    action = actions.float().to(device)  # the format of action matters a lot
    values = values.float().to(device).squeeze(1)
    dist = policy.forward(state)

    log_probs_new = dist.log_prob(action).sum(-1)
    log_probs_old = log_probs_old.sum(-1)


    # print(log_probs_old)
    ent_loss = dist.entropy().sum(-1).mean()
    # print("shape log_prob_new", log_probs_new.shape)
    # print("shape log_prob_old", log_probs_old.shape)
    # print("values", values.shape)
    ratios = (log_probs_new - log_probs_old).exp()
    # print("shape ratios", ratios.shape)
    #print("ratios", ratios)
    surr1 = ratios * values
    surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * values
    # print("surr1 shape:", surr1)
    # print("surr2 shape:", surr2)
    Ls = torch.min(surr1, surr2)
    # print("Ls shape:", Ls)
    #print(A)
    return -Ls.mean() - beta * ent_loss


def ppo_unity(policy, env, eval_func, rollout_func,
        n_episodes=1000, max_t=1000,
        gamma=1.0, SGD_epoch=5, grad_clip=0.2,
        epsilon=0.01, beta=0.01, pass_score=30.0,
        print_every=100):

    optimizer = optim.Adam(policy.parameters(), lr=1e-4) # todo, include it into the agent
    scores_deque = deque(maxlen=100)
    dataloader_params = {'batch_size': 32,
                         'shuffle': True,
                         'num_workers': 4}
    scores = []

    t0 = time.time()
    for i_episode in range(1, n_episodes + 1):
        trajectories = rollout_func(policy, env, max_t)
        PPO_Batch = MAReacherTrajectories(trajectories, gamma, eval_func)
        episodic_return = PPO_Batch.average_score()
        PPO_Batch_generator = data.DataLoader(PPO_Batch, **dataloader_params)
        for _ in range(SGD_epoch):
            for mb_states, mb_actions, mb_log_probs, mb_estimated_values in PPO_Batch_generator:
                L = clipped_surrogate(policy,
                                      mb_log_probs,
                                      mb_states,
                                      mb_actions,
                                      mb_estimated_values,
                                      epsilon=epsilon, beta=beta)
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

    return scores


if __name__ == "__main__":

    hyperparams = {"max_t": 120, # maximal possible t_step is only 1000
                   "SGD_epoch": 10,
                   "gamma": 0.99,
                   "n_episodes": 150,
                   "grad_clip": 10.0,
                   "epsilon": 0.1,
                   "beta": 0.0} # cf. tnakae
    setting = {"multi_env": True,
               "plotting": False}
    if setting["multi_env"]:
        env = UnityEnvironment(file_name='./Reacher_Multi_Linux/Reacher.x86')
    else:
        env = UnityEnvironment(file_name='./Reacher_Linux/Reacher.x86')

    env_spec = get_env_spec(env)
    agent = PPOPolicy(s_size=env_spec['state_size'],
                       h_size=512,
                       a_size=env_spec['action_size'],
                       continuous=True,
                       ).to(device)


    scores = ppo_unity(agent, env,
                       discounted_rewards_sums,
                       unity_rollout_ppo,
                       **hyperparams)

    if setting["plotting"]:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(1, len(scores) + 1), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()