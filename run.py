from unityagents import UnityEnvironment
import numpy as np
from utils import get_env_spec
import torch
from agent import GaussianPolicyNetwork, GaussianActorCriticNetwork
from rl_algorithms import ppo_unity, unity_rollout_ppo
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":

    hyperparams = {"t_max": 1000, # maximal possible t_step is only 1000
                   "SGD_epoch": 3,
                   "gamma": 0.99,
                   "n_episodes": 3,
                   "grad_clip": 0.5,
                   "epsilon": 0.1,
                   "beta": 0.01}  # cf. tnakae
    dataloader_params = {'batch_size': 256,
                         'shuffle': True,
                         'num_workers': 4}
    setting = {"multi_env": True,
               "plotting": True}

    if setting["multi_env"]:
        env = UnityEnvironment(file_name='./Reacher_Multi_Linux/Reacher.x86')
    else:
        env = UnityEnvironment(file_name='./Reacher_Linux/Reacher.x86')

    env_spec = get_env_spec(env)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = GaussianActorCriticNetwork(
        state_dim=env_spec['state_size'],
        action_dim=env_spec['action_size'],
        hiddens=[512, 256]).to(device)


    scores = ppo_unity(agent, env,
                       'gae',
                       unity_rollout_ppo,
                       dataloader_params=dataloader_params,
                       **hyperparams)

    with open(os.path.join('./data/ppo_gae/progress.txt'), 'w') as myfile:
        myfile.write(str(scores))
    myfile.close()

    if setting["plotting"]:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(1, len(scores) + 1), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()