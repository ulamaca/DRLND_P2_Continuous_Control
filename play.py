from utils import play, get_env_spec, play_random
import torch
from unityagents import UnityEnvironment
from agent import GaussianActorCriticNetwork, GaussianPolicyNetwork
import argparse

parser=argparse.ArgumentParser(description="Play an agent acting in the environment")
parser.add_argument('-p', '--params_path', type=str, metavar='', default='data/dudqn-1/checkpoint.pth',
                                           help="path to the model parameters")
args=parser.parse_args()


if __name__ == "__main__":
    setting = {"multi_env": True}
    if setting["multi_env"]:
        env = UnityEnvironment(file_name='./Reacher_Multi_Linux/Reacher.x86')
    else:
        env = UnityEnvironment(file_name='./Reacher_Linux/Reacher.x86')

    env_spec = get_env_spec(env)
    agent = GaussianActorCriticNetwork(
        state_dim=env_spec['state_size'],
        action_dim=env_spec['action_size'],
        hiddens=[512, 256])

    params_path = args.params_path

    agent.load_state_dict(torch.load(params_path))
    #play_random(env)  # option for checking the performance of a random (untrained) agent
    play(env, agent)