from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt
from ddpg_agent import Agent
from algorithms import ddpg
from utils import get_env_spec
#%matplotlib inline


if __name__ == "__main__":

    env = UnityEnvironment(file_name='./Reacher_Multi_Linux/Reacher.x86')
    env_spec = get_env_spec(env)

    agent = Agent(state_size=env_spec['state_size'],
                  action_size=env_spec['action_size'],
                  random_seed=10)

    # todo, something needs to be changed to make ddpg a multi-agent version
    scores = ddpg(agent, env, n_episodes=4000, steps_per_print=100)


    print("maximal steps in reacher (single) env is 1001")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()