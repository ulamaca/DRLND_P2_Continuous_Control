import numpy as np
import torch

# Unity Environmental Processing Tools:
def get_env_spec(env):
    """
    get and print necessary specifications from a Unity environment
    :param env: a unity environment instance
    :return: a dict with state and action size of the env
    """
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)
    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]
    state_size = len(state)
    return {"state_size":state_size,
            "action_size":action_size}



# PyTorch Tensor Processing Tools:
def standardize(tensor):
    return (tensor - tensor.mean()) / (tensor.std() + 1e-6)

def to_tensor(x, dtype=np.float32):
    return torch.from_numpy(np.array(x).astype(dtype)) #.to(self.device) #todo, devising problem

# demonstration tools
def play(env, agent, t_max=1000):
    brain_name = env.brain_names[0]  # this is specific for the Unity Environment
    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    states = env_info.vector_observations  # get the initial state
    for _ in range(t_max):
        states = to_tensor(states)  # states is now tensor type
        actions, _, _, _ = agent.forward(states)
        env_info = env.step(actions.numpy())[brain_name]  # send the action to the environment
        states = env_info.vector_observations  # get the next state
        dones = env_info.local_done  # see if episode has finished
        if np.any(dones):
            break

def play_random(env, t_max=1000):
    brain_name = env.brain_names[0]  # this is specific for the Unity Environment
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    states = env_info.vector_observations  # get the initial state
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    for _ in range(t_max):
        actions = np.random.randn(num_agents, action_size)  # select an action (for each agent)
        actions = np.clip(actions, -1, 1)
        env_info = env.step(actions)[brain_name]  # send the action to the environment
        dones = env_info.local_done  # see if episode has finished
        if np.any(dones):
            break


# misc
def random_color(choice=False):
    "select the color code from "
    lib=['r', 'b', 'g', 'k', 'y', 'c', 'm']
    if isinstance(choice, int):
        if choice<=6:
            return lib[choice]
        else:
            raise ValueError("choice value should be less than/equal to 6")
    else:
        return random.choice(lib)
