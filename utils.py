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