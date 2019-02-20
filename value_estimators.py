import torch


def GAE(rewards, values, last_dones, last_values, gamma=0.99, gae_lambda=0.96):
        t_max, n_agent = rewards.shape

        # Create empty buffer
        GAE = torch.zeros_like(rewards).float()
        returns = torch.zeros_like(rewards).float()


        # Set start values
        GAE_current = torch.zeros(n_agent).float()
        returns_current = last_values * last_dones
        values_next = last_values * last_dones

        for t in reversed(range(t_max)):
            values_current = values[t]
            rewards_current = rewards[t]

            # Calculate TD Error
            td_error = rewards_current + gamma * values_next - values_current
            # Update GAE, returns
            GAE_current = td_error + gamma * gae_lambda * GAE_current
            returns_current = rewards_current + gamma * returns_current
            # Set GAE, returns to buffer
            GAE[t] = GAE_current
            returns[t] = returns_current

            values_next = values_current

        return GAE, returns


def A2C(rewards, values, last_dones, last_values, value_type='advantage', gamma=0.99):
    if not (value_type in ['advantage', 'mc', 'state-value']):
        raise TypeError("Not a valid type of value estimator")
    t_max, n_agent = rewards.shape

    # Create empty buffer
    returns = torch.zeros_like(rewards).float()

    # Set start values
    returns_current = last_values * last_dones

    for t in reversed(range(t_max)):
        rewards_current = rewards[t]
        returns_current = rewards_current + gamma * returns_current
        returns[t] = returns_current

    if value_type =='advantage':
        return returns-values, returns
    elif value_type == 'state-value':
        return values, returns
    elif value_type =='mc':
        return returns, returns
