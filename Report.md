My PPO implementation follows basically from the John Schulman's 2017 paper, the algorithm x.x. 
PPO tries to solve the data inefficiency problem in ordinary PG problem.
The current version of the code is using future reward formulation as value estimate.

Importance sampling, and Surrogate loss clipping are the two main tricks here.
I use torch.dataset to construct the trajectories memory for updating

My algorithm is limited to reach around 60-70 in CartPole problem. Potential problem still exist in the code, 
which requires more dedicated look (ask for help)

I think my data formatting is the biggest problem
    - I should look back to my old reinforce implementation
    - also learn from tnakae to see his data formatting

More things to do, I may implement    

Reference:
github: tnakae
- data formatting:
    t_max=128, n_agents=20
    states, data shape: torch.Size([128, 20, 33])
    actions, data shape: torch.Size([128, 20, 4])
    next_states, data shape: torch.Size([128, 20, 33])
    rewards, data shape: torch.Size([128, 20])
    log_probs, data shape: torch.Size([128, 20])
    values, data shape: torch.Size([128, 20])
    dones, data shape: torch.Size([128, 20])
    advantages, data shape: torch.Size([128, 20])
    then, it will be reformatted
    returns, data shape: torch.Size([128, 20])
    advantages shape: torch.Size([2560])
    advantages_batch shape: torch.Size([128])



https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel

