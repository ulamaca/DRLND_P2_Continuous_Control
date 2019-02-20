### **Algorithms**
In this project, I used Proximal Policy Optimization (PPO, Shulman2017) to solve Reacher Environment. PPO aims to solve a major limitation in policy gradients methods: data inefficeincy. Each trajectory can validly be used for updating the policy network once in PG, which is wasteful, especially when the generation process is slow, resource-consuming or even dangerous. With tricks of importance sampling, surrogate objectives, and surrogate clipping, the policy network can then be updated multiple times using a generated trajectory (generated from an "old policy") without losing track from the true objective function. This technique enhances data effiency greatly by creating off-policy learning (improving a policy other than the trajectory generating one) alike capability for PG algorithm. 

### **Implementation**
My implementation is based on the idea of Algorithm 1 in John Schulman et al's 2017 paper: 


where policy ($\pi$) is a Gaussian whose mean and variance are tuneable (the mean $\mu$ is represented by a multi-layer fully perceptron whereas the variance is parametrized seperately by another independent set of variables). The advantage is estimated through generalized value estimation (GAE).

In addition, the actor loop for trejectories collection in Algorithm 1 is a perfect fit for parallelization. Parallelization enables efficient data collection (which may accelerate learning) and gathers potentially diverse experience data via, for example, adopting different exploration strategy in each thread. I thus choose multi-agent version of the environment to take advantage of such nature. A buffer (MAReacherTrajectories) is created for storing trajectories using torch.utils.data to organize the data format and mini-batch generation. Note that the current implementation using only fixed exploring strategy (the current policy)

### **Results**  

#### **Statistics**
[image1]: ./data/ppo_gae.png 

![Figure1][image1]
The agent solves the environment in 187 episodes. The total time elapsed is 1198.3300256729126 second (~20 minutes in my Dell XPS13 laptop, 4CPU, 16G memory)

**Video recording of a trained agent**

### **Future Work**
- compare the result with other PG based methods
- play with the latest Soft-Actor Critic.

### **Reference**
Research Papers:
- [Proximal Policy Optimization 2017](https://www.nature.com/articles/nature14236)
- [Dueling DQN 2016](https://arxiv.org/abs/1511.06581)
- [Double DQN 2016](https://arxiv.org/abs/1509.06461)

Related Projects:
- [tnakae: Udacity-DeepRL-p2-Continuous]
(https://github.com/tnakae/Udacity-DeepRL-p2-Continuous)



### **Appendix**
1. Key equations and the corresponding lines of codes in the project are summarized in ![./equations.png](./equations.png) 
2. Hyperparameters

| Hyperparameter                      | Value |
| ----------------------------------- | ----- |
| Agent Model Type                    | MLP   |
| Agent Model Arch                    | [in, 20, 20, out] |
| Update (Learning) Frequency         | every 4 steps |
| Replay buffer size                  | 1e5   |
| Batch size                          | 64    |
| $\gamma$ (discount factor)          | 0.99  |
| Optimizer                           | Adam  |
| Learning rate                       | 5e-4  |
| Soft-Update (*1)                      | True  |
| $\tau$ (soft-update mixing rate)    | 1e-3  |
| $\epsilon$ (exploration rate) start | 1.0   |
| $\epsilon$ minimum                  | 0.1   |
| $\epsilon$ decay                    | 0.995 |
