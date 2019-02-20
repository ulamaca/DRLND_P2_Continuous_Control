[image1]: ./data/ppo_gae.png 
[image2]: ./data/ppo_algorithm1.png
[anime1]: ./data/ppo_trained_animation.gif
### **Algorithms**
In this project, I used Proximal Policy Optimization (PPO) to solve Unity Reacher Environment. PPO aims to solve a major limitation in policy gradients methods: data inefficeincy. 
Each trajectory can validly be used for updating the policy network only once in Policy Gradients (PG). This is wasteful especially when the generation process is slow, resource-consuming or even dangerous.  
With tricks of importance sampling, surrogate objectives, and surrogate clipping, the policy network in PPO can then be updated multiple times using a generated trajectory (generated from an "old policy") without losing track from the true objective function. 
This technique enhances data efficiency greatly by creating off-policy learning (improving a policy other than the trajectory generating one) alike capability for PG algorithm. 

### **Implementation**
My implementation is based on the idea of Algorithm 1 in John Schulman et al's 2017 paper: ![Algorithm 1][image2]


where policy is a Gaussian whose mean and variance are tuneable (the mean $\mu$ is represented by a multi-layer fully perceptron whereas the variance is parametrized seperately by another independent set of variables). 
The value function is constructed by a NN sharing the main body with the policy and the output a 1-d state value.
The advantage is estimated through generalized value estimation (GAE) and it is standardized over trajectories.

In addition, the actor loop for trajectories collection in Algorithm 1 is a perfect fit for parallelization. Parallelization enables efficient data collection (which may accelerate learning) and gathers potentially diverse experience data via, for example, adopting different exploration strategy in each thread. I thus choose multi-agent version of the environment to take advantage of such nature. A buffer (MAReacherTrajectories) is created for storing trajectories using torch.utils.data to organize the data format and mini-batch generation. Note that the current implementation using only fixed exploring strategy (the current policy)

### **Results**  

#### **Statistics**

![Figure1][image1]
The agent solves the environment in 187 episodes. The total time elapsed is 1198.3300256729126 second (~20 minutes in my Dell XPS13 laptop, 4CPU, 16G memory)

**Video recording of a trained agent**
![trained agent][anime1]: ./data/ppo_trained_animation.gif

### **Future Work**
- Future experiments:
    - Compare performance of different value estimators to see if GAE is really the best. The currently available options are:
        1. Monte-Carlo value estimate: the future reward in PPO lecture and note that critic is not necessary in this case. 
        2. Direct value estimate: using the state value function for each state directly
        3. Advantage: i.e. 1.-2., the advantage estimate used in A3C paper 
    - Make it work in OpenAI Gym environments to see how well they work and do benchmarking.
- The current version is using multiple agents but not written in a genuinely parallel way. Therefore, it will be worthy to delve deeper into parallelization tools such as [MPI](http://mpitutorial.com/tutorials/mpi-introduction/) to further boost learning efficiency.  
- Compare the current implementation to SOTA methods able to solve continuous problems such as Soft Actor-Critic and D4PG/DDPG to compare their individual learning efficacy/efficiency.

### **Reference**
Research Papers:
- [Proximal Policy Optimization 2017](https://www.nature.com/articles/nature14236)
- [A3C 2016](https://arxiv.org/abs/1602.01783)
- [Soft Actor Critic 2018](https://arxiv.org/abs/1801.01290)

Related works:
- [tnakae: Udacity-DeepRL-p2-Continuous](https://github.com/tnakae/Udacity-DeepRL-p2-Continuous) (greatly appreciated for the work/author, really helpful for my implementation)
- [A detailed example of how to generate your data in parallel with PyTorch](https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel)



### **Appendix** 
Hyperparameters

| Hyperparameter                      | Value |
| ----------------------------------- | ----- |
| Agent Model Type                    | MLP   |
| Policy Distribution                 | Gaussian |
| Agent Model Arch (Policy)           | [in, 512, 256, out] |
| Agent Model Arch (Critic)*1         | [in, 512, 256, 1] |
| Trajectory Length (t_max)           | 1000 steps|
| PPO SGD Epoch                       | 3    |
| PPO mini-batch size                 | 256  |
| gamma (discount factor)          | 0.99  |
| GAE lambda*2                          | 0.96  |
| Optimizer                           | Adam  |
| Learning rate                       | 1e-4  |
| epsilon (surrogate clip rate)    | 0.1   |
| beta (entropy regularization strength) | 0.01   |
| gradient clip                       | 0.5 |

- *1. The parameters of the first two layers of Policy(Actor) and Critic are shared. 
- *2. GAE lambda is not exposed for tuning in current implementation (2019.2.20)