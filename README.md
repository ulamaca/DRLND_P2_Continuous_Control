[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# Project 2: Continuous Control

### Introduction

For this project, I implemented a PPO reinforcement learning agent to solve Unit Reacher continuous control [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.
The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Distributed Training

For this project, two separate versions of the Unity environment are available:
- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.

I trained my agent in the second environment to utilize the parallelizable nature of PPO algorithms.
    

### Solving the Environment

#### Option 1: Solve the First Version

The task is episodic, and in order to solve the environment,  your agent must get an average score of +30 over 100 consecutive episodes.

#### Option 2: Solve the Second Version

The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents.  In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30.  

### Prerequisites
1. Please first setup a Python3 [Anaconda](https://www.anaconda.com/download) environment. 
2. Then install the requirements for the project through:
```
pip install -r requirement.txt
```
3. clone the repo
```
git clone git@github.com:ulamaca/DRLND_P2_Continuous_Control.git
```
4. Follow the instructions to download the multi-agent version environment from the [Getting Started](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control) section in Udacity DRLND repo.

5. Place the env directory in the root of the project and rename it as "Reacher_Multi_Linux"
 
### Instructions
1. To train a PPO agent from scratch, execute in the command line:
```
python run.py  
```
After trained, two files will be saved in ./data/ppo_gae: progress.txt and checkpoint.pth. progress.txt saves the training score traces and checkpoint.pth is the model parameters of the trained agent.
More detailed instructions can be found using:

2. To get statistics plots after training, execute:
```
python plot.py -l ppo_gae
``` 

3. To see how your favorite agent plays, use
```
python play.py -p path/to/model-params 
```
If you did not get one, try out
```
plot.py play.py -p ./data/saved/checkpoint.pth
```