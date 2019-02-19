import torch
torch.manual_seed(0) # set random seed
from torch.distributions import Categorical, Normal
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

H_SIZE=20
STD_INIT=0.0 # initial std for continous PPO agent

class Critic(nn.Module):


    def __init__(self,s_size=4, h_size=H_SIZE):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class Policy(nn.Module):

    def __init__(self, s_size=4, h_size=H_SIZE, a_size=2):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        #state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def pi_value_tensor(self, state, action):
        '''
        return probability value of pi_(a|s)
        :param state: in [?, d] torch.tensor format
        :param action: in [?, 1] torch.tensor format
        :return: ?
        '''
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        action = torch.tensor([[action]]).long()
        probs = self.forward(state).cpu()
        return probs.gather(1, action)


class PPOPolicy(nn.Module):
    # todo-1, abstract out the model definition and can be defined by users

    def __init__(self, s_size=4, h_size=H_SIZE, a_size=2, continuous=True):
        super(PPOPolicy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc11 = nn.Linear(h_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)
        self.continous_action_space = continuous
        if self.continous_action_space:
            self.log_std = nn.Parameter(torch.ones(1, a_size) * STD_INIT)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc11(x))
        if self.continous_action_space:
            mu = torch.tanh(self.fc2(x))
            std = self.log_std.exp().expand_as(mu)
            dist = Normal(mu, std)
            return dist

        else:
            x = self.fc2(x)
            probs = F.softmax(x, dim=1)
            dist = Categorical(probs)
            return dist

    def act(self, state, training=True):
        state = torch.from_numpy(state).float().to(device)
        dist = self.forward(state)
        action = dist.sample() # todo, action should be formatted properl# y
        if training:
            return action.numpy(), dist.log_prob(action)
        else:
            return action.numpy()

class GymPPOPolicy(nn.Module):
    # todo-1, abstract out the model definition and can be defined by users

    def __init__(self, s_size=4, h_size=30, a_size=2, continuous=True):
        super().__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)
        self.continous_action_space = continuous
        if self.continous_action_space:
            self.log_std = nn.Parameter(torch.ones(1, a_size) * STD_INIT)

    def forward(self, x):
        if len(x.shape) <=1:
            x = x.float().unsqueeze(0).to(device)
        x = F.relu(self.fc1(x))
        if self.continous_action_space:
            mu = torch.tanh(self.fc2(x))
            std = self.log_std.exp().expand_as(mu)
            dist = Normal(mu, std)
            return dist

        else:
            x = self.fc2(x)
            probs = F.softmax(x, dim=1)
            dist = Categorical(probs)
            return dist

    def act(self, state, training=True):
        state = torch.from_numpy(state).float().to(device)
        dist = self.forward(state)
        action = dist.sample()
        # print("action taken shape", action.shape)
        # print("log prob action taken shape", dist.log_prob(action).shape)
        if training:
            return action.numpy(), dist.log_prob(action)
        else:
            return action.numpy()


class PPOActorCritic(PPOPolicy):
    def __init__(self, s_size=4, h_size=H_SIZE, a_size=2, continuous=True):
        super().__init__(s_size, h_size, a_size, continuous)
        self.critic_fc2=nn.Linear(h_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc11(x))
        if self.continous_action_space:
            mu = torch.tanh(self.fc2(x))
            std = self.log_std.exp().expand_as(mu)
            dist = Normal(mu, std)
            value = self.critic_fc2(x)
            return dist, value


class FCNetwork(nn.Module):
    def __init__(self, input_dim, hiddens, func=F.leaky_relu):
        super(FCNetwork, self).__init__()
        self.func =  func

        # Input Layer
        fc_first = nn.Linear(input_dim, hiddens[0])
        self.layers = nn.ModuleList([fc_first])
        # Hidden Layers
        layer_sizes = zip(hiddens[:-1], hiddens[1:])
        self.layers.extend([nn.Linear(h1, h2)
                            for h1, h2 in layer_sizes])

        def xavier(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
        self.layers.apply(xavier)

    def forward(self, x):
        for layer in self.layers:
            x = self.func(layer(x))

        return x


class GaussianPolicyNetwork(nn.Module):
    def __init__(self, state_dim=1, action_dim=1, hiddens=[64, 64]):
        super(GaussianPolicyNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc_hidden = FCNetwork(state_dim, hiddens)
        self.fc_actor = nn.Linear(hiddens[-1], action_dim)
        self.sigma = nn.Parameter(torch.zeros(action_dim))

    def forward(self, states, actions=None):
        phi = self.fc_hidden(states)
        mu = torch.tanh(self.fc_actor(phi))

        dist = torch.distributions.Normal(mu, F.softplus(self.sigma))
        if actions is None:
            actions = dist.sample()
        log_prob = dist.log_prob(actions)
        log_prob = torch.sum(log_prob, dim=-1)
        entropy = torch.sum(dist.entropy(), dim=-1)
        # for unified formatting
        value = None
        return actions, log_prob, entropy, value


class GaussianActorCriticNetwork(GaussianPolicyNetwork):
    def __init__(self, state_dim=1, action_dim=1, hiddens=[64, 64]):
        super().__init__(state_dim, action_dim, hiddens)
        self.fc_critic = nn.Linear(hiddens[-1], 1)

    # todo, may further simplify this part beacuse of highly repeatable
    def forward(self, states, actions=None):
        phi = self.fc_hidden(states)
        mu = F.tanh(self.fc_actor(phi))
        value = self.fc_critic(phi).squeeze(-1)

        dist = torch.distributions.Normal(mu, F.softplus(self.sigma))
        if actions is None:
            actions = dist.sample()
        log_prob = dist.log_prob(actions)
        log_prob = torch.sum(log_prob, dim=-1)
        entropy = torch.sum(dist.entropy(), dim=-1)
        return actions, log_prob, entropy, value

    def state_values(self, states):
        phi = self.fc_hidden(states)
        return self.fc_critic(phi).squeeze(-1)
