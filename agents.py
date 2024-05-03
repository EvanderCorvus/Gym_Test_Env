import torch as tr
import torch.nn as nn
from agent_utils import NNSequential, ReplayBuffer
from torch.distributions import Normal
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import numpy as np
from copy import deepcopy
import torch.nn.functional as F

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class Actor(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        fc = [config["hidden_dims_actor"]]*config['num_hidden_layers_actor']
        self.net = NNSequential([config["state_dim"]]+fc, nn.LeakyReLU, nn.LeakyReLU, batch_norm=False)
        self.mu_layer = nn.Linear(config["hidden_dims_actor"], config["action_dim"])
        # self.mu_layer = nn.Sequential(
        #     nn.Linear(config["hidden_dims"], action_dim),
        #     nn.Tanh()
        # )
        
        self.log_std_layer = nn.Linear(config["hidden_dims_actor"], config["action_dim"])
        self.act_scaling = tr.tensor(config['act_scaling']).float().to(device)

    def forward(self, state, deterministic = False, with_logprob = False):

        net_out = self.net(state)
        mu = self.mu_layer(net_out) # self.act_scaling*

        if deterministic:
            pi_action = mu
        else:
            log_std = self.log_std_layer(net_out)
            log_std = tr.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
            std = tr.exp(log_std)

            pi_distribution = Normal(mu, std)
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)

            pi_action = self.act_scaling*tr.tanh(pi_action)
            
            return pi_action, logp_pi #tr.clamp(pi_action, -self.act_scaling, self.act_scaling)
        else:
            return self.act_scaling*tr.tanh(pi_action)

        # Squash distribution
        #pi_action = tr.tanh(pi_action)*tr.pi/2
        # tr.clamp(pi_action, 0, 2*tr.pi)
        # pi_action % (2*tr.pi)
        # tr.atan2(pi_action[:, 0], pi_action[:, 1])

class Critic(nn.Module):
    def __init__(self, config):
        super().__init__()
        fc = [config["hidden_dims_critic"]]*config['num_hidden_layers_critic']
        self.net = NNSequential([config['state_dim']+config['action_dim']] + fc + [1], nn.LeakyReLU)
        
    def forward(self, state, action):
        state_action = tr.cat((state, action), dim=1)
        out = self.net(state_action)
        return out
    
class TwinCritics(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.net1 = Critic(config)
        self.net2 = Critic(config)

    def forward(self, state, action):
        return self.net1(state, action), self.net2(state, action)
    
    def Q1(self, state, action):
        return self.net1(state, action)
          
class SACAgent(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.gamma = config["future_discount_factor"]
        self.entropy_coeff = config["entropy_coeff"]
        self.entropy_decay_factor = config["entropy_decay_factor"]
        self.polyak_tau = config["polyak_tau"]
        self.batch_size = config["batch_size"]
        self.grad_clip_critic = config["grad_clip_critic"]
        self.grad_clip_actor = config["grad_clip_actor"]
        self.replay_buffer = ReplayBuffer(config['buffer_size'])
        self.device = device

        self.actor = Actor(config, device)
        self.critic = TwinCritics(config)

        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)

        for p in self.target_actor.parameters():
            p.requires_grad = False
        for p in self.target_critic.parameters():
            p.requires_grad = False

        self.actor_optimizer = tr.optim.Adam(self.actor.parameters(),
                                            lr=config["learning_rate_actor"])
        self.critic_optimizer = tr.optim.Adam(self.critic.parameters(), 
                                              lr=config["learning_rate_critic"])

        self.actor_scheduler = StepLR(self.actor_optimizer,
                                    step_size=config['n_epochs']//4,
                                    gamma=config['gamma_actor'])
        
        self.critic_scheduler = StepLR(self.critic_optimizer,
                                    step_size=config['n_epochs']//4,
                                    gamma=config['gamma_critic'])
        
    def act(self, state, deterministic = True):
        with tr.no_grad():
            return self.actor(state, deterministic)
        
    def _update_target_networks(self):
        with tr.no_grad():
            for p, p_target in zip(self.actor.parameters(), self.target_actor.parameters()):
                p_target.data.mul_(self.polyak_tau)
                p_target.data.add_((1-self.polyak_tau) * p.data)
            for p, p_target in zip(self.critic.parameters(), self.target_critic.parameters()):
                p_target.data.mul_(self.polyak_tau)
                p_target.data.add_((1-self.polyak_tau) * p.data)

    def update(self):
        state, action, reward, next_state = self.replay_buffer.sample(self.batch_size)
        # Move to GPU
        state = tr.tensor(state, dtype=tr.float32, device=self.device)
        action = tr.tensor(action, dtype=tr.float32, device=self.device)
        reward = tr.tensor(reward, dtype=tr.float32, device=self.device)
        next_state = tr.tensor(next_state, dtype=tr.float32, device=self.device)
        shapes = [state.shape, action.shape, reward.shape, next_state.shape]
        # Update Critic
        Q1, Q2 = self.critic(state, action)
        with tr.no_grad():
            action_next, logp_pi = self.actor(next_state, with_logprob=True)
            Q1_next, Q2_next = self.target_critic(next_state, action_next)
            Q_next = tr.min(Q1_next, Q2_next)
            logp_pi = logp_pi.unsqueeze(1)
            if Q_next.shape != logp_pi.shape: raise Exception('Shape Missmatch (Q_next, logp_pi):', Q_next.shape, logp_pi.shape)
            target = reward + self.gamma * (Q_next - self.entropy_coeff * logp_pi)

        if target.shape != Q1.shape: raise Exception(Q1.shape, target.shape, reward.shape, Q_next.shape, logp_pi.shape)
        
        loss_critic = F.mse_loss(Q1, target) + F.mse_loss(Q2, target)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        tr.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_critic)
        self.critic_optimizer.step()

        # Update Actor
        self.actor_optimizer.zero_grad()
        for p in self.critic.parameters():
            p.requires_grad = False

        proposed_action, logp_pi = self.actor(state, with_logprob=True)
        Q1, Q2 = self.critic(state, proposed_action)
        Q = tr.min(Q1, Q2)
        logp_pi = logp_pi.unsqueeze(1)
        if Q.shape != logp_pi.shape: raise Exception('Shape Missmatch (Q, logp_pi):', Q.shape, logp_pi.shape)
        loss_actor = (self.entropy_coeff * logp_pi - Q).mean()
        loss_actor.backward()
        tr.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_actor)
        self.actor_optimizer.step()

        for p in self.critic.parameters():
            p.requires_grad = True
        
        return loss_actor, loss_critic
    
    def decay_entropy(self):
        self.entropy_coeff *= self.entropy_decay_factor



class MinimalAgent(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.gamma = config["future_discount_factor"]
        self.polyak_tau = config["polyak_tau"]
        self.batch_size = config["batch_size"]
        self.grad_clip_critic = config["grad_clip_critic"]
        self.grad_clip_actor = config["grad_clip_actor"]
        self.replay_buffer = ReplayBuffer(config['buffer_size'])
        self.device = device

        self.loss_function = nn.MSELoss()

        self.actor = Actor(config)
        self.critic = Critic(config)

        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)

        for p in self.target_actor.parameters():
            p.requires_grad = False
        for p in self.target_critic.parameters():
            p.requires_grad = False

        self.actor_optimizer = tr.optim.Adam(self.actor.parameters(), lr=config["learning_rate_actor"])
        self.critic_optimizer = tr.optim.Adam(self.critic.parameters(), lr=config["learning_rate_critic"])

        self.actor_scheduler = StepLR(self.actor_optimizer,
                                    step_size=config['n_epochs']//4,
                                    gamma=config['gamma_actor'])
        
        self.critic_scheduler = StepLR(self.critic_optimizer,
                                    step_size=config['n_epochs']//4,
                                    gamma=config['gamma_critic'])
    
    def act(self, state, deterministic = True):
        with tr.no_grad():
            return self.actor(state, deterministic)
        
    def _update_target_networks(self):
        with tr.no_grad():
            for p, p_target in zip(self.actor.parameters(), self.target_actor.parameters()):
                p_target.data.mul_(self.polyak_tau)
                p_target.data.add_((1-self.polyak_tau) * p.data)
            for p, p_target in zip(self.critic.parameters(), self.target_critic.parameters()):
                p_target.data.mul_(self.polyak_tau)
                p_target.data.add_((1-self.polyak_tau) * p.data)

    def update(self, step):
        state, action, reward, next_state = self.replay_buffer.sample(min(self.batch_size, len(self.replay_buffer)))

        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()

        Q = self.critic(state, action)
        with tr.no_grad():
            noise = tr.normal(0, self.target_noise_std, size=(min(self.batch_size, len(self.replay_buffer)), 1)).to(self.device)
            next_action = tr.clamp(self.target_actor(next_state) + noise, -tr.pi, tr.pi)
            Q_next = self.target_critic(next_state, next_action)
            target = reward + self.gamma * Q_next

        loss_critic = self.loss_function(Q, target)
        loss_critic.backward()

        self.critic_optimizer.step()

        proposed_action = self.actor(state)#.unsqueeze(1)
        Q_target = self.target_critic(state, proposed_action)
        loss_actor = -Q_target.mean()
        loss_actor.backward()
        self.actor_optimizer.step()

        self._update_target_networks()

        return loss_actor, loss_critic
    

class MergedSAC(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.gamma = config["future_discount_factor"]
        self.replay_buffer = ReplayBuffer(config['buffer_size'])
        self.batch_size = config["batch_size"]
        self.device = device

        self.shared_layer = NNSequential([config["state_dim"]] + config["shared_hidden_dims"], nn.LeakyReLU, batch_norm=False)
        # Actor Layers
        self.mu_layer = nn.Linear(config["shared_hidden_dims"], config["action_dim"])
        self.log_std_layer = nn.Linear(config["shared_hidden_dims"], config["action_dim"])
        self.act_scaling = tr.tensor(config['act_scaling']).float().to(device)

        # Critic Layers
        self.critic_net = nn.Linear(config["shared_hidden_dims"]+config['acion_dim'], 1)


    def forward(self, state):
        out = self.shared_layer(state)
        return out
    
    def actor(self, state, deterministic = False, with_logprob = False):
        net_out = self.shared_layer(state)
        mu = self.mu_layer(net_out) # self.act_scaling*

        if deterministic:
            pi_action = mu
        else:
            log_std = self.log_std_layer(net_out)
            log_std = tr.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
            std = tr.exp(log_std)

            pi_distribution = Normal(mu, std)
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)

            pi_action = self.act_scaling*tr.tanh(pi_action)
            
            return pi_action, logp_pi #tr.clamp(pi_action, -self.act_scaling, self.act_scaling)
        else:
            return self.act_scaling*tr.tanh(pi_action)
        
    def critic(self, state, action):
        pass
