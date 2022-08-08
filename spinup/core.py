"""
Credits: OpenAI Spin Up

https://github.com/openai/spinningup.git

@article{SpinningUp2018,
    author = {Achiam, Joshua},
    title = {{Spinning Up in Deep Reinforcement Learning}},
    year = {2018}
}
"""

import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

import model

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(64,64), activation=nn.ReLU):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes=(64,64), activation=nn.ReLU):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.


class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64,64), activation=nn.ReLU, *args, **kwargs):
        super().__init__()
        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)
        self.use_gpu = False

    def reset(self):
        pass

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]


class ImageEncoderShim(nn.Module):
    def __init__(self, image_encoder, max_ep_len):
        super().__init__()
        # TOOD(jiahang): this is a bad design
        self.hc = None
        self.max_ep_len = max_ep_len

        self.image_encoder = image_encoder
        self.output_size = image_encoder.lstm_predictor.hidden_size
    
    def reset(self):
        self.hc = None

    def forward(self, obs, simulation_mode=False):
        """ obs: either (64, 64) or (1000, 64, 64) """
        if simulation_mode:
            # simulation mode: remember hidden state
            assert obs.dim() == 2
            H, W = obs.shape
            obs = obs.view(1, 1, 1, H, W).expand(1, 1, 3, H, W)
            y, self.hc, _ = self.image_encoder(obs, self.hc)
            y = y.view(-1)
        else:
            # training mode
            assert obs.dim() == 3
            self.hc_prev = None
            BT, H, W = obs.shape
            assert BT % self.max_ep_len == 0

            B = BT // self.max_ep_len
            T = self.max_ep_len

            obs = obs.view(B, T, 1, H, W).expand(B, T, 3, H, W)
            y = self.image_encoder(obs)[0] # B, T, H
            y = y.reshape(BT, -1)
        return y


class ImageActor(Actor):
    def __init__(self, freeze, image_encoder, act_dim):
        super().__init__()

        # hack: force torch to not register parameters if frozen
        self.freeze = freeze
        if freeze:
            self.image_encoder = [ image_encoder ]
        else:
            self.image_encoder = image_encoder

        self.actor = MLPGaussianActor(image_encoder.output_size, act_dim)

    def get_encoder(self):
        if self.freeze:
            return self.image_encoder[0]
        else:
            return self.image_encoder

    def _distribution(self, obs):
        y = self.get_encoder()(obs)
        return self.actor._distribution(y)
    
    def _fast_distribution(self, y):
        return self.actor._distribution(y)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)


class ImageCritic(nn.Module):
    def __init__(self, freeze, image_encoder):
        super().__init__()

        # hack: force torch to not register parameters if frozen
        self.freeze = freeze
        if freeze:
            self.image_encoder = [ image_encoder ]
        else:
            self.image_encoder = image_encoder
        self.critic = MLPCritic(image_encoder.output_size)

    def get_encoder(self):
        if self.freeze:
            return self.image_encoder[0]
        else:
            return self.image_encoder

    def forward(self, obs):
        y = self.get_encoder()(obs)
        return self.critic(y)
    
    def fast_forward(self, y):
        return self.critic(y)


class ImageActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, max_ep_len,
                 savedir=None, freeze=True,
                 rewards="agent_x,agent_y,target_x,target_y"):
        super().__init__()
        self.use_gpu = torch.cuda.is_available()
        image_encoder = model.RewardPredictor([
            r.strip() for r in rewards.split(',') if len(r.strip()) > 0 ],
            use_gpu=self.use_gpu
        )
        if savedir is not None:
            image_encoder.load_state_dict(torch.load(savedir))
            print(f'Loaded {savedir}')
        else:
            print('Start from Scratch')
        if freeze:
            for param in image_encoder.parameters():
                param.require_grad = False
            print("Image encoder weights FROZEN")
        else:
            print("Image encoder is TRAINABLE")
        assert observation_space.shape[0] == image_encoder.image_encoder.R

        self.image_encoder = ImageEncoderShim(image_encoder, max_ep_len)
        self.pi = ImageActor(freeze, self.image_encoder, action_space.shape[0])
        self.v  = ImageCritic(freeze, self.image_encoder)

    def reset(self):
        self.image_encoder.reset()

    def step(self, obs):
        # simulation mode
        with torch.no_grad():
            y = self.image_encoder(obs, simulation_mode=True)
            pi = self.pi._fast_distribution(y)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v.fast_forward(y)
        if self.use_gpu:
            return a.to('cpu').numpy(), v.to('cpu').numpy(), logp_a.to('cpu').numpy()
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]


class SimpleImageEncoderShim(nn.Module):
    def __init__(self, image_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.output_size = image_encoder.C
    
    def reset(self):
        pass

    def forward(self, obs, simulation_mode=False):
        """ obs: either (64, 64) or (1000, 64, 64) """
        if simulation_mode:
            # simulation mode: remember hidden state
            assert obs.dim() == 2
            H, W = obs.shape
            obs = obs.view(1, 1, 1, H, W).expand(1, 1, 3, H, W)
            y = self.image_encoder(obs)
            y = y.view(-1)
        else:
            # training mode
            assert obs.dim() == 3
            BT, H, W = obs.shape
            obs = obs.view(1, BT, 1, H, W).expand(1, BT, 3, H, W)
            y = self.image_encoder(obs)
            y = y.reshape(BT, -1)
        return y


class SimpleImageActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, max_ep_len,
                 savedir=None, freeze=True,
                 rewards="agent_x,agent_y,target_x,target_y"):
        super().__init__()
        self.use_gpu = torch.cuda.is_available()
        image_encoder = model.RewardPredictor([
            r.strip() for r in rewards.split(',') if len(r.strip()) > 0 ],
            use_gpu=self.use_gpu
        )
        if savedir is not None:
            image_encoder.load_state_dict(torch.load(savedir))
            print(f'Loaded {savedir}')
        else:
            print('Start from Scratch')
        if freeze:
            for param in image_encoder.parameters():
                param.require_grad = False
            print("Image encoder weights FROZEN")
        else:
            print("Image encoder is TRAINABLE")
        assert observation_space.shape[0] == image_encoder.image_encoder.R

        self.image_encoder = SimpleImageEncoderShim(image_encoder.image_encoder)
        self.pi = ImageActor(freeze, self.image_encoder, action_space.shape[0])
        self.v  = ImageCritic(freeze, self.image_encoder)

    def reset(self):
        pass

    def step(self, obs):
        # simulation mode
        with torch.no_grad():
            y = self.image_encoder(obs, simulation_mode=True)
            pi = self.pi._fast_distribution(y)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v.fast_forward(y)
        if self.use_gpu:
            return a.to('cpu').numpy(), v.to('cpu').numpy(), logp_a.to('cpu').numpy()
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]
