"""
The vanilla policy gradient method, REINFORCE 
[1] Yan Duan, et al., "Benchmarking Deep Reinforcement Learning for 
    Continuous Control", ICML 2016.
[2] Richard Sutton, Andrew Barto, "Reinforcement learning: An introduction",
    2018 (http://incompleteideas.net/book/the-book-2nd.html).
"""

import torch
import numpy as np
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)
from torch.distributions.kl import kl_divergence

from myrllib.utils.torch_utils import (weighted_mean, detach_distribution,
                                       weighted_normalize)
from myrllib.utils.optimization import conjugate_gradient
from myrllib.baselines.baseline import LinearFeatureBaseline
from torch.optim import Adam, SGD

def norm_01(weights, epsilon=1e-10):
    return (weights-weights.min()+epsilon)/(weights.max()-weights.min()+epsilon)
def norm_sum1(weights, epsilon=1e-10):
    weights += epsilon
    return weights/weights.sum()*weights.size(0)

class REINFORCE(object):
    def __init__(self, policy, lr=1e-2, device='cpu', 
            pr_smooth=1e-20, iw_smooth=None):
        self.policy = policy 
        self.lr = lr
        self.opt = SGD(policy.parameters(), lr=lr)
        self.pr_smooth = pr_smooth 
        self.iw_smooth = iw_smooth
        self.to(device)

    def inner_loss(self, episodes, pr=False, iw=False):
        returns = episodes.returns
        pi = self.policy(episodes.observations)
        log_probs = pi.log_prob(episodes.actions)
        if log_probs.dim() > 2:
            log_probs = torch.sum(log_probs, dim=2)

        ### apply importance sampling / policy relaxation
        if pr:
            log_probs_old = episodes.log_probs 
            if log_probs_old.dim() > 2:
                log_probs_old = torch.sum(log_probs_old, dim=2)
            ### compute p(x)/q(x), estimate p(x) under samples from q(x)
            importance_ = torch.ones(episodes.rewards.size(1)).to(self.device)
            importances = []
            for log_prob, log_prob_old, mask in zip(log_probs, log_probs_old,
                    episodes.mask):
                importance_ = importance_*torch.div(
                        log_prob.exp() * mask + self.pr_smooth, 
                        log_prob_old.exp() * mask + self.pr_smooth)
                importance_ = weighted_normalize(importance_)
                importance_ = importance_ - importance_.min()
                importances.append(importance_)
            importances = torch.stack(importances, dim=0)
            importances = importances.detach()

        ### apply importance weighting 
        if iw:
            weights = torch.sum(returns*episodes.mask,dim=0)/torch.sum(
                    episodes.mask,dim=0)
            weights = weights.max() - weights 
            weights = weighted_normalize(weights)
            weights = weights - weights.min()
            if self.iw_smooth is not None:
                weights = weights + self.iw_smooth 
                weights = weights/weights.sum()*weights.size(0)

        if pr and iw:
            ### the proposed method, PR + IW
            t_loss = torch.sum(importances * returns * log_probs * episodes.mask, 
                    dim=0)/torch.sum(episodes.mask,dim=0)
            loss = - torch.mean(weights*t_loss)
        elif pr and not iw:
            ### only apply Policy Relaxation
            loss = - weighted_mean(log_probs * returns * importances, 
                    weights=episodes.mask)
        elif not pr and iw:
            ### only apply Importance Weighting
            t_loss = torch.sum(returns*log_probs*episodes.mask,dim=0)/torch.sum(
                    episodes.mask,dim=0)
            loss = - torch.mean(weights*t_loss)
        else:
            ### the baseline 
            loss = - weighted_mean(log_probs * returns, weights=episodes.mask)
        return loss


    def step(self, episodes, pr=False, iw=False):
        loss = self.inner_loss(episodes, pr=pr, iw=iw)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


    def to(self, device, **kwargs):
        self.policy.to(device, **kwargs)
        self.device = device



