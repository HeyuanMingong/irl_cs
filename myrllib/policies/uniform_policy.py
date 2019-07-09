import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform

from collections import OrderedDict
from myrllib.policies.policy import Policy, weight_init

class UniformPolicy(Policy):
    """Policy network based on a multi-layer perceptron (MLP), with a 
    `Categorical` distribution output. This policy network can be used on tasks 
    with discrete action spaces (eg. `TabularMDPEnv`). The code is adapted from 
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/sandbox/rocky/tf/policies/maml_minimal_categorical_mlp_policy.py
    """
    def __init__(self, input_size, output_size, low=None, high=None):
        super(UniformPolicy, self).__init__(
            input_size=input_size, output_size=output_size)
        if low is None:
            print('Please provide the action space...')
        self.low = torch.FloatTensor(low) 
        self.high = torch.FloatTensor(high) 

    def forward(self, input, params=None):
        batch_size = input.size(0)
        low = torch.stack([self.low for _ in range(batch_size)],dim=0)
        high = torch.stack([self.high for _ in range(batch_size)],dim=0)

        return Uniform(low, high)
