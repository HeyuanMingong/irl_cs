#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This the code for the paper:
[1] Zhi Wang, Han-Xiong Li, and Chunlin Chen, "Incremental Reinforcement Learning 
    in Continuous Spaces via Policy Relaxation and Importance Weighting for 
    Dynamic Environments", IEEE Transactions on Neural Networks and Learning 
    Systems, 2019.

The implementation consists of two steps:
1. Train the policy network in an original environment, acquire the model parameters
2. In a new environment, initialze the policy parameters from the original one,
    and continue to train the policy using all tested methods including
    - Random, a baseline that trains from scratch
    - Pretrained, a baseline that directly trains in the new environment
    - PRPG, a baseline of policy reuse policy gradient
    - PR, a variant of the proposed method that only applies Policy Relaxation
    - IW, a variant of the proposed method that only applies Importance Weighting
    - PR+IW, the proposed method

https://github.com/HeyuanMingong/irl_cs
"""

### common lib
import sys
import gym
import numpy as np
import argparse 
import torch
from tqdm import tqdm
import os
import time 
from torch.optim import Adam, SGD 
import scipy.io as sio
import copy 

### personal lib
from myrllib.episodes.episode import BatchEpisodes 
from myrllib.samplers.sampler import BatchSampler 
from myrllib.policies import NormalMLPPolicy 
from myrllib.algorithms.reinforce import REINFORCE 
from myrllib.algorithms.trpo import TRPO 


######################## Arguments ############################################
parser = argparse.ArgumentParser()
parser.add_argument('--num_workers', type=int, default=4, 
        help='number of cpu processors for parallelly sampling of gym environment')
parser.add_argument('--batch_size', type=int, default=20, 
        help='number of rollouts/learning episodes in one policy iteration')
parser.add_argument('--hidden_size', type=int, default=100,
        help='hidden size of the policy network')
parser.add_argument('--num_layers', type=int, default=2,
        help='number of hidden layers of the policy network')
parser.add_argument('--num_iter', type=int, default=100,
        help='number of policy iterations')
parser.add_argument('--lr', type=float, default=1e-2,
        help='learning rate, if REINFORCE algorithm is used')
parser.add_argument('--output', type=str, default='output/navi_v1',
        help='output folder for saving the experimental results')
parser.add_argument('--model_path', type=str, default='saves/navi_v1',
        help='the folder for saving and loading the pretrained model')
parser.add_argument('--stage', type=str, default='finetune',
        help='pretrain or finetune, in the original or new environment')
parser.add_argument('--algorithm', type=str, default='reinforce',
        help='reinforce or trpo, the base algorithm for policy gradient')
parser.add_argument('--iw', action='store_true', default=False,
        help='using the Importance Weighting method or not')
parser.add_argument('--pr', action='store_true', default=False,
        help='using the Policy Relaxation method or not')
parser.add_argument('--priw', action='store_true', default=False,
        help='using the proposed method (PR+IW) or not')
parser.add_argument('--prpg', action='store_true', default=False,
        help='using the PRPG baseline or not')
parser.add_argument('--pretrained', action='store_true', default=False,
        help='using the Pretrained baseline or not')
parser.add_argument('--random', action='store_true', default=False,
        help='using the Random baseline or not')
parser.add_argument('--env', type=str, default='Navigation2D-v1')
parser.add_argument('--task', nargs='+', type=float, default=None,
        help='the randomly chosen task in the original or new environment')
parser.add_argument('--device', type=str, default='cpu')
args = parser.parse_args()
print(args)
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
start_time = time.time()


####################### Hyperparameters ######################################
"""
If the rewards are positive, the Importance Weighting assigns a weight to 
a learning episode that is proportional to its received reward; IW_INV = False.
Else, the rewards are negative, the weight is inversely proportional to the 
received reward; IW_INV = True.
"""
IW_INV = True

### for the PRPG method
NU = 0.8; RMAX = 200; PSI = 1.0; UPSILON = 0.01

### for the Policy Relaxation implementations
PR_SMOOTH = 0.1; RELAX_ITERS = 1

### task information
TASK = args.task


######################## Small functions ######################################
### build a learner given a policy network
def generate_learner(policy, pr_smooth=1e-20):
    if args.algorithm == 'trpo':
        learner = TRPO(policy, pr_smooth=PR_SMOOTH, iw_inv=IW_INV, device=device)
    else:
        learner = REINFORCE(policy, lr=args.lr, pr_smooth=PR_SMOOTH, device=device)
    return learner 


######################## Main Functions #######################################
### build a sampler given an environment
sampler = BatchSampler(args.env, args.batch_size, num_workers=args.num_workers) 
state_dim = int(np.prod(sampler.envs.observation_space.shape))
action_dim = int(np.prod(sampler.envs.action_space.shape))
print('state dim: %d; action dim: %d'%(state_dim,action_dim))


"""
'Navigation2D-v1': the task is to reaching a dynamic goal by a point agent
'Navigation2D-v2': reaching a stationary goal with three dynamic puddles
"""
if args.env == 'Navigation2D-v3':
    ### reaching a dynamic goal with three dynamic puddles
    RELAX_ITERS = 2

elif args.env in ['SwimmerVel-v1', 'HopperVel-v1', 'HalfCheetahVel-v1']:
    ### locomotion tasks, let the agent run at a dynamic velocity
    RELAX_ITERS = 0; NU = 0.5
    if args.env == 'HopperVel-v1':
        IW_INV = False
    TASK = args.task[0]

elif args.env in ['ReacherDyna-v1', 'ReacherDyna-v2', 'ReacherDyna-v3']:
    ### v1: reaching a dynamic goal by a two-linked robotic arm
    ### v2: reaching a stationary goal with different physical parameters
    ### v3: reaching a dynamic goal with different physical parameters
    RELAX_ITERS = 0
    if args.env == 'ReacherDyna-v2':
        TASK = int(args.task[0])

### set the task, i.e., given an environment   
print('Taks information', TASK)
sampler.reset_task(TASK)


### in an original environment
if args.stage == 'pretrain':
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    policy = NormalMLPPolicy(state_dim, action_dim, 
            hidden_sizes=(args.hidden_size,) * args.num_layers)
    learner = generate_learner(policy)
    rewards = np.zeros(args.num_iter)
   
    ### training 
    for idx in tqdm(range(args.num_iter)):
        episodes = sampler.sample(policy, device=device)
        rewards[idx] = episodes.evaluate()
        learner.step(episodes)

    ### save the model for initialization in the new environment
    name = os.path.join(args.model_path, 'model_pretrained.pkl')
    print('Save the model to %s'%name); torch.save(policy, name)
    np.save(os.path.join(args.model_path, 'rewards_pretrained.npy'), rewards)

### in a new environment
elif args.stage == 'finetune':
    ### generate a random policy for policy relaxation 
    policy_relax = NormalMLPPolicy(state_dim, action_dim,
            hidden_sizes=(args.hidden_size,) * args.num_layers).to(device)

    ### the policy model name from the original environment
    model_name = os.path.join(args.model_path, 'model_pretrained.pkl')

    ### create an output folder
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    ### the Random baseline 
    if args.random:
        print('\n========== The Random baseline ==========')
        print('Always randomly initialize policy parameters...')
        policy_ran = NormalMLPPolicy(state_dim, action_dim,
                hidden_sizes=(args.hidden_size,) * args.num_layers)
        learner_ran = generate_learner(policy_ran)
        rewards_ran = np.zeros(args.num_iter)

        ### training in the new environment
        for idx in tqdm(range(args.num_iter)):
            ### Random baseline
            episodes_ran = sampler.sample(policy_ran, device=device)
            rewards_ran[idx] = episodes_ran.evaluate()
            learner_ran.step(episodes_ran)

        ### save the data 
        name = os.path.join(args.output, 'random.npy')
        print('Save the Random baseline to file: %s'%name)
        np.save(name, rewards_ran)
    
    ### Pretrained baseline
    if args.pretrained:
        print('\n========== The Pretrained baseline ==========')
        print('Load the policy from %s'%model_name)
        policy_pre = torch.load(model_name).to(args.device)
        learner_pre = generate_learner(policy_pre)
        rewards_pre = np.zeros(args.num_iter)

        for idx in tqdm(range(args.num_iter)):
            episodes_pre = sampler.sample(policy_pre, device=device)
            rewards_pre[idx] = episodes_pre.evaluate()
            learner_pre.step(episodes_pre)

        ### save the data 
        name = os.path.join(args.output, 'pretrained.npy')
        print('Save the Pretrained baseline to file: %s'%name)
        np.save(name, rewards_pre)

    ### a variant of the proposed method:  Importance Weighting
    if args.iw:
        print('\n========== Importance Weighting method ==========') 
        print('Load policy from %s'%model_name)
        policy_iw = torch.load(model_name).to(args.device)
        learner_iw = generate_learner(policy_iw)
        rewards_iw = np.zeros(args.num_iter)

        for idx in tqdm(range(args.num_iter)):
            episodes_iw = sampler.sample(policy_iw, device=device)
            rewards_iw[idx] = episodes_iw.evaluate()
            ### importance weighting step
            learner_iw.step(episodes_iw, iw=True)

        ### save the data 
        name = os.path.join(args.output, 'iw.npy')
        print('Save the IW method to file: %s'%name)
        np.save(name, rewards_iw)

    ### a variant of the proposed method: Policy Relaxation
    if args.pr:
        print('\n========== Policy Relaxation method ==========')
        print('Load policy from %s'%model_name)
        policy_pr = torch.load(model_name).to(args.device)
        learner_pr = generate_learner(policy_pr, pr_smooth=PR_SMOOTH)
        rewards_pr = np.zeros(args.num_iter)

        for idx in tqdm(range(args.num_iter)):
            episodes_pr = sampler.sample(policy_pr, device=device)
            pr = False
            rewards_pr[idx] = episodes_pr.evaluate()
            if idx < RELAX_ITERS:
                ### generate samples from the relaxed policy
                episodes_pr = sampler.sample(policy_relax, device=device) 
                pr = True
            learner_pr.step(episodes_pr, pr=pr)

        name = os.path.join(args.output, 'pr.npy')
        print('Save the PR method to file: %s'%name)
        np.save(name, rewards_pr)

    ### the proposed method: Policy Relaxation + Importance Weighting
    if args.priw:
        print('\n========== Proposed method, PR + IW ==========')
        print('Load policy from %s'%model_name)
        policy_priw = torch.load(model_name).to(args.device)
        learner_priw = generate_learner(policy_priw, pr_smooth=PR_SMOOTH)
        rewards_priw = np.zeros(args.num_iter)

        for idx in tqdm(range(args.num_iter)):
            episodes_priw = sampler.sample(policy_priw, device=device)
            pr = False
            rewards_priw[idx] = episodes_priw.evaluate()
            if idx < RELAX_ITERS:
                ### policy relaxation step
                episodes_priw = sampler.sample(policy_relax, device=device) 
                pr = True
            ### importance weighting step
            learner_priw.step(episodes_priw, iw=True, pr=pr)

        ### save the data 
        name = os.path.join(args.output, 'priw.npy')
        print('Save the proposed method to file: %s'%name)
        np.save(name, rewards_priw)

    ### PRPG baseline
    if args.prpg:
        '''
        Hyperparameters of PRQ-learning, more details can be found in: 
        [2] Fernando Fernandez, Javier Garcia, and Manuela Veloso, 
            "Probabilistic Policy Reuse for inter-task transfer learning", 
            Robotics and Autonomous Systems, 2010.
        '''
 
        print('\n========== PRPG method ==========')
        print('Randomly initialize the policy parameters...')
        policy_prpg = NormalMLPPolicy(state_dim, action_dim,
                hidden_sizes=(args.hidden_size,) * args.num_layers)
        learner_prpg = generate_learner(policy_prpg)
        rewards_prpg = np.zeros(args.num_iter)

        ### reuse the policy from the original environment
        print('Load the old policy from %s'%model_name)
        policy_old = torch.load(model_name).to(args.device)

        ### UPSILON: temperature for weighting the old and new policies
        ### PSI: the probability for using the old policy
        ### NU: weight decay for using the old policy
        score_old, score_new = 0.0, 0.0
        used_old, used_new = 0, 0

        ### record the probability for selecting the old policy
        ### for debuging the PRQ-learning algorithm
        select_old_p, use_old_epis = [], []

        for idx in tqdm(range(args.num_iter)):
            p_old = np.exp(UPSILON * score_old) / (
                    np.exp(UPSILON * score_new) + np.exp(UPSILON * score_old))
            select_old = np.random.binomial(n=1, p=p_old, size=1)
            episodes_prpg = sampler.sample(policy_prpg, device=device)
            pr = False 
            rewards_prpg[idx] = episodes_prpg.evaluate()
            r_tau_new = RMAX + episodes_prpg.evaluate()
            to_use_new = True
            if select_old==1:
                reuse = np.random.binomial(n=1, p=PSI, size=1)
                if reuse==1:
                    use_old_epis.append(idx)
                    episodes_prpg = sampler.sample(policy_old, device=device)
                    pr = True
                    r_tau_old = RMAX + episodes_prpg.evaluate()
                    score_old = (score_old * used_old + r_tau_old) / (used_old+1)
                    used_old += 1
                    to_use_new = False
            if to_use_new:
                score_new = (score_new*used_new+r_tau_new)/(used_new+1)
                used_new += 1
            learner_prpg.step(episodes_prpg, pr=pr)
            PSI *= NU
            select_old_p.append(p_old*PSI)

        ### save the data 
        name = os.path.join(args.output, 'prpg.npy')
        print('Save PRPG method to file: %s'%name)
        np.save(name, rewards_prpg)




print('Running time: %.2f'%(time.time()-start_time))


