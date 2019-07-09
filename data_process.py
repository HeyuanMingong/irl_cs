#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 14:27:58 2018

@author: qiutian
"""

import numpy as np
import os
from myrllib.utils.myplot import simple_plot
import scipy.io as sio
import matplotlib.pyplot as plt

def arr_ave(arr, bs=1):
    arr = arr.squeeze()
    nl = arr.shape[0]//bs
    arr_n = np.zeros(nl)
    for i in range(nl):
        arr_n[i] = np.mean(arr[bs*i:bs*(i+1)])
    return arr_n


###############################################################################
### observing the rewards of tested methods
### including: Random, Pretrained, PRPG, PR, IW, PR+IW
    
DOMAIN = 'swimmer'
p_model = 'saves/%s'%DOMAIN; p_output = 'output/%s'%DOMAIN

rewards = np.load(os.path.join(p_model, 'rewards_pretrained.npy'))
simple_plot([rewards])

if DOMAIN in ['navi_v1', 'navi_v2', 'navi_v3']:
    navigation_domains()
elif DOMAIN in ['reacher_v1', 'reacher_v2', 'reacher_v3',
                'swimmer']:
    mujoco_domains()  


###############################################################################
def navigation_domains():
    rewards_ran = np.load(os.path.join(p_output, 'random.npy'))
    rewards_pre = np.load(os.path.join(p_output, 'pretrained.npy'))
    rewards_prpg = np.load(os.path.join(p_output, 'prpg.npy'))
    rewards_pr = np.load(os.path.join(p_output, 'pr.npy'))
    rewards_iw = np.load(os.path.join(p_output, 'iw.npy'))
    rewards_priw = np.load(os.path.join(p_output, 'priw.npy'))
    
    
    cutoff = rewards_ran.reshape(-1).shape[0]
    num = 20; bs = cutoff//num
    rewards_ran = arr_ave(rewards_ran.reshape(-1), bs=bs) 
    rewards_pre = arr_ave(rewards_pre.reshape(-1), bs=bs) 
    rewards_prpg = arr_ave(rewards_prpg.reshape(-1), bs=bs) 
    rewards_pr = arr_ave(rewards_pr.reshape(-1), bs=bs) 
    rewards_iw = arr_ave(rewards_iw.reshape(-1), bs=bs) 
    rewards_priw = arr_ave(rewards_priw.reshape(-1), bs=bs) 
        
    xx = np.arange(0, num); mark = num // 10
    plt.figure(figsize=(6,4))
    plt.plot(xx, rewards_ran[xx], color='black', lw=2, ls='--')
    plt.plot(xx, rewards_pre[xx], color='purple', lw=2, ls='--')
    plt.plot(xx, rewards_prpg[xx], color='c', lw=2, ls='--')
    plt.plot(xx, rewards_pr[xx], color='green', lw=2, 
             marker='o', markevery=mark, ms=8, mew=2, mfc='white')
    plt.plot(xx, rewards_iw[xx], color='blue', lw=2, 
             marker='^', markevery=mark, ms=8, mew=2, mfc='white')
    plt.plot(xx, rewards_priw[xx], color='red', lw=2, 
             marker='x', markevery=mark, ms=8, mew=2, mfc='white')
    
    plt.legend(['Random', 'Pretrained', 'PRPG', 'Policy Relaxation', 
                'Importance Weighting', 'PR+IW'],
               labelspacing=0.1,
               fancybox=True, shadow=True, fontsize=10)
    plt.xlabel('Policy iterations', fontsize=18)
    plt.ylabel('Average return', fontsize=18)
    plt.xticks(np.arange(0,num+1,num//5), bs*np.arange(0,num+1,num//5),
               fontsize=10)
    plt.grid(axis='y', ls='--')


###############################################################################
def mujoco_domains():
    rewards_ran = np.load(os.path.join(p_output, 'random.npy'))
    rewards_pre = np.load(os.path.join(p_output, 'pretrained.npy'))
    rewards_prpg = np.load(os.path.join(p_output, 'prpg.npy'))
    rewards_priw = np.load(os.path.join(p_output, 'priw.npy'))
    
    
    cutoff = rewards_ran.reshape(-1).shape[0]
    num = 20; bs = cutoff//num
    rewards_ran = arr_ave(rewards_ran.reshape(-1), bs=bs) 
    rewards_pre = arr_ave(rewards_pre.reshape(-1), bs=bs) 
    rewards_prpg = arr_ave(rewards_prpg.reshape(-1), bs=bs) 
    rewards_priw = arr_ave(rewards_priw.reshape(-1), bs=bs) 
        
    xx = np.arange(0, num); mark = num // 10
    plt.figure(figsize=(6,4))
    plt.plot(xx, rewards_ran[xx], color='c', lw=2, 
             marker='o', markevery=mark, ms=8, mew=2, mfc='white')
    plt.plot(xx, rewards_pre[xx], color='green', lw=2, 
             marker='s', markevery=mark, ms=8, mew=2, mfc='white')
    plt.plot(xx, rewards_prpg[xx], color='blue', lw=2, 
             marker='^', markevery=mark, ms=8, mew=2, mfc='white')
    plt.plot(xx, rewards_priw[xx], color='red', lw=2, 
             marker='x', markevery=mark, ms=8, mew=2, mfc='white')
    
    plt.legend(['Random', 'Pretrained', 'PRPG', 'PR+IW'],
               labelspacing=0.1,
               fancybox=True, shadow=True, fontsize=10)
    plt.xlabel('Policy iterations', fontsize=18)
    plt.ylabel('Average return', fontsize=18)
    plt.xticks(np.arange(0,num+1,num//5), bs*np.arange(0,num+1,num//5),
               fontsize=10)
    plt.grid(axis='y', ls='--') 
  


































