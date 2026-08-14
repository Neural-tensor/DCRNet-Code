#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import math
import torch


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)    






def get_distance_arr(patch_num_axis):
    window_size = patch_num_axis * patch_num_axis
    distances_x = np.zeros((window_size, window_size))
    distances_y = np.zeros((window_size, window_size))
    
    for i in range(patch_num_axis):
        for j in range(patch_num_axis):
            for i_ in range(patch_num_axis):
                for j_ in range(patch_num_axis):
                    distances_x[i+j*patch_num_axis][i_+j_*patch_num_axis] = i_ - i
            
    for i in range(patch_num_axis):
        for j in range(patch_num_axis):
            for i_ in range(patch_num_axis):
                for j_ in range(patch_num_axis):
                    distances_y[i+j*patch_num_axis][i_+j_*patch_num_axis] = j_ - j
    return distances_x, distances_y



def cal_gauss_prior(patch_num_axis=14, sigma_x=2, sigma_y=2, thres=0.1):
    distances_x, distances_y = get_distance_arr(patch_num_axis)
    
    # plt.figure()
    # plt.imshow(distances_y,plt.cm.gray)
    # plt.colorbar(shrink=1)
    # plt.show()
    
    prior = 1.0 / (2 * math.pi * sigma_x * sigma_y) * np.exp(- distances_x ** 2 / 2 / (sigma_x ** 2)
                                                             - distances_y ** 2 / 2 / (sigma_y ** 2))
    prior = min_max_norm(prior)   ### norm to 0-1
    prior = prior.astype(np.float32)
    
    ### --- Gauss Prior Visualization
    plt.figure()
    plt.imshow(prior,plt.cm.gray)
    plt.colorbar(shrink=1)
    plt.show()
    
    prior[prior<thres] = 0
    prior[prior>=thres] = 1
    
    avg_masked_patch = prior.sum() / (patch_num_axis*patch_num_axis)
    print('average masked patch:', round(avg_masked_patch,1))
    print('average radius:', round(avg_masked_patch**0.5,1))
    
    ### make the attention on the far patches
    # prior = prior * -10000
    
    ### make the attention on the neighbor patches
    prior = (1-prior) * -10000
    
    return prior


def get_attn_mask(device, mask_threshold=0.2, patch_num_axis=14, NUM_HEADS=8):
    neighbor_mask = cal_gauss_prior(patch_num_axis=patch_num_axis, thres=mask_threshold)
    print('neighbor_mask:', neighbor_mask.shape)
    neighbor_mask = torch.Tensor(neighbor_mask).to(device)
    neighbor_mask = neighbor_mask.unsqueeze(0).unsqueeze(0).repeat(1,NUM_HEADS, 1,1)
    return neighbor_mask




if __name__=='__main__':


    prior = cal_gauss_prior(sigma_x=2, sigma_y=2, thres=0.03)
    print('prior shape:', prior.shape)
    

    plt.figure()
    plt.imshow(prior,plt.cm.gray)
    plt.axis('off')
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    