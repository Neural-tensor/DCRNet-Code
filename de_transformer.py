# -*- coding: utf-8 -*-
"""
https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/blob/master/pytorch_classification/vision_transformer/vit_model.py
"""
from functools import partial
from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.nn as nn
from einops import rearrange
import math

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv5x5(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 2) -> nn.Conv2d:
    """5x5 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=1)


def conv7x7(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 2) -> nn.Conv2d:
    """5x5 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=1)



def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def deconv2x2(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=stride,
                              groups=groups, bias=False, dilation=dilation)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    

class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class UnLinearProj(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,  
                 out_dim,
                 phase = None,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.phase = phase
        
        self.num_heads = num_heads
        self.hidden_dim = out_dim
        self.head_dim = self.hidden_dim // num_heads
        self.head_dim_v = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5 * 1
        
        self.q_proj = nn.Linear(dim, self.hidden_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, self.hidden_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        
        self.dim = dim
        

    def forward(self, x_q, x_kv, use_attn_prior = False, attn_prior=None):

        B, N, C = x_q.shape
        N_kv = x_kv.shape[1]

        q = self.q_proj(x_q).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x_kv).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x_kv).reshape(B, N_kv, self.num_heads, self.head_dim_v).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        

        
        if use_attn_prior:
            ### let the attention focus on the neighbor patches
            attn = attn + attn_prior

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 out_dim,
                 num_heads=8,
                 phase = None,
                 mlp_ratio=4.,
                 mlp_drop_ratio=0.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 ori_skip_drop_ratio=0.,
                 drop_path_ratio=0.,
                  act_layer=nn.GELU, 
                  # act_layer=nn.ReLU,
                 norm_layer=nn.LayerNorm
                 ):
        super(Block, self).__init__()
        
        self.norm1 = norm_layer(dim)

        self.attn = Attention(dim,
                              out_dim,
                              phase = phase,
                              num_heads=num_heads, 
                              qkv_bias=qkv_bias, 
                              qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, 
                              proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.skip_drop = DropPath(ori_skip_drop_ratio) if ori_skip_drop_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=mlp_drop_ratio)
        # self.act = act_layer()
        
        
    def forward(self, x_q, x_kv, use_mlp=True, use_attn_prior = False, attn_prior=None, skip_connect_off=False):

        x_q_re = self.attn(x_q, x_kv, use_attn_prior, attn_prior)
        if not skip_connect_off:
            x = self.skip_drop(x_q) + x_q_re
        else:
            x = x_q_re
            
        if use_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x





class TransformerDecoder(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=4,
                 mlp_ratio=1.,
                 qkv_bias=False,
                 qk_scale=None,
                 mlp_drop_ratio=0.,
                 attn_drop_ratio=0.,
                 ori_skip_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 use_input_mlp = False,
                 attn_mask = None
                 ):
        super(TransformerDecoder, self).__init__()

        self.use_input_mlp = use_input_mlp
        
        self.attn_mask = attn_mask
        
        
        hidden_dim = int(dim/2)
        
        
        self.block1 = Block(dim=dim,  
                            out_dim=hidden_dim,
                            num_heads = num_heads,
                            mlp_ratio = mlp_ratio,
                            mlp_drop_ratio = mlp_drop_ratio,
                            attn_drop_ratio=attn_drop_ratio,
                            ori_skip_drop_ratio=ori_skip_drop_ratio
                            )
        self.block2 = Block(dim=dim, 
                            out_dim=hidden_dim, 
                            num_heads = num_heads, 
                            mlp_ratio = mlp_ratio,
                            mlp_drop_ratio = mlp_drop_ratio,
                            attn_drop_ratio=attn_drop_ratio,
                            ori_skip_drop_ratio=ori_skip_drop_ratio
                            )

        ## 
        self.num_tokens = 256
        
        self.x1_learned_token = nn.Parameter(torch.zeros(1, self.num_tokens, dim))
        self.x2_learned_token = nn.Parameter(torch.zeros(1, self.num_tokens, dim))
        
        nn.init.uniform_(self.x1_learned_token, -0.05, 0.05)
        nn.init.uniform_(self.x2_learned_token, -0.05, 0.05)
        
    def forward(self, x1, x2, use_attn_flag=False, cross_input=True):

        if cross_input:
            x1_out = self.block1(x1, x2,
                                  use_mlp = True,
                                  use_attn_prior=use_attn_flag,
                                  attn_prior=self.attn_mask,
                                  skip_connect_off=False
                                  )
            x2_out = self.block2(x2, x1, 
                                  use_mlp = True,
                                  use_attn_prior=use_attn_flag, 
                                  attn_prior=self.attn_mask, 
                                  skip_connect_off=False
                                  )
        else:
            x1_out = self.block1(x1, x1,
                                  use_mlp = True,
                                  use_attn_prior=use_attn_flag,
                                  attn_prior=self.attn_mask,
                                  skip_connect_off=False
                                  )
            x2_out = self.block2(x2, x2, 
                                  use_mlp = True,
                                  use_attn_prior=use_attn_flag, 
                                  attn_prior=self.attn_mask, 
                                  skip_connect_off=False
                                  )
            
        return x1_out, x2_out
        






if __name__ == "__main__":
    BATCH_SIZE = 4
    N_QUERY = 196
    N_SUPPORT = 64
    INPUT_C = 256
    
    NUM_HEADS = 8
    MLP_RATIO = 4
    
    query_feat = torch.rand(BATCH_SIZE, N_QUERY, INPUT_C)
    support_feat = torch.rand(BATCH_SIZE, N_SUPPORT, INPUT_C)
    
    corre_module = TransformerDecoder(dim=INPUT_C, num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO)
    
    query_feat_out, support_feat_out = corre_module(query_feat,support_feat)
    
    print('query_feat_out:', query_feat_out.shape)
    print('support_feat_out:', support_feat_out.shape)
    
    
    

