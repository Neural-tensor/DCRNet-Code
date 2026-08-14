import torch
from torch import Tensor
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from typing import Callable, Optional
import random
import numpy as np
from torch.nn import functional as F
from einops import rearrange
from de_transformer import TransformerDecoder, DropPath
from gen_attn_mask import get_attn_mask


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv5x5(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 2) -> nn.Conv2d:
    """5x5 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=1)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def deconv2x2(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=stride,
                              groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        upsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if stride == 2:
            self.conv1 = deconv2x2(inplanes, planes, stride)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class SELayer(nn.Module):
    def __init__(self, num_channels, reduction_ratio=4):
        '''
            num_channels: The number of input channels
            reduction_ratio: The reduction ratio 'r' from the paper
        '''
        super(SELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        batch_size, num_channels, H, W = input_tensor.size()

        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


class DepthWiseConv(nn.Module):
    def __init__(self,in_channel, stride=1):
 
        #这一行千万不要忘记
        super(DepthWiseConv, self).__init__()
 
        # 逐通道卷积
        self.depth_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=in_channel,
                                    kernel_size=3,
                                    stride=stride,
                                    padding=1,
                                    groups=in_channel)
        # groups是一个数，当groups=in_channel时,表示做逐通道卷积
 
        # #逐点卷积
        # self.point_conv = nn.Conv2d(in_channels=in_channel,
        #                             out_channels=out_channel,
        #                             kernel_size=1,
        #                             stride=1,
        #                             padding=0,
        #                             groups=1)
    
    def forward(self,x):
         out = self.depth_conv(x)
         # out = self.point_conv(out)
         return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        upsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.upsample layers upsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out





class Block(nn.Module):
    def __init__(self,hidden_planes, mode, conv_mask = False, use_se = False):
        super(Block, self).__init__()
        self.use_se = use_se
        if mode == 'down':
            self.maskconv = conv3x3(hidden_planes, int(hidden_planes*0.5), stride=1, groups=1, dilation=1)
            self.bn = nn.BatchNorm2d(int(hidden_planes*0.5),momentum=0.1)
            if use_se:
                self.se = SELayer(int(hidden_planes*0.5))
        elif mode == 'up':
            self.maskconv = conv3x3(hidden_planes, hidden_planes*2, stride=1, groups=1, dilation=1)
            self.bn = nn.BatchNorm2d(hidden_planes*2,momentum=0.1)
            if use_se:
                self.se = SELayer(hidden_planes*2)
        else:
            self.maskconv = conv3x3(hidden_planes, hidden_planes, stride=1, groups=1, dilation=1)
            self.bn = nn.BatchNorm2d(hidden_planes,momentum=0.1)
            if use_se:
                self.se = SELayer(hidden_planes)
        
        if conv_mask:
            self.maskconv.weight.requires_grad = False
            self.maskconv.weight.data[:,:,1,1] = 0.0
            self.maskconv.weight.requires_grad = True

        self.relu = nn.ReLU(inplace=True)
        self.lrelu = nn.LeakyReLU()
        
    def forward(self, x):
        # identity = x
        out = self.maskconv(x)
        out = self.bn(out)
        out = self.relu(out)
        # out = self.lrelu(out)
        
        if self.use_se:
            out = self.se(out)
        return out

class Block2(nn.Module):
    def __init__(self,c1, c2):
        super(Block2, self).__init__()

        self.maskconv = conv3x3(c1, c2, stride=1, groups=1, dilation=1)
        self.bn = nn.BatchNorm2d(c2,momentum=0.1)

        self.relu = nn.ReLU(inplace=True)
        # self.lrelu = nn.LeakyReLU()
        
    def forward(self, x):
        # identity = x
        out = self.maskconv(x)
        out = self.bn(out)
        out = self.lrelu(out)
        # out = self.relu(out)
        return out
    
    
class SPPF(nn.Module):
    """
        This code referenced to https://github.com/ultralytics/yolov5
    """
    def __init__(self, in_dim, out_dim, expand_ratio=0.5, pooling_size=3, act_type='lrelu', norm_type='BN'):
        super().__init__()
        inter_dim = int(in_dim * expand_ratio)
        self.out_dim = out_dim
        self.cv1 = Block2(in_dim, inter_dim)
        self.cv2 = Block2(inter_dim * 4, out_dim)
        self.m = nn.MaxPool2d(kernel_size=pooling_size, stride=1, padding=pooling_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)

        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class ConvBatchRelu(nn.Module):
    def __init__(self, c_in, c_out, norm='batch', mode='3x3', stride=1, dw_conv=False):
        super(ConvBatchRelu, self).__init__()
        
        if mode == '1x1':
            self.conv = conv1x1(c_in, c_out, stride=1)
        else:
            self.conv = conv3x3(c_in, c_out, stride=stride)
            
        if norm == 'batch':
            self.bn = nn.BatchNorm2d(c_out, momentum=0.1)
        elif norm == 'instance':
            self.bn = nn.InstanceNorm2d(c_out, momentum=0.1)
        else:
            self.bn = nn.Identity()
            
        self.dw_conv = dw_conv
        if self.dw_conv:
            self.dwconv = DepthWiseConv(c_in)
        # self.relu = nn.GELU()
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        if self.dw_conv:
            x = self.dwconv(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x



class ResizeConvBatchRelu(nn.Module):
    def __init__(self,s_out, c_in, c_out, norm='batch', mode='3x3'):
        super(ResizeConvBatchRelu, self).__init__()
        self.s = s_out
        
        if mode == '1x1':
            self.conv = conv1x1(c_in, c_out, stride=1)
        else:
            self.conv = conv3x3(c_in, c_out, stride=1)
            
        if norm == 'batch':
            self.bn = nn.BatchNorm2d(c_out, momentum=0.1)
        elif norm == 'instance':
            self.bn = nn.InstanceNorm2d(c_out, momentum=0.1)
        else:
            self.bn = nn.Identity()
            
        # self.relu = nn.GELU()
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = F.interpolate(x, size=[self.s, self.s], mode='bilinear')
        
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x

# ### 8x   layerx4
# class XYZFeatEncoder(nn.Module):
#     def __init__(self,input_c, latent_c, norm='batch'):
#         super(XYZFeatEncoder, self).__init__()
        
#         self.conv1 = ConvBatchRelu(input_c,  int(input_c/2), norm, mode='1x1')
#         self.conv2 = ConvBatchRelu(int(input_c/2), int(input_c/4), norm, mode='3x3')
#         self.conv3 = ConvBatchRelu(int(input_c/4), int(input_c/8), norm, mode='3x3', stride=1)
#         self.conv4 = ConvBatchRelu(int(input_c/8), latent_c,       norm, mode='3x3', stride=2)
        
#     def forward(self, out):
        
#         out = self.conv1(out)
#         out = self.conv2(out)
#         out = self.conv3(out)
#         out = self.conv4(out)
#         return out




# ### 4x   layerx4
# class XYZFeatEncoder(nn.Module):
#     def __init__(self,input_c, latent_c, norm='batch'):
#         super(XYZFeatEncoder, self).__init__()
        
#         self.conv1 = ConvBatchRelu(input_c,  int(input_c/2), norm, mode='1x1')
#         self.conv2 = ConvBatchRelu(int(input_c/2), int(input_c/4), norm, mode='3x3')
#         self.conv3 = ConvBatchRelu(int(input_c/4), int(input_c/4), norm, mode='3x3', stride=1)
#         self.conv4 = ConvBatchRelu(int(input_c/4), latent_c,       norm, mode='3x3', stride=2)
        
#     def forward(self, out):
        
#         out = self.conv1(out)
#         out = self.conv2(out)
#         out = self.conv3(out)
#         out = self.conv4(out)
#         return out


### 4x  layerx3
class XYZFeatEncoder(nn.Module):
    def __init__(self,input_c, latent_c, norm='batch'):
        super(XYZFeatEncoder, self).__init__()
        
        self.conv1 = ConvBatchRelu(input_c,  int(input_c/2), norm, mode='1x1')
        self.conv2 = ConvBatchRelu(int(input_c/2), int(input_c/4), norm, mode='3x3')
        self.conv3 = ConvBatchRelu(int(input_c/4), latent_c, norm, mode='3x3', stride=2)
        # self.conv4 = ConvBatchRelu(int(input_c/8), latent_c,       norm, mode='3x3', stride=2)
        
    def forward(self, out):
        
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        # out = self.conv4(out)
        return out


class XYZFeatDecoder(nn.Module):
    def __init__(self,out_c, latent_c, norm='batch'):
        super(XYZFeatDecoder, self).__init__()
        self.conv0 = ResizeConvBatchRelu(56, latent_c,  int(out_c/4), norm, mode='3x3')
        # self.conv1 = ConvBatchRelu(int(out_c/8),  int(out_c/4), norm, mode='3x3')
        self.conv2 = ConvBatchRelu(int(out_c/4), int(out_c/2), norm, mode='1x1')
        self.conv3 = ConvBatchRelu(int(out_c/2), int(out_c), norm, mode='1x1')
        
        self.conv4 = conv1x1(out_c, out_c)
        
    def forward(self, out):
        out = self.conv0(out)
        # out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
 
        return out


### 4x
class RGBFeatEncoder(nn.Module):
    def __init__(self,input_c=1024 + 512 + 256, latent_c=int(1152/4), norm='batch'):
        super(RGBFeatEncoder, self).__init__()
        
        self.conv11 = ConvBatchRelu(input_c,        int(input_c/2), norm, mode='1x1',dw_conv=False)
        self.conv12 = ConvBatchRelu(int(input_c/2), int(input_c/4), norm, mode='1x1',dw_conv=False)
        self.conv13 = ConvBatchRelu(int(input_c/4), latent_c, norm, mode='3x3')
        self.conv14 = ConvBatchRelu(latent_c, latent_c, norm, mode='3x3', stride=1)
        
    def forward(self, x1):
        
        x1 = self.conv11(x1)
        x1 = self.conv12(x1)
        x1 = self.conv13(x1)
        x1 = self.conv14(x1)
 
        return x1
    
### 8x
class RGBFeatDecoder(nn.Module):
    def __init__(self,input_c=1024 + 512 + 256, latent_c=int(1152/4), norm='batch'):
        super(RGBFeatDecoder, self).__init__()
        
        # self.conv11 = ResizeConvBatchRelu(56, latent_c,  int(input_c/8), norm, mode='3x3')
        self.conv11 = ConvBatchRelu(latent_c,  latent_c, norm, mode='3x3')
        self.conv12 = ConvBatchRelu(latent_c, int(input_c/4), norm, mode='3x3')
        self.conv13 = ConvBatchRelu(int(input_c/4), int(input_c/2), norm, mode='1x1',dw_conv=False)
        self.conv14 = ConvBatchRelu(int(input_c/2), input_c, norm, mode='1x1',dw_conv=False)
        self.conv1_11 = conv1x1(input_c, input_c)
        
    def forward(self, x1):
        
        x1 = self.conv11(x1)
        x1 = self.conv12(x1)
        x1 = self.conv13(x1)
        x1 = self.conv14(x1)
        x1 = self.conv1_11(x1)
        
        return x1



class RGBConcat(nn.Module):
    def __init__(self,hw_size=28):
        super(RGBConcat, self).__init__()
        self.s = hw_size
        
    def forward(self, x_lst):
        x1 = F.interpolate(x_lst[0], size=[self.s, self.s], mode='bilinear')
        x2 = F.interpolate(x_lst[1], size=[self.s, self.s], mode='bilinear')
        x3 = F.interpolate(x_lst[2], size=[self.s, self.s], mode='bilinear')
        return torch.cat([x1,x2,x3], dim=1)


class DecoderConcat(nn.Module):
    def __init__(self, device, channel_dim_lst):
        super(DecoderConcat, self).__init__()
        self.device = device
        
        # rgb_dim = 64+128+256  ## for res18 34
        rgb_dim = 1024+512+256  ## for res50 101  wres50 101
        
        xyz_dim = 1152
        latent_dim = 256
        num_heads = 2
        mlp_ratio = 2
        latent_hw = 28
        
        
        self.latent_hw = latent_hw
        self.use_attn_flag = True
        
        attn_drop_ratio = 0.8  
        mlp_drop_ratio = 0.
        ori_skip_drop_ratio=0.0
        norm = 'batch'
        
        
        ###  2D-gaussian based mask, for neighbor-aware masked cross-attention
        attn_mask = get_attn_mask(device, 
                                  mask_threshold=0.03, 
                                  patch_num_axis=latent_hw, 
                                  NUM_HEADS=num_heads
                                  )
     
        self.xyz_encoder = XYZFeatEncoder(int(xyz_dim/2), latent_dim, norm)
        self.xyz_decoder = XYZFeatDecoder(xyz_dim, latent_dim, norm)
        
        self.rgb_encoder = RGBFeatEncoder(int(rgb_dim/2), latent_dim)
        self.rgb_decoder = RGBFeatDecoder(rgb_dim, latent_dim)
        

        self.cross_fuse1 = TransformerDecoder(dim=latent_dim,
                                             num_heads=num_heads,
                                             mlp_ratio=mlp_ratio,
                                             mlp_drop_ratio=mlp_drop_ratio,
                                             attn_mask=attn_mask,
                                             attn_drop_ratio=attn_drop_ratio,
                                             ori_skip_drop_ratio=ori_skip_drop_ratio
                                             )  
        
        self.cross_fuse2 = TransformerDecoder(dim=latent_dim,
                                              num_heads=num_heads,
                                              mlp_ratio=mlp_ratio,
                                              attn_mask=attn_mask,
                                              attn_drop_ratio=attn_drop_ratio,
                                              ori_skip_drop_ratio=ori_skip_drop_ratio
                                              )  

        
        
    def forward(self,rgb, xyz, fore_mask56=None, fore_mask28=None):

        rgb_latent = self.rgb_encoder(rgb)
        xyz_latent = self.xyz_encoder(xyz)
        
        
        ### cross-modal information guided fusion
        ##### ---------------------------------------------------------------
        rgb_latent = rearrange(rgb_latent, 'b c h w -> b (h w) c')
        xyz_latent = rearrange(xyz_latent, 'b c h w -> b (h w) c')
        
        rgb_latent1, xyz_latent1 = self.cross_fuse1(xyz_latent, rgb_latent, use_attn_flag=self.use_attn_flag)
        
        rgb_latent2, xyz_latent2 = self.cross_fuse2(rgb_latent1, xyz_latent1, use_attn_flag=self.use_attn_flag)
        
        rgb_latent_fuse = rgb_latent1 + rgb_latent2 
        xyz_latent_fuse = xyz_latent1 + xyz_latent2 

        rgb_latent_out = rearrange(rgb_latent_fuse, 'b (h w) c -> b c h w', h=self.latent_hw, w=self.latent_hw)
        xyz_latent_out = rearrange(xyz_latent_fuse, 'b (h w) c -> b c h w', h=self.latent_hw, w=self.latent_hw)
        
        
        # ## -----------------------  direct intra-modal recons ----------------
        # rgb_latent_out = rgb_latent
        # xyz_latent_out = xyz_latent
        
        # ## -----------------------  direct cross-modal recons -----------------
        # rgb_latent_out = xyz_latent
        # xyz_latent_out = rgb_latent
        
        rgb_out = self.rgb_decoder(rgb_latent_out)
        xyz_out = self.xyz_decoder(xyz_latent_out)
        
        return rgb_out, xyz_out, rgb_latent_out, xyz_latent_out

    
    
    
    
