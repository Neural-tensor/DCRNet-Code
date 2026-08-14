
import torch
import numpy as np
import random
import os
import time
from numpy import ndarray
from PIL import Image
import glob
import cv2
from torch.nn import functional as F
from sklearn.metrics import auc
from statistics import mean
from skimage import measure
import pandas as pd


avgpool = torch.nn.AvgPool2d(3, 1, 1) 
erode_kernel = np.ones((3, 3), np.uint8)




def output_file_init(_class_):
    if not os.path.exists('./log'): os.mkdir('./log')
    if not os.path.exists('./checkpoints'): os.mkdir('./checkpoints')
    if not os.path.exists('./log/output'): os.mkdir('./log/output')
    if not os.path.exists('./log/score'): os.mkdir('./log/score')
    img_save_base_path = os.path.join('./log/output', _class_)
    if not os.path.exists(img_save_base_path):
        os.mkdir(img_save_base_path)
    return img_save_base_path

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def xyz_channel_masking(feat, dim=384, half_dim=int(384/2)):
    f1 = feat[:, 0:half_dim, :, :]
    f2 = feat[:, dim:dim+half_dim, :, :]
    f3 = feat[:, dim*2:dim*2+half_dim, :, :]
    out = torch.cat([f1, f2, f3], dim=1)
    return out

def rgb_channel_masking(inlst, channel_dim_lst=[256,512,1024]):
    out_lst = []
    for idx, feat in enumerate(inlst):
        half_dim = int(channel_dim_lst[idx] * 0.5)
        out_lst.append(feat[:, :half_dim, :, :])
    return out_lst




def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_pc_rgb_fore_mask(pc_feat, device, cls_name, size_lst=[56,28,14]):
    fore_map = pc_feat.abs().mean(dim=1)

    if cls_name == 'foam': 
        fore_map = (fore_map > 0.01).float()
        
    else:
        fore_map = (fore_map > 0.0).float()

    
    s1,s2,s3 = size_lst[0],size_lst[1],size_lst[2]

    ### pc foreground mask
    pc_fore_mask = fore_map.cpu().numpy()
    pc_fore_mask = cv2.erode(pc_fore_mask, erode_kernel, iterations=1) 
    fore_mask56 = torch.Tensor(pc_fore_mask).to(device)
    
    
    ### rgb foreground map
    rgb_fore_mask1 = avgpool(fore_mask56.unsqueeze(dim=1)) 
    rgb_fore_mask2 = F.interpolate(rgb_fore_mask1, size=[s2, s2], mode='bilinear') 
    fore_mask28 = (rgb_fore_mask2 > 0.2).float().squeeze()
    
    return fore_mask56, fore_mask28


mse_loss = torch.nn.MSELoss()
cos_loss = torch.nn.CosineSimilarity()


def loss_fucntion(a, b, fore_map, use_fore_mask=False):
    a_map = 1-cos_loss(a, b)
    if use_fore_mask:
        a_map = a_map * fore_map
        loss = torch.sum(a_map) / torch.sum(fore_map)
    else:
        loss = torch.mean(a_map)
    return loss, a_map



def l2_normalize(x, eps=1e-12):
    norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    x = x / (norm + eps)
    return x
    
def cal_anomaly_map(fs, ft, fore_map=None, fore_map_flag=True, out_size=224):
    # ### norm L2 distance
    c,h,w = fs.shape[1],fs.shape[2], fs.shape[3]
    fs = fs.squeeze().permute(1,2,0)
    ft = ft.squeeze().permute(1,2,0)
    fs = l2_normalize(fs)
    ft = l2_normalize(ft)
    tensor1_flat = fs.view(-1, c)
    tensor2_flat = ft.view(-1, c)
    l2_distances = torch.norm(tensor1_flat - tensor2_flat, dim=1)
    a_map = l2_distances.view(h, w).unsqueeze(dim=0)

    if fore_map_flag:
        a_map = a_map * fore_map
    
    anomaly_map = torch.unsqueeze(a_map, dim=1)
            
    anomaly_map = F.interpolate(anomaly_map, size=out_size, mode='bilinear', align_corners=True)
    
    anomaly_map = anomaly_map[0, 0, :, :].to('cpu').detach().numpy()
    
    return anomaly_map, a_map

def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)


def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap




class TrainMVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, memory_img_lst, ROTATE_FLAG = False):
        self.rotate = ROTATE_FLAG
        self.root_path = root
        self.memory_img_lst = memory_img_lst 
        self.transform = transform
        self.imgdic = {}
        # load dataset
        self.img_paths = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        
    def load_dataset(self):
        print('load images...')
        ss = time.time()
        if 'MVTec3D' in self.root_path:
            img_paths = glob.glob(os.path.join(self.root_path, 'good', 'rgb') + "/*.png")
        else:
            img_paths = glob.glob(os.path.join(self.root_path, 'good') + "/*.png")
        print(len(img_paths))
        img_paths = [p for p in img_paths if p not in self.memory_img_lst]
        print(len(img_paths))
        counter = 0
        for i,img_p in enumerate(img_paths):
            img = Image.open(img_p).convert('RGB')
            if self.rotate:
                for r in [0, 90, 180, 270]:
                    imgr = img.rotate(r)
                    imgtrans = self.transform(imgr)            
                    self.imgdic[counter] = imgtrans
                    counter += 1
            
            img = self.transform(img)            
            self.imgdic[i] = img
        print('read {} images time used:'.format(len(img_paths)), time.time() - ss)
        return img_paths

    def __len__(self):
        if self.rotate:
            return len(self.img_paths) * 4
        else:
            return len(self.img_paths)

    def __getitem__(self, idx):
        
        return self.imgdic[idx]
    




def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """
    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        # df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)
        df = pd.concat([df, pd.DataFrame({"pro": [mean(pros)], "fpr": [fpr], "threshold": [th]})], ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc





