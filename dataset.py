from torchvision import transforms
from PIL import Image
import os
import torch
import glob
import numpy as np
import time

def get_data_transforms(size, isize):
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    data_transforms = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        #transforms.CenterCrop(args.input_size),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms

    
    

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, img_root, feat_root, transform):
        self.feat_root_path = feat_root
        self.img_root_path = img_root
        
        self.transform = transform
        
        self.imgdic = {}
        self.feat_dic = {}
        
        self.img_paths = self.load_dataset()
        

    def load_dataset(self):
        print('load images...')
        ss = time.time()

        img_paths = glob.glob(os.path.join(self.img_root_path, 'good', 'rgb') + "/*.png")
        
        for i,img_p in enumerate(img_paths):
            
            img = Image.open(img_p).convert('RGB')
            
            img = self.transform(img)            
            self.imgdic[i] = img
            
            img_idx = img_p.split('/')[-1].split('.')[0]
            pt_path = os.path.join(self.feat_root_path, img_idx+'.pt')
            feat = torch.load(pt_path)  
            C = feat.shape[-1]
            feat = feat.permute(1,0).reshape(C, 56, 56)
            self.feat_dic[i] = feat
        print('read {} images time used:'.format(len(img_paths)), time.time() - ss)
        return img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        
        return self.imgdic[idx], self.feat_dic[idx]


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, img_root, feat_root, transform, gt_transform):

        self.img_path = os.path.join(img_root, 'test')
        self.gt_path = os.path.join(img_root, 'test')
        
        self.img_root_path = img_root
        self.feat_root_path = feat_root
        
        self.transform = transform
        self.gt_transform = gt_transform
        # labels => good : 0, anomaly : 1
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_imgs()  
        
        print('load test images...')
        ss = time.time()
        
        ### load features
        self.feat_dic = {}
        self.load_features()
        print('feat num:',len(self.feat_paths))
        
        self.img_dic = {}
        for i, p in enumerate(self.img_paths):
            img = Image.open(p).convert('RGB')
            img = self.transform(img)
            self.img_dic[i] = img
        
        self.gt_dic = {}
        for i, gtp in enumerate(self.gt_paths):
            if gtp == 0:
                gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
            else:
                gt = Image.open(gtp).convert("L")
                gt = self.gt_transform(gt)
            self.gt_dic[i] = gt
        
        print('img num:',len(self.img_paths))
        

        print('load test images time used:', time.time() - ss)
    
    
    def load_features(self):
        print('load features...')
        self.feat_paths = []
        for i in range(len(self.img_paths)):
            img_p = self.img_paths[i]
            defect_type = self.types[i]
            img_idx = img_p.split('/')[-1].split('.')[0]
            feat_path = os.path.join(self.feat_root_path, defect_type, img_idx+'.pt')
            self.feat_paths.append(feat_path)
            
        for i,pt_path in enumerate(self.feat_paths):
            feat = torch.load(pt_path)
            C = feat.shape[-1]
            feat = feat.permute(1,0).reshape(C, 56, 56) ## torch.Size([1152, 56, 56])
            
            self.feat_dic[i] = feat

    
    def load_imgs(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type, 'gt') + "/*.png")

                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        label, img_type = self.labels[idx], self.types[idx]

        img = self.img_dic[idx]
        gt = self.gt_dic[idx]
        
        feat = self.feat_dic[idx]

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, feat, gt, label, img_type

