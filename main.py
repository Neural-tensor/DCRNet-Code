
import torch
from dataset import get_data_transforms
import numpy as np
import random
import cv2
import os
import time
import json
from torch.optim import lr_scheduler

from sklearn.metrics import roc_auc_score
from resnet import resnet18, resnet50, wide_resnet50_2

from DCRNet import DecoderConcat, RGBConcat         ### latent  288,28,28

from dataset import TrainDataset, TestDataset

from scipy.ndimage import gaussian_filter

from utils import loss_fucntion, setup_seed, cal_anomaly_map, show_cam_on_image, cvt2heatmap, min_max_norm, output_file_init
from utils import get_pc_rgb_fore_mask, compute_pro
from utils import rgb_channel_masking, xyz_channel_masking
import warnings
warnings.filterwarnings("ignore")






def train(_class_,channel_dim_lst, epochs=10, EVAL_FLAG = True):

    
    print(_class_)
    
    img_save_base_path = output_file_init(_class_)
    
    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    
    ckp_path = './checkpoints/' + 'wres50_'+_class_
    
    
    # encoder, bn = resnet18(pretrained=True)
    # encoder, bn = resnet50(pretrained=True)
    # encoder, bn = resnet101(pretrained=True)
    encoder, bn = wide_resnet50_2(pretrained=True)
    # encoder, bn = wide_resnet101_2(pretrained=True)
    
    encoder = encoder.to(device)
    encoder.eval()
    
    train_img_path = os.path.join(DATASET_PATH,  _class_, 'train')
    train_feat_path = os.path.join(PonitCloudFeat_PATH, 'train', _class_, 'good')
    
    test_img_path = os.path.join(DATASET_PATH,  _class_)
    test_feat_path = os.path.join(PonitCloudFeat_PATH, 'test', _class_)
    
    train_data = TrainDataset(train_img_path, train_feat_path, transform=data_transform)
    test_data = TestDataset(test_img_path, test_feat_path, transform=data_transform, gt_transform=gt_transform)
    
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    
    
    decoder = DecoderConcat(device, channel_dim_lst)
    concater = RGBConcat()

    decoder = decoder.to(device)
    
    optimizer = torch.optim.Adam(list(decoder.parameters()), lr=learning_rate, betas=(0.9, 0.999) , weight_decay=1e-6)
    
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, lr_milestones, gamma=weight_decay_gamma)



    ### ---------------------------- train -------------------------------
    for epoch in range(epochs):
        decoder.train()
        loss_list = []
        rgb_loss_list = []
        xyz_loss_list = []
        
        for img, pc_feat in train_dataloader:
            img = img.to(device)
            pc_feat = pc_feat.to(device)  ## torch.Size([16, 1152, 56, 56])
            
            with torch.no_grad():
                inputs = encoder(img) ## torch.Size([16, 256, 56, 56]), torch.Size([16, 512, 28, 28]), torch.Size([16, 1024, 14, 14])
                
            rgb_input = concater(inputs)
            
            
            channel_masked_inputs = rgb_channel_masking(inputs, channel_dim_lst)
            
            
            channel_masked_rgb_feat = concater(channel_masked_inputs)
            
            channel_masked_pc_feat = xyz_channel_masking(pc_feat)
            
            fore_mask56, fore_mask28 = get_pc_rgb_fore_mask(pc_feat, device, _class_) 
            
            rgb_output, pc_output, rgb_latent, xyz_latent = decoder(channel_masked_rgb_feat, channel_masked_pc_feat, fore_mask56, fore_mask28) 

            xyz_loss, _ = loss_fucntion(pc_feat, pc_output, fore_mask56, use_fore_mask=TRAIN_USE_FORE_MASK_XYZ)   
            rgb_loss, _ = loss_fucntion(rgb_input, rgb_output, fore_mask28, use_fore_mask=TRAIN_USE_FORE_MASK_RGB)
            
            # loss = xyz_loss
            # loss = rgb_loss
            loss = rgb_loss + xyz_loss
            
            
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
            
            loss_list.append( loss.item() )
            rgb_loss_list.append( rgb_loss.item() )
            xyz_loss_list.append( xyz_loss.item() )

        exp_lr_scheduler.step()
        print('epoch [{}/{}], loss:{:.4f} rgb:{:.4f} xyz:{:.4f}'.format(epoch + 1,
                                                  epochs,
                                                  np.mean(loss_list),
                                                  np.mean(rgb_loss_list),
                                                  np.mean(xyz_loss_list)
                                                  ))
        
        
        
        ### ----------------------------evaluate-------------------------------
        if (epoch + 1) % EVAL_INTERVAL == 0 and EVAL_FLAG:
            decoder.eval()
            
            gt_list_px = []
            pr_list_px = []
            gt_list_sp = []
            pr_list_sp = []
            aupro_list = []
            
            count = 0
            with torch.no_grad():
                for img, pc_feat, gt, label, _ in test_dataloader:
                    img = img.to(device)
                    pc_feat = pc_feat.to(device)
                    inputs = encoder(img)
                    
                    rgb_input = concater(inputs)
                    
                    channel_masked_inputs = rgb_channel_masking(inputs, channel_dim_lst)
                    
                    channel_masked_rgb_feat = concater(channel_masked_inputs)
                    
                    channel_masked_pc_feat = xyz_channel_masking(pc_feat)
                    
                    fore_mask56, fore_mask28 = get_pc_rgb_fore_mask(pc_feat, device, _class_)
                    
                    rgb_output, pc_output, rgb_latent, xyz_latent = decoder(channel_masked_rgb_feat, channel_masked_pc_feat, fore_mask56, fore_mask28) 
                    
                    xyz_ano_map, xyz_a_map = cal_anomaly_map(  pc_feat, 
                                                    pc_output, 
                                                    fore_mask56, 
                                                    USE_PC_FORE_MASK, 
                                                    img.shape[-1]
                                                    )
    
                    rgb_ano_map, rgb_a_map = cal_anomaly_map(  rgb_input, 
                                                    rgb_output, 
                                                    fore_mask28, 
                                                    USE_RGB_FORE_MASK, 
                                                    img.shape[-1]
                                                    )
                    
                    # anomaly_map = xyz_ano_map
                    # anomaly_map = rgb_ano_map
                    anomaly_map = xyz_ano_map + rgb_ano_map
                    # anomaly_map = xyz_ano_map * rgb_ano_map
                    # anomaly_map = np.maximum(xyz_ano_map, rgb_ano_map)

                    anomaly_map = gaussian_filter(anomaly_map, sigma=4)
                    
                    gt[gt > 0.5] = 1
                    gt[gt <= 0.5] = 0
        
                    if gt.max()!=0:
                        aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
                                                      anomaly_map[np.newaxis,:,:]))
                        # aupro_list.append(0)
         
                    gt = gt[:, 0, :, :]
                    gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
                    pr_list_px.extend(anomaly_map.ravel())
                    gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
                    pr_list_sp.append(np.max(anomaly_map))
                    
                    
                    
                    ## -------------- visualization  -----------------------
                    ano_map = min_max_norm(anomaly_map)
                    ano_map = cvt2heatmap(ano_map*255)
                    img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
                    img = np.uint8(min_max_norm(img)*255)
                    ano_map = show_cam_on_image(img, ano_map)
                    gt = gt.cpu().numpy().astype(int)[0]*255
                    
                    cv2.imwrite(os.path.join(img_save_base_path,
                                                  '{}_a{}.png'.format(count,'img')), img)
                    cv2.imwrite(os.path.join(img_save_base_path,
                                              '{}_c{}.png'.format(count,'gt')), gt)
                    cv2.imwrite(os.path.join(img_save_base_path,
                                              '{}_b{}.png'.format(count,'map')), ano_map)
                    count += 1
                    
                auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
                auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
                aupro_px = round(np.mean(aupro_list),3)

                print('Pixel Auroc:{:.3f}, Sample Auroc{:.3f}, Pixel Aupro{:.3}'.format(auroc_px, auroc_sp, aupro_px))
                
                torch.save({'decoder': decoder.state_dict()}, ckp_path + str(epoch) + '.pth')

if __name__ == '__main__':
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    
    batch_size = 16
    image_size = 224

    
    learning_rate = 1e-3
    lr_milestones = [50]
    weight_decay_gamma = 0.2
    EVAL_INTERVAL = 5
    
    setup_seed(12)
    
    ### train masked loss
    TRAIN_USE_FORE_MASK_RGB = True
    TRAIN_USE_FORE_MASK_XYZ = True
    
    ### inference background mask 
    USE_RGB_FORE_MASK = True
    USE_PC_FORE_MASK = True
    
    ### for resnet18 34
    # channel_dim_lst = [64, 128, 256]  
    ### for resnet50 101
    channel_dim_lst = [256, 512, 1024]   
    
    ### RGB Images Path
    DATASET_PATH = '/media/CODE/Dataset/MVTec3D-M3DM'
    ### point cloud features Path
    PonitCloudFeat_PATH = '/media/CODE/Code/M3DM/M3DM-main/datasets'
    
    
    item_list = ['cable_gland', 'cookie', 'potato', 'bagel', 'carrot',  'dowel',
                  'foam', 'peach',  'rope', 'tire']
    
    
    
    ep = 250
    for CATEGORY in item_list:
        
        train(CATEGORY, channel_dim_lst, ep, EVAL_FLAG=1)
    

    
