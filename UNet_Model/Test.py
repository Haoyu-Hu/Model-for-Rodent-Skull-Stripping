#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import numpy as np
import scipy.ndimage as snd
from torch.autograd import Variable
from dataset import VolumeDataset, BlockDataset
from torch.utils.data import DataLoader
from model import MultiSliceBcUNet, MultiSliceSsUNet, MultiSliceModel, UNet2d
from function import predict_volumes
import os, sys, pickle
import nibabel as nib

class Test():
    def __init__(self, args=None):
        self.args = args
        
    def test(self):
        args = self.args
    
        NoneType=type(None)

        print("===================================Testing Model====================================")

        if not os.path.exists(args.test_msk) or not os.path.exists(args.test_t1w):
            print("Invalid test directory, please check again!")
            sys.exit(2)

        if not os.path.exists(args.test_model):
            print("Invalid test model, please check again!")
            sys.exit(2)

        train_model=UNet2d(dim_in=args.input_slice, num_conv_block=args.conv_block, kernel_root=args.kernel_root)
        checkpoint=torch.load(args.test_model, map_location={'cuda:0':'cpu'})
        train_model.load_state_dict(checkpoint['state_dict'])

        model=nn.Sequential(train_model, nn.Softmax2d())
        dice_dict=predict_volumes(model, rimg_in=None, cimg_in=args.test_t1w, bmsk_in=args.test_msk
                                  , rescale_dim=args.rescale_dim, save_nii=True, nii_outdir=args.out_dir, save_dice=True
                                  , suffix=args.mask_suffix, ed_iter=args.erosion_dilation_iteration)
        dice_array=np.array([v for v in dice_dict.values()])
        print("\t%.4f +/- %.4f" % (dice_array.mean(), dice_array.std()))
        with open(os.path.join(args.out_dir, "Dice.pkl"), 'wb') as f:
            pickle.dump(dice_dict, f)
            
        return dice_dict


    def Use(self):
        args = self.args
    
        NoneType=type(None)

        print("===================================Using Model====================================")

        if not os.path.exists(args.test_t1w):
            print("Invalid test directory, please check again!")
            sys.exit(2)

        if not os.path.exists(args.test_model):
            print("Invalid test model, please check again!")
            sys.exit(2)

        train_model=UNet2d(dim_in=args.input_slice, num_conv_block=args.conv_block, kernel_root=args.kernel_root)
        checkpoint=torch.load(args.test_model, map_location={'cuda:0':'cpu'})
        train_model.load_state_dict(checkpoint['state_dict'])

        model=nn.Sequential(train_model, nn.Softmax2d())
        predict_volumes(model, rimg_in=None, cimg_in=args.test_t1w, bmsk_in=None
                                  , rescale_dim=args.rescale_dim, save_nii=True, nii_outdir=args.out_dir, save_dice=False
                                  , suffix=args.mask_suffix, ed_iter=args.erosion_dilation_iteration)

