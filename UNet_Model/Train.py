#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from model import UNet2d
from function import predict_volumes
from dataset import BlockDataset, VolumeDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import os, sys, pickle
import argparse
import numpy as np

class Train():
    def __init__(self, args=None):
       self.args = args

    def train(self):
        args = self.args
        print("===================================Training Model===================================")

        NoneType=type(None)

        if not os.path.exists(args.train_msk) or not os.path.exists(args.train_t1w):
            print("Invalid train directory, please check again!")
            sys.exit(2)

        use_validate=True
        if isinstance(args.validate_msk, NoneType) or isinstance(args.validate_t1w, NoneType) or                 not os.path.exists(args.validate_msk) or not os.path.exists(args.validate_t1w):
            use_validate=False
            print("NOTE: Do not use validate dataset.")

        use_gpu=torch.cuda.is_available()
        model=UNet2d(dim_in=args.input_slice, num_conv_block=args.conv_block, kernel_root=args.kernel_root)
        if isinstance(args.init_model, str):
            if not os.path.exists(args.init_model):
                print("Invalid init model, please check again!")
                sys.exit(2)
            checkpoint=torch.load(args.init_model, map_location={'cuda:0':'cpu'})
            model.load_state_dict(checkpoint['state_dict'])

        if use_gpu:
            model.cuda()
            cudnn.benchmark=True

        # optimizer
        optimizerSs=optim.Adam(model.parameters(), lr=args.learning_rate)

        # loss function
        criterionSs=nn.CrossEntropyLoss()
        if use_gpu:
            criterionSs.cuda()

        

        blk_batch_size=20

        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)

        # Init Dice and Loss Dict
        DL_Dict=dict()
        dice_list=list()
        loss_list=list()

        if use_validate:
            valid_model=nn.Sequential(model, nn.Softmax2d())
            dice_dict=predict_volumes(valid_model, rimg_in=None, cimg_in=args.validate_t1w, bmsk_in=args.validate_msk, 
                rescale_dim=args.rescale_dim, num_slice=args.input_slice, save_nii=False, save_dice=True)
            dice_array=np.array([v for v in dice_dict.values()])
            DL_Dict["origin_dice"]=dice_array
            print("Origin Dice: %.4f +/- %.4f" % (dice_array.mean(), dice_array.std()))

        for epoch in range(0, args.num_epoch):
            lossSs_v=[]
            print("Begin Epoch %d" % epoch)
            volume_dataset=VolumeDataset(rimg_in=None, cimg_in=args.train_t1w, bmsk_in=args.train_msk)
            volume_loader=DataLoader(dataset=volume_dataset, batch_size=1, shuffle=True, num_workers=0)
            for i, (cimg, bmsk) in enumerate(volume_loader):
                block_dataset=BlockDataset(rimg=cimg, bfld=None, bmsk=bmsk, num_slice=args.input_slice, rescale_dim=args.rescale_dim)
                block_loader=DataLoader(dataset=block_dataset, batch_size=blk_batch_size, shuffle=True, num_workers=0)
                for j, (cimg_blk, bmsk_blk) in enumerate(block_loader):
                    bmsk_blk=bmsk_blk[:,1,:,:]
                    cimg_blk, bmsk_blk=Variable(cimg_blk), Variable(bmsk_blk)
                    if use_gpu:
                        cimg_blk=cimg_blk.cuda()
                        bmsk_blk=bmsk_blk.cuda()
                    pr_bmsk_blk=model(cimg_blk)

                    # Loss Backward
                    lossSs=criterionSs(pr_bmsk_blk, bmsk_blk)
                    optimizerSs.zero_grad()
                    lossSs.backward()
                    optimizerSs.step()

                    if use_gpu:
                        lossSs=lossSs.cuda()
                
                    
                    lossSs_v.append(lossSs.cpu().data.detach().numpy())

                    print('\tEpoch:%.2d [%.3d/%.3d (%.4d/%.4d)]\tLoss Ss: %.6f' %                         (   epoch, i, len(volume_loader.dataset)-1, 
                            j*blk_batch_size, len(block_loader.dataset), 
                            lossSs.data.detach()
                        )
                    )
#             if isinstance(args.rotation, list):
#                 for rot in args.rotation:
#                     print("Rotation is %d"%(rot))
#                     volume_dataset=VolumeDataset(rimg_in=None, cimg_in=args.train_t1w, bmsk_in=args.train_msk
#                              , rotation=rot, axis=args.axis)
#                     volume_loader=DataLoader(dataset=volume_dataset, batch_size=1, shuffle=True, num_workers=0)
#                     for i, (cimg, bmsk) in enumerate(volume_loader):
#                         block_dataset=BlockDataset(rimg=cimg, bfld=None, bmsk=bmsk, num_slice=args.input_slice, rescale_dim=args.rescale_dim)
#                         block_loader=DataLoader(dataset=block_dataset, batch_size=blk_batch_size, shuffle=True, num_workers=0)
#                         for j, (cimg_blk, bmsk_blk) in enumerate(block_loader):
#                             bmsk_blk=bmsk_blk[:,1,:,:]
#                             cimg_blk, bmsk_blk=Variable(cimg_blk), Variable(bmsk_blk)
#                             if use_gpu:
#                                 cimg_blk=cimg_blk.cuda()
#                                 bmsk_blk=bmsk_blk.cuda()
#                             pr_bmsk_blk=model(cimg_blk)

#                             # Loss Backward
#                             lossSs=criterionSs(pr_bmsk_blk, bmsk_blk)
#                             optimizerSs.zero_grad()
#                             lossSs.backward()
#                             optimizerSs.step()

#                             if use_gpu:
#                                 lossSs=lossSs.cuda()

#                             lossSs_v.append(lossSs.data.detach().numpy())

#                             print('\tEpoch:%.2d [%.3d/%.3d (%.4d/%.4d)]\tLoss Ss: %.6f' %                         (   epoch, i, len(volume_loader.dataset)-1, 
#                                     j*blk_batch_size, len(block_loader.dataset), 
#                                     lossSs.data.detach()
#                                 )
#                             )
#             elif isinstance(args.rotation, int) or isinstance(args.rotation, float):
#                 print("Rotation is %d"%(args.rotation))
#                 volume_dataset=VolumeDataset(rimg_in=None, cimg_in=args.train_t1w, bmsk_in=args.train_msk
#                          , rotation=args.rotation, axis=args.axis)
#                 volume_loader=DataLoader(dataset=volume_dataset, batch_size=1, shuffle=True, num_workers=0)
#                 for i, (cimg, bmsk) in enumerate(volume_loader):
#                     block_dataset=BlockDataset(rimg=cimg, bfld=None, bmsk=bmsk, num_slice=args.input_slice, rescale_dim=args.rescale_dim)
#                     block_loader=DataLoader(dataset=block_dataset, batch_size=blk_batch_size, shuffle=True, num_workers=0)
#                     for j, (cimg_blk, bmsk_blk) in enumerate(block_loader):
#                         bmsk_blk=bmsk_blk[:,1,:,:]
#                         cimg_blk, bmsk_blk=Variable(cimg_blk), Variable(bmsk_blk)
#                         if use_gpu:
#                             cimg_blk=cimg_blk.cuda()
#                             bmsk_blk=bmsk_blk.cuda()
#                         pr_bmsk_blk=model(cimg_blk)

#                         # Loss Backward
#                         lossSs=criterionSs(pr_bmsk_blk, bmsk_blk)
#                         optimizerSs.zero_grad()
#                         lossSs.backward()
#                         optimizerSs.step()

#                         if use_gpu:
#                             lossSs=lossSs.cuda()

#                         lossSs_v.append(lossSs.data.detach().numpy())

#                         print('\tEpoch:%.2d [%.3d/%.3d (%.4d/%.4d)]\tLoss Ss: %.6f' %                         (   epoch, i, len(volume_loader.dataset)-1, 
#                                 j*blk_batch_size, len(block_loader.dataset), 
#                                 lossSs.data.detach()
#                             )
#                         )
            if args.rotation:
                print("Now train the rotated image")
                
                volume_dataset=VolumeDataset(rimg_in=None, cimg_in=args.train_t1w, bmsk_in=args.train_msk
                         , rotation=True)
                volume_loader=DataLoader(dataset=volume_dataset, batch_size=1, shuffle=True, num_workers=0)
                for i, (cimg, bmsk) in enumerate(volume_loader):
                    block_dataset=BlockDataset(rimg=cimg, bfld=None, bmsk=bmsk, num_slice=args.input_slice, rescale_dim=args.rescale_dim)
                    block_loader=DataLoader(dataset=block_dataset, batch_size=blk_batch_size, shuffle=True, num_workers=0)
                    for j, (cimg_blk, bmsk_blk) in enumerate(block_loader):
                        bmsk_blk=bmsk_blk[:,1,:,:]
                        cimg_blk, bmsk_blk=Variable(cimg_blk), Variable(bmsk_blk)
                        if use_gpu:
                            cimg_blk=cimg_blk.cuda()
                            bmsk_blk=bmsk_blk.cuda()
                        pr_bmsk_blk=model(cimg_blk)

                        # Loss Backward
                        lossSs=criterionSs(pr_bmsk_blk, bmsk_blk)
                        optimizerSs.zero_grad()
                        lossSs.backward()
                        optimizerSs.step()

                        if use_gpu:
                            lossSs=lossSs.cuda()

                        lossSs_v.append(lossSs.cpu().data.detach().numpy())

                        print('\tRotated_Epoch:%.2d [%.3d/%.3d (%.4d/%.4d)]\tLoss Ss: %.6f' %                         (   epoch, i, len(volume_loader.dataset)-1, 
                                j*blk_batch_size, len(block_loader.dataset), 
                                lossSs.data.detach()
                            )
                        )
            loss=np.array(lossSs_v).sum()
            

            if use_validate:
                valid_model=nn.Sequential(model, nn.Softmax2d())
                dice_dict=predict_volumes(valid_model, rimg_in=None, cimg_in=args.validate_t1w, bmsk_in=args.validate_msk, save_dice=True)
                dice_array=np.array([v for v in dice_dict.values()])
                dice_list.append(dice_array)
                loss_list.append(loss)
                print("\tEpoch: %d; Dice: %.4f +/- %.4f; Loss: %.4f" % (epoch, dice_array.mean(), dice_array.std(), loss))
            else:
                dice_array=[]
                print("\tEpoch: %d; Loss: %.4f" % (epoch, loss))

            if (epoch)%1==0:
                checkpoint={
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizerSs': optimizerSs.state_dict(),
                    'lossSs': lossSs_v,
                    'validate_dice': dice_array
                    }
                torch.save(checkpoint, os.path.join(args.out_dir, 'model-%.2d-epoch.model' % ( epoch )))
        DL_Dict["dice"]=dice_list
        DL_Dict["loss"]=loss_list
        with open(os.path.join(args.out_dir, "DiceAndLoss.pkl"), "wb") as handle:
            pickle.dump((dice_list, loss_list), handle)
            
        return DL_Dict

