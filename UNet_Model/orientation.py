#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import nibabel as nib
import numpy as np
import os
import sys
import pandas as pd
import scipy
from nibabel.processing import resample_to_output

class orientation():
    def __init__(self, t1_path):
        self.path = t1_path
    
    def check(self):
        if isinstance(self.path, str):
            if os.path.isfile(self.path):
                img = nib.load(self.path)
                if nib.aff2axcodes(img.affine) != ('R', 'A', 'S'):
                    return False
                if np.min(img.shape) != np.max(img.shape):
                    return False
            else:
                for file in os.listdir(self.path):
                    if ".nii" in file:
                        img = nib.load(os.path.join(self.path, file))
                        if nib.aff2axcodes(img.affine) != ('R', 'A', 'S'):
                            return False
                        if np.min(img.shape) != np.max(img.shape):
                            return False
            return True
        
    def affine_rescale(self, img_in, voxel_size=None, out_shape=None, multiple=None):
        voxel_size_in = np.sqrt(np.sum(img_in.affine[:3, :3]**2, axis=0))
        if not voxel_size:
            voxel_size = voxel_size_in
        if not out_shape:
            out_shape = img_in.shape
        if not multiple:
            multiple = voxel_size/voxel_size_in
        trans = img_in.affine[:-1, -1]
        rzs_out = img_in.affine.copy()[:3, :3] * multiple
        centroid = img_in.affine[:3, :3] @ ((np.asarray(img_in.shape) - 1) / 2) + trans[None, :]
        t_out = centroid - rzs_out @ ((np.asarray(out_shape) - 1) / 2)
        new_affine = np.diag([0, 0, 0, 1]).astype(np.float32)
        new_affine[:3, :3] = rzs_out
        new_affine[:-1, -1] = t_out
        return new_affine

    def resample_orientation(self, from_img, orientation='RAS'
                             , cval=0, voxel_size=None
                             , order=3, out_class=None):

        if not voxel_size:
            vox_min = np.min(from_img.header.get_zooms())
            voxel_size = (vox_min, vox_min, vox_min)

        start_ornt = nib.orientations.io_orientation(from_img.affine)
        end_ornt = nib.orientations.axcodes2ornt(orientation)
        transform = nib.orientations.ornt_transform(start_ornt, end_ornt)

        # Reorient first to ensure shape matches expectations
        reoriented = from_img.as_reoriented(transform)

#         outx, outy, outz = out_shape
#         inx, iny, inz = from_img.shape
#         sxin, syin, szin = from_img.header.get_zooms()
#         sxout = sxin/outx*inx
#         syout = syin/outy*iny
#         szout = szin/outz*inz
#         voxel_size = (sxout, syout, szout)
#         print(voxel_size)
#         shx, shy, shz = from_img.shape
#         sx, sy, sz = from_img.header.get_zooms()
#         osx, osy, osz = voxel_size
#         outx = int(np.round(sx/osx*shx))
#         outy = int(np.round(sy/osy*shy))
#         outz = int(np.round(sz/osz*shz))
#         out_shape = (outx, outy, outz)
#         print(out_shape)
        
#         out_aff = self.affine_rescale(reoriented, voxel_size, out_shape)

#         # Resample input image.
#     #     out_img = resample_from_to(
#     #         from_img=from_img, to_vox_map=(out_shape, out_aff), order=order, mode="constant",
#     #         cval=cval, out_class=out_class)

#     #     try:
#     #         to_shape, to_affine = to_vox_map.shape, to_vox_map.affine
#     #     except AttributeError:
#     #         to_shape, to_affine = to_vox_map
#     #     a_to_affine = adapt_affine(out_aff, len(out_shape))
#         if out_class is None:
#             out_class = reoriented.header
#         from_n_dim = len(reoriented.shape)
#         if from_n_dim != 3:
#             raise AffineError('from_img must be 3D')
#     #     a_from_affine = adapt_affine(from_img.affine, from_n_dim)
#         to_vox2from_vox = np.linalg.inv(reoriented.affine).dot(out_aff)
#         trans = to_vox2from_vox[:-1, -1]
#         rzs = to_vox2from_vox[:3, :3]
#         data = scipy.ndimage.affine_transform(reoriented.get_fdata(),
#                                               rzs,
#                                               trans,
#                                               out_shape,
#                                               order=order,
#                                               mode='constant',
#                                               cval=cval)
        return resample_to_output(reoriented, voxel_size)

    def orient(self, multiple=None, mode='image'):
        if isinstance(self.path, str):
            if os.path.isfile(self.path):
                img = nib.load(self.path)
                print(mode, ":",file)
                if multiple:
                    aff_re = self.affine_rescale(img, multiple=multiple)
                    img = nib.Nifti1Image(img.get_fdata(), aff_re, img.header)
                vox_size = img.header.get_zooms()
                vox_min = np.min(vox_size)
                if nib.aff2axcodes(img.affine) != ('R', 'A', 'S') or np.max(vox_size) != vox_min:
                    new_img = self.resample_orientation(img, voxel_size=(vox_min, vox_min, vox_min))
                    print("original shape:", img.shape, "resample shape:", img_new.shape)
                    if nib.aff2axcodes(img_new.affine) != ('R', 'A', 'S'):
                        print('Invalid Image!')
                        sys.exit(1)
                else:
                    new_img = img
                if mode == 'mask':
                    data = img_new.get_fdata()
                    me_da = np.mean(data)
                    data[data<me_da] = 0
                    data[data>me_da] = 1
                    new_img = nib.Nifti1Image(data, new_img.affine, new_img.header)
                img_dir, img_in = os.path.split(self.path)
                os.makedirs(os.path.join(img_dir, 'new_path'))
                nib.save(new_img, os.path.join(img_dir, 'new_path','new_'+img_in))
                return os.path.join(img_dir, 'new_path')
            else:
                for file in os.listdir(self.path):
                    if ".nii" in file:
                        img = nib.load(os.path.join(self.path, file))
                        print(mode, ":", file)
                        if multiple:
                            aff_re = self.affine_rescale(img, multiple=multiple)
                            img = nib.Nifti1Image(img.get_fdata(), aff_re, img.header)
                        vox_size = img.header.get_zooms()
                        vox_min = np.min(vox_size)
                        if nib.aff2axcodes(img.affine) != ('R', 'A', 'S') or np.max(vox_size) != vox_min:
                            img_new = self.resample_orientation(img, voxel_size=(vox_min, vox_min, vox_min))
                            print("original shape:", img.shape, "resample shape:", img_new.shape)
                            if nib.aff2axcodes(img_new.affine) != ('R', 'A', 'S'):
                                print('Invalid Image!')
                                sys.exit(1)
                        else:
                            img_new = img
                        if mode == 'mask':
                            data = img_new.get_fdata()
                            me_da = np.mean(data)
                            data[data<me_da] = 0
                            data[data>me_da] = 1
                            img_new = nib.Nifti1Image(data, img_new.affine, img_new.header)
                        new_path = os.path.join(self.path, 'new_path')
                        if os.path.exists(new_path):
                            nib.save(img_new, os.path.join(new_path, 'new_'+file))
                        else:
                            os.makedirs(new_path)
                            nib.save(img_new, os.path.join(new_path, 'new_'+file))
                        
                return new_path