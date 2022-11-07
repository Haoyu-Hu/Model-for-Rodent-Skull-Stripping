#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class Args():
    def __init__(self, train_t1w=None, train_msk=None, out_dir=None
                 , validate_t1w=None, validate_msk=None, init_model=None
              , conv_block=5, input_slice=3
              , kernel_root=16, rescale_dim=256
              , num_epoch=40, learning_rate=0.0001
              , rotation=False):
        self.train_t1w = train_t1w
        self.train_msk = train_msk
        self.out_dir = out_dir
        self.validate_t1w = validate_t1w
        self.validate_msk = validate_msk
        self.init_model = init_model
        self.conv_block = conv_block
        self.input_slice = input_slice
        self.kernel_root = kernel_root
        self.rescale_dim = rescale_dim
        self.num_epoch = num_epoch
        self.learning_rate = learning_rate
        self.rotation = rotation
        
    def change_train_t1w(self, train_t1w):
        self.train_t1w = train_t1w
        
    def change_train_msk(self, train_msk):
        self.train_msk = train_msk
        
    def change_out_dir(self, out_dir):
        self.out_dir = out_dir
        
    def change_validate_t1w(self, validate_t1w):
        self.validate_t1w = validate_t1w
        
    def change_validate_msk(self, validate_msk):
        self.validate_msk = validate_msk
        
    def change_init_model(self, init_model):
        self.init_model = init_model

class Args_test():
    def __init__(self, test_t1w=None, test_msk=None, out_dir=None, test_model=None
              , conv_block=5, input_slice=3
              , kernel_root=16, rescale_dim=256
              , mask_suffix='pre_mask'
              , erosion_dilation_iteration=0):
        self.test_t1w = test_t1w
        self.test_msk = test_msk
        self.out_dir = out_dir
        self.test_model = test_model
        self.conv_block = conv_block
        self.input_slice = input_slice
        self.kernel_root = kernel_root
        self.rescale_dim = rescale_dim
        self.mask_suffix = mask_suffix
        self.erosion_dilation_iteration = erosion_dilation_iteration
        
    def change_test_t1w(self, test_t1w):
        self.test_t1w = test_t1w
        
    def change_test_msk(self, test_msk):
        self.test_msk = test_msk
        
    def change_out_dir(self, out_dir):
        self.out_dir = out_dir
        
    def change_test_model(self, test_model):
        self.test_model = test_model

class Args_baco():
    def __init__(self, input=None, output=None, mask=None
                 , sf=None, nfl=None, iters=20):
        self.input = input
        self.output = output
        self.mask = mask
        self.sf = sf
        self.nfl = nfl
        self.iters = iters