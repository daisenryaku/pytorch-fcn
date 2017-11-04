#coding:utf8
import time
import warnings

class DefaultConfig(object):
    pascal_data_path = '~/VOCdevkit/VOC2012/'
    sbd_data_path = '~/benchmark_RELEASE'

    arch = 'fcn8s'
    dataset = 'pascal'
    img_rows = 256
    img_cols = 256
    n_epoch = 10
    save_freq = 5
    batch_size = 1
    lr = 1e-4
    momentum = 0.99
    weight_decay = 5e-4

    def parse(self, kwargs):
        for k, v in kwargs.iteritems():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribute{}".forma)
            setattr(self, k, v)
        print('user config:')
        for k, v in self.__class__.__dict__.iteritems():
            if not k.startswith('__'):
                print (k, getattr(self, k))

opt = DefaultConfig()
