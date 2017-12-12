import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import os
import scipy.misc as m
import scipy.io as io
import numpy as np
import collections
import matplotlib.pyplot as plt

class pascalVOCLoader(Dataset):
    def __init__(self, config, split="train_aug", is_transform=False, img_size=512):
        self.root = config['voc_path']
        self.sbd_path = config['sbd_path']
        self.split = split
        self.is_transform = is_transform
        self.n_classes = 21
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = collections.defaultdict(list)

        for split in ["train", "val", "trainval"]:
            file_list = tuple(open(self.root + '/ImageSets/Segmentation/' + split + '.txt', 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list

        if not os.path.isdir(self.root + '/SegmentationClass/pre_encoded'):
            self.setup(pre_encode=True)
        else:
            self.setup(pre_encode=False)

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        img_path = self.root + '/JPEGImages/' + img_name + '.jpg'
        lbl_path = self.root + '/SegmentationClass/pre_encoded/' + img_name + '.png'
        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)
        lbl = m.imread(lbl_path)
        lbl = np.array(lbl, dtype=np.uint32)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)
        
        return img, lbl

    def transform(self, img, lbl):
        img = img[:, :, ::-1]
        img = img.astype(np.float64)
        img -= self.mean
        img = m.imresize(img, (self.img_size[0], self.img_size[1]))
        img = img.astype(float) / 255.0
        # NHWC -> NCWH
        img = img.transpose(2, 0, 1)

        lbl[lbl==255] = 0
        lbl = lbl.astype(float)
        lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), 'nearest', mode='F')
        lbl = lbl.astype(int)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl


    def get_pascal_labels(self):
        return np.asarray([[0,0,0], [128,0,0], [0,128,0], [128,128,0], [0,0,128], [128,0,128],
                              [0,128,128], [128,128,128], [64,0,0], [192,0,0], [64,128,0], [192,128,0],
                              [64,0,128], [192,0,128], [64,128,128], [192,128,128], [0, 64,0], [128, 64, 0],
                              [0,192,0], [128,192,0], [0,64,128]])


    def encode_segmap(self, mask):
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for i, label in enumerate(self.get_pascal_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = i
        label_mask = label_mask.astype(int)
        return label_mask


    def decode_segmap(self, temp, plot=False):
        label_colours = self.get_pascal_labels()
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

    def setup(self, pre_encode=False):
        sbd_path = self.sbd_path
        voc_path = self.root
        target_path = self.root + '/SegmentationClass/pre_encoded/'
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        
        sbd_train_list = tuple(open(sbd_path + '/dataset/train.txt', 'r'))
        sbd_train_list = [id_.rstrip() for id_ in sbd_train_list]

        self.files['train_aug'] = self.files['train'] + sbd_train_list

        if pre_encode:
            for i in tqdm(sbd_train_list):
                lbl_path = sbd_path + '/dataset/cls/' + i + '.mat'
                lbl = io.loadmat(lbl_path)['GTcls'][0]['Segmentation'][0].astype(np.int32)
                lbl = m.toimage(lbl, high=lbl.max(), low=lbl.min())
                m.imsave(target_path + i + '.png', lbl)
            
            for i in tqdm(self.files['trainval']):
                lbl_path = self.root + '/SegmentationClass/' + i + '.png'
                lbl = self.encode_segmap(m.imread(lbl_path))
                lbl = m.toimage(lbl, high=lbl.max(), low=lbl.min())
                m.imsave(target_path + i + '.png', lbl)


if __name__ == '__main__':
    config = {}
    config['voc_path'] = "/home/z/VOCdevkit/VOC2012"
    config['sbd_path'] = "/home/z/benchmark_RELEASE"
    loader = pascalVOCLoader(config, is_transform=True)
    trainloader = DataLoader(loader, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i==0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
            plt.imshow(loader.decode_segmap(labels.numpy()[i+1]))
            plt.show()
