import sys
import torch
import argparse
import numpy as np
import scipy.misc as misc
from tqdm import tqdm

from config import config
from data import pascalVOCLoader

import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.models as models

def test(args):
    img = misc.imread(args.img_path)
    loader = pascalVOCLoader(config, is_transform=True)
    
    img = img[:, :, ::-1]
    img = img.astype(np.float64)
    img -= loader.mean
    img = misc.imresize(img, (loader.img_size[0], loader.img_size[1]))
    img = img.astype(float) / 255.0
    
    # NHWC -> NCWH
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    model = torch.load(args.model_path)
    model.eval()

    if torch.cuda.is_available():
        model.cuda(0)
        images = Variable(img.cuda(0))
    else:
        images = Variable(img)

    outputs = model(images)
    pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=1)
    decoded = loader.decode_segmap(pred[0])
    misc.imsave(args.out_path, decoded)

if __name__ == '_main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='1.pkl')
    parser.add_argument('--img_path', default=None)
    parser.add_argument('--out_path', default=None)
    parser.parse_args()
    test(args)
