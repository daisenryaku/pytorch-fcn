import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data

from model import fcn8s
from data import VOCbase
from loss import CrossEntropyLoss2d
from config import config

def train(args):
    data_path = config['voc_path']
    loader = VOCbase(data_path, is_transform=True, img_size=(args.img_rows, args.img_cols))
    trainloader = data.DataLoader(loader, batch_size=args.batch_size, num_workers=4, shuffle=True)

    model = fcn8s() 
    model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.l_rate, momentum=0.99, weight_decay=5e-4)

    for epoch in range(args.n_epoch):
        for i, (images, labels) in enumerate(trainloader):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
            optimizer.zero_grad()
            outputs = model(images)
            loss = CrossEntropyLoss2d(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i+1) % 20 == 0:
                print("Epoch [%d/%d] Loss: %.4f" % (epoch+1, args.n_epoch, loss.data[0]))

        torch.save(model, "{}_{}_{}.pkl".format(args.arch, args.dataset, epoch))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='fcn8s')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal')
    parser.add_argument('--img_rows', nargs='?', type=int, default=256)
    parser.add_argument('--img_cols', nargs='?', type=int, default=256)
    parser.add_argument('--n_epoch', nargs='?', type=int, default=1)
    parser.add_argument('--batch_size', nargs='?', type=int, default=1)
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-5)  
    args = parser.parse_args()
    train(args)
