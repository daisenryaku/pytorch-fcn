import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

import argparse

import sys
from config import config
from data import pascalVOCLoader
from models import get_model
from utils import lr_scheduler, cross_entropy2d

def train(args):
    # 1. Define data loader
    loader = pascalVOCLoader(config, is_transform=True, img_size=(args.img_rows, args.img_cols))
    trainloader = DataLoader(loader, batch_size=args.batch_size, num_workers=4, shuffle=True)

    # 2. Define model
    model = get_model()
    if torch.cuda.is_available():
        model.cuda(0)
    optimizer = torch.optim.SGD(model.parameters(),lr=args.l_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    # 3. Loop
    for epoch in range(args.n_epoch):
        for i, (images, labels) in enumerate(trainloader):
            if torch.cuda.is_available():
                images, labels = Variable(images.cuda(0)), Variable(labels.cuda(0))
            else:
                images, labels = Variable(images), Variable(labels)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = cross_entropy2d(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if (i+1)%20 == 0:
                print ("Epoch:{}/{} Loss:{}".format(epoch+1, args.n_epoch, loss.data[0]))
        torch.save(model, "{}_{}_{}_{}.pkl".format(args.arch, args.dataset, args.features_scale, epoch))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_rows', default=256)
    parser.add_argument('--img_cols', default=256)
    parser.add_argument('--n_epoch', default=100)
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--l_rate', default=1e-5)
    parser.add_argument('--momentum', default=0.99)
    parser.add_argument('--weight_decay', default=5e-4)
    parser.add_argument('--feature_scale', default=1)
    args = parser.parse_args()
    train(args)
