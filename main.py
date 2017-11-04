import torch
import torch.nn as nn
import torch.nn.functional ad F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.models as models

from model import get_model, cross_entropy2d
from data import get_loader
from utils import poly_lr_sheduler
from config import opt

def train(**kargs):
    opt.parse(kwargs)
    
    data_loader = get_loader(opt.dataset)
    data_path = get_data_path(opt.dataset)
    loader = data_loader(data_path, istransform=True, img_size=(opt.img_rows, opt.img_cols))
    trainloader = DataLoader(loader, batch_size=opt.batch_size, num_workers=4, shuffle=True)
    
    model = get_model(opt.arch, loader.n_classes)
    model.cuda(0).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

    for epoch in range(args.n_epoch):
        for i, (images, labels) in enumerate(trainloader):
            images = Variable(images.cuda(0))
            labels = Variable(labels.cuda(0))
            iter = len(trainloader) * epoch + i
            poly_lr_scheduler(optimizer, args.lr, iter)
            optimizer.zero_grad()
            outputs = model(images)
            loss = cross_entropy2d(outputs, labels)
            loss.backward()
            optimizer.step()
            if (i+1) % 20 == 0:
                print("Epoch: [%d/%d] Loss:%.5f" % (epoch+1, args.n_epoch, loss.data[0]))
        if epoch % opt.save_freq == 0:
            net.save()

if __name__ == '__main__':
    import file
    file.Fire()
