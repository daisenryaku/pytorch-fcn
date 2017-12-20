import argparse
from tqdm import tqdm

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import FCN8s
from data import VOCbase
from utils import CrossEntropyLoss2d
from config import config

def train(args):
    data_path = config['voc_path']
    loader = VOCbase(data_path, is_transform=True, img_size=(args.img_rows, args.img_cols))
    trainloader = DataLoader(loader, batch_size=args.batch_size, num_workers=4, shuffle=True)

    model = FCN8s() 
    model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.l_rate, momentum=0.99, weight_decay=5e-4)

    for epoch in range(args.n_epoch):
        for i, (images, labels) in tqdm(enumerate(trainloader)):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
            optimizer.zero_grad()
            outputs = model(images)
            loss = CrossEntropyLoss2d(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i+1) % 40 == 0:
                print("Epoch [%d/%d] Loss: %.4f" % (epoch+1, args.n_epoch, loss.data[0]))
        
        model.save()

def val(args):
    data_path = config['voc_path']
    loader = VOCbase(data_path, is_transform=True, img_size=(args.img_rows, args.img_cols))
    valloader = DataLoader(loader, batch_size=args.batch_size, num_workers=4)

    model = FCN8s()
    model.load(args.model_path)
    model.cuda()
    model.eval()

    gts, preds = [], []
    for i, (images, labels) in tqdm(enumerate(valloader)):
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        outputs = model(images)
        pred = outputs.data.max(1)[1].cpu().numpy()
        gt = labels.data.cpu().numpy()

        for gt_, pred_ in zip(gt, pred):
            gts.append(gt_)
            preds.append(pred_)

    score, class_iou = scores(gts, preds, n_class=n_classes)

    for k, v in score.items():
        print(k, v)
    for i in range(n_classes):
        print(i, class_iou[i])

def test(args):
    img = misc.imread(args.img_path)
    data_path = config['voc_path']
    loader = VOCbase(data_path, is_transform=True)

    img = img[:, :, ::-1]
    img = img.astype(np.float64)
    img -= loader.mean
    img = misc.imresize(img, (loader.img_size[0], loader.img_size[1]))
    img = img.astype(float) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    model = FCN8s()
    model.load(args.model_path)
    model.cuda()
    images = Variable(img.cuda())
    outputs = model(images)
    pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
    decoded = loader.decode_segmap(pred)
    out_path = args.out_path + args.in_path.split('/')[-1]
    misc.imsave(out_path, decoded)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FCN Hyperparams')
    parser.add_argument('--phase', type=str, default='train')
    # params for train phase
    parser.add_argument('--img_rows', type=int, default=256)
    parser.add_argument('--img_cols', type=int, default=256)
    parser.add_argument('--n_epoch', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--l_rate', type=float, default=1e-5)
    # params for test phase
    parser.add_argument('--model_path', type=str, default='./checkpoints/FCN8s_1220_1310.pkl')
    parser.add_argument('--in_path', type=str, default=config['voc_path'] + 'JPEGImages/2008_000002.jpg')
    parser.add_argument('--out_path', type=str, default='./results/')
    
    args = parser.parse_args()
    if args.phase == 'train':
        train(args)
    elif args.phase == 'test':
        test(args)
    elif args.phase == 'val':
        val(args)
    else:
        raise ValueError('only support train / val / test phase.')

