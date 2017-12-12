# pytorch-fcn
[Fully Convolutional Networks](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) implemented in PyTorch

## Requirements
- pytorch
- torchvision
- visdom
- scipy
- tqdm

**Note**:
- You can install all the python packages one-line by running:
```shell
sudo pip install -r requirements.txt
```

## Data
Support Pascal VOC 2012 dataset. 
1. Download data from [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#data). 
2. Download benckmark from [here](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/)
3. Extract the *VOCtrainval_11-May-2012.tar* and *benchmark_RELEASE.tgz*, modify the data path in *config.py*.

## To train the model:
```shell
python main.py train [--arch [ARCH]] [--dataset [DATASET]]
  -- arch Architecture to use ['fcn8s']
  -- dataset Dataset to use ['pascal']
```
