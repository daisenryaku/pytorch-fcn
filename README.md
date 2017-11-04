# pytorch-fcn
Fully Convolutional Networks Implemented in PyTorch

## Requirements
- pytorch == 0.2.0
- torchvision == 0.1.7
- scipy
- tqdm

**Note**:
- You can install all the python packages one-line by running:
```shell
sudo pip install -r requirements.txt
```

## To train the model:
```shell
python main.py train [--arch [ARCH]] [--dataset [DATASET]]
  -- arch Architecture to use ['fcn8s']
  -- dataset Dataset to use ['pascal']
```
