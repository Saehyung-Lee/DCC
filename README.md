# Dataset Condensation with Contrastive Signals

This repository is the official implementation of [Dataset Condensation with Contrastive Signals (DCC)](https://arxiv.org/abs/2202.02916), published as a conference paper at ICML 2022.
The implementation is based on (https://github.com/VICO-UoE/DatasetCondensation).

## Prerequisites

* pytorch (1.2.0)
* numpy (1.15.1)
* torchvision (0.4.0)
* scipy (1.1.0)

## Training and evaluation

To train the DCC (or DSAC) model in the paper, run this command:

```train and eval
python main.py --ipc <1, 10, or 50> --model ConvNet --dataset <CIFAR10, CIFAR100, or imagenet> (--imagenet_group <fine-grained dataset>) --method <DC or DSA> --contrast --save_path <save path name>
```

> Please download ImageNet32x32 at (https://image-net.org/download-images)
