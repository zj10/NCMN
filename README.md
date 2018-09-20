# Non-Correlating Multiplicative Noise

This repository contains the code for the paper [Removing the Feature Correlation Effect of Multiplicative Noise](https://arxiv.org/abs/1809.07023). Non-Correlating Multiplicative Noise (NCMN) exploits batch normalization to remove the feature correlation effect of multiplicative noise (e.g. dropout).

The code is based on a [PyTorch implementation](https://github.com/szagoruyko/wide-residual-networks/tree/master/pytorch) of wide residual networks.

## Usage
For practical uses, NCMN-$0$ is simple, fast, and can be applied to any batch-normalized neural networks, while NCMN-2 yields better generalization performance on ResNets.
#### NCMN-0
```
## CIFAR-10
main.py --ncmn 0 0.35 --weightDecay 5e-6 --depth 22 --width 7.5 --dataroot ../cifar10 --save ./logs/resnet_model --ngpu 1 --gpu_id 0
## CIFAR-100
main.py --ncmn 0 0.25 --weightDecay 2e-5 --depth 22 --width 7.5 --dataset CIFAR100 --dataroot ../cifar100 --save ./logs/resnet_model --ngpu 1 --gpu_id 0
```
#### NCMN-1
```
## CIFAR-10
main.py --ncmn 1 0.35 --weightDecay 5e-6 --depth 22 --width 7.5 --dataroot ../cifar10 --save ./logs/resnet_model --ngpu 1 --gpu_id 0
## CIFAR-100
main.py --ncmn 1 0.25 --weightDecay 2e-4 --depth 22 --width 7.5 --dataset CIFAR100 --dataroot ../cifar100 --save ./logs/resnet_model --ngpu 1 --gpu_id 0
```
#### NCMN-2
```
## CIFAR-10
main.py --ncmn 2 0.4 --lr 0.03 --weightDecay 2e-5 --depth 22 --width 7.5 --dataroot ../cifar10 --save ./logs/resnet_model --ngpu 1 --gpu_id 0
main.py --ncmn 2 0.45 --lr 0.03 --weightDecay 2e-5 --depth 28 --width 10 --dataroot ../cifar10 --save ./logs/resnet_model --ngpu 1 --gpu_id 0
## CIFAR-100
main.py --ncmn 2 0.3 --weightDecay 2e-4 --depth 22 --width 7.5 --dataset CIFAR100 --dataroot ../cifar100 --save ./logs/resnet_model --ngpu 1 --gpu_id 0
main.py --ncmn 2 0.35 --weightDecay 2e-4 --depth 28 --width 10 --dataset CIFAR100 --dataroot ../cifar100 --save ./logs/resnet_model --ngpu 1 --gpu_id 0
```