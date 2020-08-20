# Data selection using reinforcement learning teacher models 

## Cifar10 Dataset
```
mkdir data
cd data
wget http://www.cs.toronto.edu/~cqwang/projects/active-learning/data/learning-to-teach/cifar10-splitted.tar.gz
tar -xzvf cifar10-splitted.tar.gz
```

## Setup 
```
pip install -r requirements.txt
```

## How to run
First, you need to start Tensorboard for logging. 
```
tensorboard --logdir log --bind_all
```
To train: 
```
> python train.py -h
usage: train.py [-h] [--hparams HPARAMS] [--run RUN]

Data selection using RL

optional arguments:
  -h, --help         show this help message and exit
  --hparams HPARAMS  Choose hyper parameter configuration.
                     [cifar10_l2t, multi_cifar10_l2t, cifar10_l2t_augment, cifar10_l2t_vae]
  --run RUN          experiment name
```


## Sources
Methods based on [Learning to teach](https://openreview.net/pdf?id=HJewuJWCZ) and [Optimizing Data Usage via Differentiable Rewards](https://arxiv.org/abs/1911.10088)

Code based on [Learning to teach](https://openreview.net/pdf?id=HJewuJWCZ) implementation by [hyzcn](https://github.com/hyzcn/learning-to-teach)


