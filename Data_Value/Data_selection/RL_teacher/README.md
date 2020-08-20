# Data selection using reinforcement learning teacher models 

## Cifar10 Dataset:
First download splitted dataset.
```
mkdir data
cd data
wget http://www.cs.toronto.edu/~cqwang/projects/active-learning/data/learning-to-teach/cifar10-splitted.tar.gz
tar -xzvf cifar10-splitted.tar.gz
```

# How to run
```
python train.py --hparams=cifar10_l2t
```


## Sources
Methods based on [Learning to teach](https://openreview.net/pdf?id=HJewuJWCZ) and [Optimizing Data Usage via Differentiable Rewards](https://arxiv.org/abs/1911.10088)
Code based on [Learning to teach](https://openreview.net/pdf?id=HJewuJWCZ) implementation by [hyzcn](https://github.com/hyzcn/learning-to-teach)


