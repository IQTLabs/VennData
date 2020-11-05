from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from PIL import Image

import os
import torch
import numpy as np
from functools import partial
import pickle

import torch.utils.data as data
import torchvision
from misc.utils import to_var



class Cifar10Dataloader(data.Dataset):

    def __init__(self, configs):
        split = configs['split']
        root = configs['root']
        self.transform = configs['transform']
        self.split = split
        #pickle.load = partial(pickle.load, encoding="latin1")
        #pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        #data_dict = torch.load(os.path.join(root, 'data', 'cifar10', split + '.pth'))
        data_dict = torch.load(os.path.join(root, 'data', 'cifar10', split + '.pth'), encoding='latin1')
        #                       pickle_module=pickle)
        labels = []
        data = []
        # data_dict is [10 x (label, data_list=(3072=3*32*32))]
        # data_dict: dictionary with 10 keys, each item is a list of images
        for label, data_list in data_dict.items():
            if 'label' in configs and configs['label'] != label:
                continue # skip if config['label'] does not match; used when doing label based dataloader for multi teacher
            n_samples = len(data_list)
            labels.extend([label] * n_samples)
            data.extend(data_list)
        print('Loaded split {}: {:d} data, {:d} labels'.format(self.split, len(labels), len(data)))
        self.data = np.concatenate([x.reshape(1, -1) for x in data])
        #print ('Concatenated shape:', self.data.shape)
        self.data = self.data.reshape((-1, 3, 32, 32))
        self.data = self.data.transpose((0, 2, 3, 1)) # convert to HWC
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def collate_fn(self, data):
        inputs, labels = zip(*data)
        # print (len(inputs), inputs[0].shape)
        labels = torch.LongTensor(labels)
        inputs = torch.cat([x.view(1, 3, 32, 32) for x in inputs], 0)
        return inputs, labels

def get_dataloader(configs, seed=None):
    batch_size = configs['batch_size']
    shuffle = configs['shuffle']

    if configs['dataset'] == 'cifar10' or (configs['dataset'] ==  'multi_cifar10' and configs['split'] != 'dev'):
        dataset = Cifar10Dataloader(configs)
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=0,
                                                  collate_fn=dataset.collate_fn)
        return data_loader

    elif configs['dataset'] == 'multi_cifar10' and configs['split'] == 'dev':
        multi_dataloaders = {}
        for i in range(10):
            configs['label'] = i
            dataset = Cifar10Dataloader(configs)
            data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                      batch_size=batch_size,
                                                      shuffle=shuffle,
                                                      num_workers=0,
                                                      collate_fn=dataset.collate_fn)
            multi_dataloaders[i] = data_loader
        return multi_dataloaders

    elif configs['dataset'] == 'bionic_regroup':
        dataset = torchvision.datasets.ImageFolder(os.path.join(configs['root'], 'data','lab41_bio','bionic_regroup'), transform=configs['transform'])

        teacher_split = int(0.4 * len(dataset))
        student_split = int(0.4 * len(dataset))
        dev_split = int(0.10 * len(dataset))
        test_split = int(0.10 * len(dataset))
        np.random.seed(seed)
        indices = np.random.permutation(len(dataset))
        splits = {
                    'teacher_train': indices[:teacher_split],
                    'student_train': indices[teacher_split:teacher_split+student_split],
                    'dev': indices[teacher_split+student_split:teacher_split+student_split+dev_split],
                    'test': indices[teacher_split+student_split+dev_split:]
                }
        dataset_split = torch.utils.data.Subset(dataset, splits[configs['split']])

        data_loader = torch.utils.data.DataLoader(dataset=dataset_split,
                                                  batch_size = batch_size,
                                                  shuffle = shuffle,
                                                  num_workers = 0,
                                                  )
        return data_loader

    else:
        raise NotImplementedError


