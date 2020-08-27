import torch
import numpy as np
import math
import torchvision
from torch import nn
import os

from torch.utils.data import TensorDataset, DataLoader

import OnlineShiftDetectors
import DataSetManipulator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_up_data():
    
    # Function: set_up_data
    # Inputs:   none
    # Process:  returns datasets for testing
    # Output:   cifar10_trainloader (pytorch dataloader)
    #           cifar10_testloader (pytorch dataloader)
    
    n_epochs   = 150
    batch_size = int(1e2)
    lr         = 0.01

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    cifar10_trainset = torchvision.datasets.CIFAR10('/home/fmejia/fmejia/Cypercat/cyphercat/datasets//', train=True, transform=transform, download=True)
    cifar10_trainloader = torch.utils.data.DataLoader(cifar10_trainset, batch_size=batch_size, shuffle=True, num_workers=16)

    cifar10_testset = torchvision.datasets.CIFAR10('/home/fmejia/fmejia/Cypercat/cyphercat//datasets//', train=False, transform=transform, download=True)
    cifar10_testloader = torch.utils.data.DataLoader(cifar10_testset, batch_size=batch_size, shuffle=True, num_workers=16)
    
    return cifar10_trainloader,cifar10_testloader

def run_test(classifier_detector,batch_size,data_loader,iterations):
    
    # Function: run_test
    # Inputs:   iteration (int)
    #           batch_size (int)
    #           data_loader (pytorch dataloader)
    #           iterations (int)
    # Process:  runs test for classifier covariate shift
    # Output:   g (torch tensor)
    
    g = torch.zeros([iterations,batch_size*len(data_loader)])

    classifier_detector.x = torch.zeros([classifier_detector.class_size,1])
    classifier_detector.y = torch.zeros([classifier_detector.class_size,1])

    for epoch in range(iterations):

        k = 0
        
        for i, batch in enumerate(data_loader, 0):

            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)

            model_output = model(imgs.to(device))

            for ii in range(batch_size):

                _,g[epoch,k]  = classifier_detector.shift_filter(model_output[ii,:].unsqueeze(0).cpu(),1)

                k += 1

            if k >= (batch_size*len(data_loader))-1:
                break

    return g

if __name__ == '__main__':
    
    cifar10_trainloader,cifar10_testloader = set_up_data()

    model = torchvision.models.resnet18(pretrained=True)

    model.fc = nn.Linear(512,10)
    model    = nn.Sequential(model,
                             nn.Dropout(),
                            )
    model.to(device)

    model.load_state_dict(torch.load('CIFAR_10_Classifier.pt'))
    model.eval()

    batch_size = int(1e2)
    iterations = 100
    percentages = [ii/10 for ii in range(10)]
    
    classifier_detector = OnlineShiftDetectors.ClassifierDetector(0.01,10)
    classifier_detector.set_label_distribution((1/10)*torch.ones([10]))

    stats_path = '/home/jgornet/Covariate_Testing/Classifier_Statistics'
    
    images = torch.zeros([int(1e4),3,32,32])
    labels = torch.zeros([int(1e4)]).long()

    k = 0

    for i, batch in enumerate(cifar10_testloader, 0):

        imgs, lab = batch

        for ii in range(100):

            images[k,:,:,:] = imgs[ii,:,:,:]
            labels[k]       = lab[ii]

            k += 1
        
    for p in range(len(percentages)):
        
        shifted_images,shifted_labels = DataSetManipulator.shiftLabelDistribution(images,labels,3,percentages[p])

        shifted_dataset = TensorDataset(shifted_images,shifted_labels)
        shifted_loader  = DataLoader(shifted_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,drop_last=True)

        g = run_test(classifier_detector,batch_size,shifted_loader,iterations)
    
        save_file = os.path.join(stats_path,'classifier_percent_' + str(p) + '.npy')
    
        np.save(save_file,g)
    