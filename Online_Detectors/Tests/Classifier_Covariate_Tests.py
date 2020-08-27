import torch
import numpy as np
import math
import torchvision
from torch import nn
import os

import OnlineShiftDetectors

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_up_data():
    
    # Function: set_up_data
    # Inputs:   none
    # Process:  returns datasets for testing
    # Output:   cifar10_trainloader,cifar10_testloader,gen_testloader,flip_loader,zoom_loader,bright_loader,translate_loader (pytorch dataloader)
    
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

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(256),
        torchvision.transforms.Resize(32),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    gen_testloader = torch.load('/home/jgornet/Generative_Models/Covariate_Measurement_Models/altcifar_dataloader.pth')

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomHorizontalFlip(p=1.0),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    flip_testset = torchvision.datasets.CIFAR10('/home/fmejia/fmejia/Cypercat/cyphercat//datasets//', train=False, transform=transform, download=True)
    flip_loader = torch.utils.data.DataLoader(flip_testset, batch_size=batch_size, shuffle=True, num_workers=16)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomAffine(0, translate=None, scale=(1.3,1.5), shear=None, resample=False, fillcolor=0),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    zoom_testset = torchvision.datasets.CIFAR10('/home/fmejia/fmejia/Cypercat/cyphercat//datasets//', train=False, transform=transform, download=True)
    zoom_loader = torch.utils.data.DataLoader(zoom_testset, batch_size=batch_size, shuffle=True, num_workers=16)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.ColorJitter(brightness=1.3, contrast=0, saturation=0, hue=0),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    bright_testset = torchvision.datasets.CIFAR10('/home/fmejia/fmejia/Cypercat/cyphercat//datasets//', train=False, transform=transform, download=True)
    bright_loader = torch.utils.data.DataLoader(bright_testset, batch_size=batch_size, shuffle=True, num_workers=16)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomAffine(0, translate=(0.3,0.3), scale=(1.3,1.3), shear=None, resample=False, fillcolor=0),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    translate_testset = torchvision.datasets.CIFAR10('/home/fmejia/fmejia/Cypercat/cyphercat//datasets//', train=False, transform=transform, download=True)
    translate_loader = torch.utils.data.DataLoader(translate_testset, batch_size=batch_size, shuffle=True, num_workers=16)


    return cifar10_trainloader,cifar10_testloader,gen_testloader,flip_loader,zoom_loader,bright_loader,translate_loader

def run_test(classifier_detector,batch_size,data_loader,iterations):
    
    # Function: run_test
    # Inputs:   classifier_detector (pytorch model)
    #           batch_size (int)
    #           data_loader (pytorch dataloader)
    #           iterations (int)
    # Process:  runs test for classifier covariate detector
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

def run_noise_test(classifier_detector,batch_size,data_loader,iterations):
    
    # Function: run_noise_test
    # Inputs:   classifier_detector (pytorch model)
    #           batch_size (int)
    #           data_loader (pytorch dataloader)
    #           iterations (int)
    # Process:  runs noise input test for classifier covariate detector
    # Output:   g (torch tensor)
    
    g = torch.zeros([iterations,batch_size*len(data_loader)])

    classifier_detector.x = torch.zeros([classifier_detector.class_size,1])
    classifier_detector.y = torch.zeros([classifier_detector.class_size,1])

    for epoch in range(iterations):

        k = 0
        
        for i, batch in enumerate(data_loader, 0):

            imgs, labels = batch
            imgs = imgs + 0.4*torch.randn(imgs.size())
            imgs, labels = imgs.to(device), labels.to(device)

            model_output = model(imgs.to(device))

            for ii in range(batch_size):

                _,g[epoch,k]  = classifier_detector.shift_filter(model_output[ii,:].unsqueeze(0).cpu(),1)

                k += 1

            if k >= (batch_size*len(data_loader))-1:
                break

    return g

if __name__ == '__main__':
    
    cifar10_trainloader,cifar10_testloader,gen_testloader,flip_loader,zoom_loader,bright_loader,translate_loader = set_up_data()

    model = torchvision.models.resnet18(pretrained=True)

    model.fc = nn.Linear(512,10)
    model    = nn.Sequential(model,
                             nn.Dropout(),
                            )
    model.to(device)

    model.load_state_dict(torch.load('CIFAR_10_Classifier.pt'))
    model.eval()

    batch_size = int(1e2)
    
    classifier_detector = OnlineShiftDetectors.ClassifierDetector(0.01,10)

    classifier_detector.set_label_pred_distribution(model,cifar10_trainloader,device)

    iterations  = 100
    
    g_train     = run_test(classifier_detector,batch_size,cifar10_trainloader,int(iterations/5))
    g_test      = run_test(classifier_detector,batch_size,cifar10_testloader,iterations)
    g_gen       = run_test(classifier_detector,batch_size,gen_testloader,iterations)
    g_flip      = run_test(classifier_detector,batch_size,flip_loader,iterations)
    g_zoom      = run_test(classifier_detector,batch_size,zoom_loader,iterations)
    g_bright    = run_test(classifier_detector,batch_size,bright_loader,iterations)
    g_translate = run_test(classifier_detector,batch_size,translate_loader,iterations)

    g_noise     = run_noise_test(classifier_detector,batch_size,cifar10_testloader,iterations)

    stats_path = '/home/jgornet/Covariate_Testing/Classifier_Statistics'
        
    train_file     = os.path.join(stats_path,'classifier_train.npy')
    test_file      = os.path.join(stats_path,'classifier_test.npy')
    gen_file       = os.path.join(stats_path,'classifier_gen.npy')
    flip_file      = os.path.join(stats_path,'classifier_flip.npy')
    zoom_file      = os.path.join(stats_path,'classifier_zoom.npy')
    bright_file    = os.path.join(stats_path,'classifier_bright.npy')
    translate_file = os.path.join(stats_path,'classifier_translate.npy')
    
    noise_file = os.path.join(stats_path,'classifier_noise.npy')

    np.save(train_file,g_train)
    np.save(test_file,g_test)
    np.save(gen_file,g_gen)
    np.save(flip_file,g_flip)
    np.save(zoom_file,g_zoom)
    np.save(bright_file,g_bright)
    np.save(translate_file,g_translate)
    np.save(noise_file,g_noise)
    
    