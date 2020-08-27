import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torch.utils.data import TensorDataset, DataLoader
from torch import optim 

import math 

import os

def getAndSetDevice():
    
    # Function: getAndSetDevice
    # Inputs:   none
    # Process: creates pytorch device
    # Output:   device (pytorch device)
    
    device = torch.device("cuda:1")

    torch.cuda.set_device(device)

    return device

def getDetector(save_path):
    
    # Function: getDetector
    # Inputs:   none
    # Process: creates pytorch model for detection
    # Output:   detector (pytorch model)
            
    if os.path.exists(save_path):
        detector.load_state_dict(torch.load(save_path))
        detector.train()
    else:
        detector = math.nan

    return detector

def loadCifar10Data(batch_size):
    
    # Function: loadData
    # Inputs:   none
    # Process: returns Cifar10 dataset
    # Output:   training_loader (pytorch data loader)
    #           testing_loader (pytorch data loader)
    #           len(training_labels) (int)
    #           len(testing_labels) (int)
    
    training_images, training_labels = tfds.as_numpy(tfds.load(
                                            'cifar10',
                                            split='train', 
                                            batch_size=-1, 
                                            as_supervised=True,
                                        ))

    testing_images, testing_labels = tfds.as_numpy(tfds.load(
                                            'cifar10',
                                            split='test', 
                                            batch_size=-1, 
                                            as_supervised=True,
                                        ))

    training_images_pytorch = torch.Tensor(training_images).transpose(1,3)
    training_labels_pytorch = torch.Tensor(training_labels).type(torch.LongTensor)

    testing_images_pytorch = torch.Tensor(testing_images).transpose(1,3)
    testing_labels_pytorch = torch.Tensor(testing_labels).type(torch.LongTensor)

    training_dataset = TensorDataset(training_images_pytorch,training_labels_pytorch)
    testing_dataset  = TensorDataset(testing_images_pytorch,testing_labels_pytorch)

    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,drop_last=True)
    testing_loader  = DataLoader(testing_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,drop_last=True)

    return training_loader, testing_loader, len(training_labels), len(testing_labels)

def trainNetwork(model,device,training_loader,testing_loader,num_training_examples,num_testing_examples,numIter,save_path):

    # Function: trainNetwork
    # Inputs:   model (pytorch model) 
    #           device (pytorch device)
    #           training_loader (pytorch data loader)
    #           testing_loader (pytorch data loader)
    #           num_training_examples (int)
    #           num_testing_examples (int)
    #           numIter (int)
    #           save_path (file path for saving)
    # Process: trains pytorch model, saves model every 10 epochs
    # Output:   none

    model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer     = optim.Adam(model.parameters(),lr=0.0001)

    loss_array     = np.zeros([numIter,2])
    accuracy_array = np.zeros([numIter,2])

    for epoch in range(numIter):

        training_loss_array = []
        testing_loss_array  = []

        training_correct = 0
        testing_correct  = 0

        model.train()
        
        for image_batch, label_batch in training_loader:

            optimizer.zero_grad()
            
            output            = model(image_batch.to(device))
            
            training_loss = loss_function(output,label_batch.to(device))
            
            training_loss.backward()
            optimizer.step()

            training_loss_array.append(training_loss.cpu().item())
            
            _, predicted      = torch.max(output.data, 1)
            training_correct += (predicted.cpu() == label_batch).sum().item()
            
        model.eval()
        
        for image_batch, label_batch in testing_loader:
            
            output            = model(image_batch.to(device))
            
            testing_loss = loss_function(model(image_batch.to(device)),label_batch.to(device))
            
            testing_loss_array.append(testing_loss.cpu().item())

            _, predicted     = torch.max(output.data, 1)
            testing_correct += (predicted.cpu() == label_batch).sum().item()
            
        print('Epoch: ' + str(epoch))
        print()
        print("Training Loss: " + '\t\t' + str(np.mean(training_loss_array)))
        print("Testing Loss: " + '\t\t' + str(np.mean(testing_loss_array)))
        print("Training Accuracy: " + '\t' + str(100.0*training_correct/num_training_examples) + '%')
        print("Testing Accuracy: " + '\t' + str(100.0*testing_correct/num_testing_examples) + '%')

        loss_array[epoch,0] = np.mean(training_loss_array)
        loss_array[epoch,1] = np.mean(testing_loss_array)
        
        accuracy_array[epoch,0] = np.mean(100.0*training_correct/num_training_examples)
        accuracy_array[epoch,1] = np.mean(100.0*testing_correct/num_testing_examples)
        
        print()
    
    if epoch % 10 == 0:
        torch.save(detector.state_dict(),save_path)