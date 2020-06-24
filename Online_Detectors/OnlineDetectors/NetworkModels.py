import torch 

from torch import nn
from torch import optim

import numpy as np

def TrainNetwork(model,device,training_loader,testing_loader,numIter):

    # Function: TrainNetwork
    # Inputs:   model             (Pytorch Neural Network) 
    #           device            (Pytorch Device)
    #           training_loader   (Pytorch Data Loader)
    #           testing_loader    (Pytorch Data Loader)
    #           numIter           (int)
    # Process: trains neural network
    # Output:   loss_array        (numpy array)    size=(numIter,2) 
    #           accuracy_array    (numpy array)    size=(numIter,2) 
    
    model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer     = optim.Adam(model.parameters())

    loss_array     = np.zeros([numIter,2])
    accuracy_array = np.zeros([numIter,2])
    
    for epoch in range(numIter):

        training_loss_array = []
        testing_loss_array  = []

        training_correct = 0
        testing_correct  = 0
        
        training_num   = 0
        testing_num    = 0

        for image_batch, label_batch in training_loader:

            model.train()

            optimizer.zero_grad()
            
            output            = model(image_batch.to(device))
            
            training_loss = loss_function(output,label_batch.to(device))
            
            training_loss.backward()
            optimizer.step()

            training_loss_array.append(training_loss.cpu().item())
            
            _, predicted      = torch.max(output.data, 1)
            training_correct += (predicted.cpu() == label_batch).sum().item()
            
            training_num += len(label_batch)
            
        for image_batch, label_batch in testing_loader:

            model.eval()
            
            output            = model(image_batch.to(device))
            
            testing_loss = loss_function(model(image_batch.to(device)),label_batch.to(device))
            
            testing_loss_array.append(testing_loss.cpu().item())

            _, predicted     = torch.max(output.data, 1)
            testing_correct += (predicted.cpu() == label_batch).sum().item()
            
            testing_num += len(label_batch)
            
        print('Epoch: ' + str(epoch))
        print()
        print("Training Loss: " + '\t\t' + str(np.mean(training_loss_array)))
        print("Testing Loss: " + '\t\t' + str(np.mean(testing_loss_array)))
        print("Training Accuracy: " + '\t' + str(100.0*training_correct/training_num) + '%')
        print("Testing Accuracy: " + '\t' + str(100.0*testing_correct/testing_num) + '%')

        loss_array[epoch,0] = np.mean(training_loss_array)
        loss_array[epoch,1] = np.mean(testing_loss_array)
        
        accuracy_array[epoch,0] = np.mean(100.0*training_correct/training_num)
        accuracy_array[epoch,1] = np.mean(100.0*testing_correct/testing_num)
        
        print()
        
    return loss_array, accuracy_array
