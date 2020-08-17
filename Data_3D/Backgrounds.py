import numpy as np
import torch

import torch.nn.functional as F
from torch import nn
from torch import optim

def list_system_randomized(list_array,image_size,max_val,complexity):
    
    # Function: list_system_randomized
    # Inputs:   list_array (list)
    #           image_size (int)
    #           max_val    (float)
    #           complexity (int)
    # Process: recursion process to generate random backgrounds using L-Systems
    # Output:   list_array (list)

    new_array = []
    
    if complexity == 1:
        
        length_1 = np.random.randint(10,500)
        length_2 = np.random.randint(10,500)
    
        if len(list_array) < image_size**2:
            for n in range(len(list_array)):

                if list_array[n] == 0:
                    for k1 in range(length_1):
                        new_array.append(max_val)
                        new_array.append(0)
                if list_array[n] == max_val:
                    for k1 in range(length_2):
                        new_array.append(0)
                        new_array.append(max_val)

            list_array = list_system_randomized(new_array,image_size,max_val,complexity)
    
    if complexity == 2:
        
        length_1 = np.random.randint(10,500)
        depth_1_length_1 = np.random.randint(10,50)
        depth_1_length_2 = np.random.randint(10,50)
        length_2 = np.random.randint(10,500)
        depth_2_length_1 = np.random.randint(10,50)
        depth_2_length_2 = np.random.randint(10,50)

        if len(list_array) < image_size**2:
            for n in range(len(list_array)):

                if list_array[n] == 0:
                    for k1 in range(length_1):
                        for k2 in range(depth_1_length_1):
                            new_array.append(max_val)
                        for k2 in range(depth_1_length_2):
                            new_array.append(0)
                if list_array[n] == max_val:
                    for k in range(length_2):
                        for k2 in range(depth_2_length_1):
                            new_array.append(0)
                        for k2 in range(depth_2_length_2):
                            new_array.append(max_val)
                            
            list_array = list_system_randomized(new_array,image_size,max_val,complexity)
    
    return list_array

def returnExample(data_loader):

    # Function: returnExample
    # Inputs:   data_loader (pytorch dataloader)
    # Process: returns example of image
    # Output:   image_batch (pytorch Tensor)
    #           label_batch (pytorch Tensor)
    
    for image_batch, label_batch in data_loader:

        break
        
    if len(label_batch) > 1:
        r = np.random.randint(len(label_batch))
        image_batch = image_batch[r,:,:,:]
        label_batch = label_batch[r]

    return image_batch, label_batch

def FGSM(image_batch,label_batch,model,device):

    # Function: FGSM
    # Inputs:   image_batch (pytorch image), size=(1,features,image_size,image_size)
    #           label_batch (pytorch Tensor), size=(1,num_labels)
    #           model (pytorch network)
    # Process: generates backgrounds using FGSM
    # Output:   image_batch.grad.data (pytorch Tensor)
    
    loss_function = nn.BCEWithLogitsLoss()

    image_batch.requires_grad = True
    model.eval()

    optimizer = optim.Adam(model.parameters())

    model.zero_grad()

    output = model(image_batch.unsqueeze(0).to(device))
    loss   = loss_function(output,label_batch.to(device))

    loss.backward()

    return image_batch.grad.data

def applyBackground(object_image,background_image):
    
    # Function: applyBackground
    # Inputs:   object_image (pytorch image), size=(1,features,image_size,image_size)
    #           background_image (pytorch Tensor), size=(1,features,image_size,image_size)
    # Process: places background for given object image
    # Output:   object_image (pytorch Tensor), size=(1,features,image_size,image_size)
    
    object_image[np.where(object_image==0.0)] = background_image[np.where(object_image==0.0)]
    
    return object_image


    