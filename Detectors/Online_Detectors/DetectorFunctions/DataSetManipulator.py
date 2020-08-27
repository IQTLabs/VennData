import torch
import numpy as np

def shiftLabelDistribution(images,labels,label_remove,percentage):
    
    # Function: shiftLabelDistribution
    # Inputs:   images        (Pytorch Tensor) size=(batch,features,height,width)
    #           labels        (Pytorch Tensor) size=(batch)
    #           label_remove  (int)
    #           percentage    (float)
    # Process: sets a label to have a specific percentage amount of the data set
    # Output:   shifted_images (Pytorch Tensor) size=(batch,features,height,width)
    #           shifted_labels (Pytorch Tensor) size=(batch)
    
    num_examples = len(np.where(labels!=label_remove)[0])+int(len(np.where(labels==label_remove)[0])*percentage)

    shifted_images = torch.zeros([num_examples,3,32,32])
    shifted_labels = torch.zeros([num_examples])

    shifted_images[0:len(np.where(labels!=label_remove)[0]),:,:,:] = images[np.where(labels!=label_remove)[0],:,:,:]
    shifted_labels[0:len(np.where(labels!=label_remove)[0])]       = labels[np.where(labels!=label_remove)[0]]

    r = np.random.randint(len(np.where(labels==label_remove)[0]),size=int(len(np.where(labels==label_remove)[0])*percentage))

    shifted_images[len(np.where(labels!=label_remove)[0]):,:,:,:] = images[np.where(labels==label_remove)[0][r],:,:,:]
    shifted_labels[len(np.where(labels!=label_remove)[0]):]       = labels[np.where(labels==label_remove)[0][r]]

    return shifted_images,shifted_labels