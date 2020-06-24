import numpy as np
import torch 

def calculate_lambda(labels,label_vals):
    
    # Function: calculate_lambda
    # Inputs:   labels        (Pytorch Tensor) size=(batch)
    #           label_vals    (Pytorch Tensor) size=(class_size)
    # Process: finds the frequency a label shows up in the data set
    # Output:   lambda_vals   (numpy array)    size=(class_size)
    
    lambda_vals = np.zeros([len(label_vals)])
    
    for val in range(len(label_vals)):
        
        lambda_vals[val] = (val == labels).sum().item()/len(labels)
        
    return lambda_vals

def LambdaPredictionTransition(x,lambda_vals):
    
    # Function: LambdaPredictionTransition
    # Inputs:   x           (numpy array) size=(class_size)
    #           lambda_vals (numpy array)    size=(class_size)
    # Process: iterates next process
    # Output:   x           (numpy array) size=(class_size)
    
    x = x + lambda_vals
    return x

def LambdaFilterTransition(x,y,epsilon):
    
    # Function: LambdaFilterTransition
    # Inputs:   x           (numpy array) size=(class_size)
    #           y           (numpy array) size=(class_size)
    #           epsilon     (float) 
    # Process: adjusts process error
    # Output:   x           (numpy array) size=(class_size)
    
    x = x + epsilon*(y-x)
    return x

def LambdaObservation(y,label_pred):
    
    # Function: LambdaObservation
    # Inputs:   y           (numpy array) size=(class_size)
    #           label_pred  (Pytorch Tensor) size=(class_size)
    # Process: observation equation for lambda observation
    # Used Functions: calculate_lambda
    # Output:   y           (numpy array) size=(class_size)
    
    lambda_vals = calculate_lambda(label_pred,torch.Tensor(np.arange(0,10)).int())
    y = y + lambda_vals
    return y
    
def Residual(x,y):
    
    # Function: LambdaObservation
    # Inputs:   x           (numpy array) size=(class_size)
    #           y           (numpy array) size=(class_size)
    # Process: finds error between true label frequency and predicted label frequency
    # Output:   r           (float) 
    
    r = np.linalg.norm(x-y)
    return r

def runDetector(model,device,lambda_values,data_loader,epsilon):
    
    # Function: runDetector
    # Inputs:   model         (Pytorch Neural Network) 
    #           device        (Pytorch Device)
    #           lambda_vals   (numpy array)    size=(class_size)
    #           data_loader   (Pytorch Data Loader)
    #           epsilon       (float)
    # Process:  tests if there is a shift in label distribution
    # Used Functions:  LambdaPredictionTransition (1)
    #                  LambdaObservation          (2)
    #                  Residual                   (3)
    #                  LambdaFilterTransition     (4)
    # Output:   r           (float) 
    
    k = 0

    x = np.zeros([len(lambda_values),len(data_loader)])
    y = np.zeros([len(lambda_values),len(data_loader)])
    r = np.zeros([len(data_loader)])

    for image_batch, label_batch in data_loader:

        model.eval()

        output           = model(image_batch.to(device))

        _, predicted     = torch.max(output.data, 1)

        x[:,k+1] = LambdaPredictionTransition(x[:,k],lambda_values) #(1)
        y[:,k+1] = LambdaObservation(y[:,k],predicted)              #(2)
        r[k]     = Residual(x[:,k+1],y[:,k+1])                      #(3)
        x[:,k+1] = LambdaFilterTransition(x[:,k+1],y[:,k+1],epsilon)#(4)

        k += 1

        if k+1 == len(data_loader):
            break

    return r
    
    