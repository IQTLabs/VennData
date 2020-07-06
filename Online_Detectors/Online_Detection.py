import numpy as np
import torch 
import math 

class Detector():
    
    def __init__(self,epsilon,model,device,class_size):
        
        self.set_epsilon(epsilon)
        self.set_model(model)
        self.set_device(device)
        self.set_class_size(class_size)
        
        self.set_lambda_values(math.nan)
                
    #getters and setters
    def get_epsilon(self):
        
        return self.epsilon
    
    def set_epsilon(self,epsilon):
        
        self.epsilon = epsilon
        
    def get_model(self):
        
        return self.model
    
    def set_model(self,model):
        
        self.model = model
        
    def get_device(self):
        
        return self.device
    
    def set_device(self,device):
        
        self.device = device
        
    def get_class_size(self):
        
        return self.class_size
    
    def set_class_size(self,class_size):
        
        self.class_size = class_size
        
    def get_lambda_values(self):
        
        return self.lambda_values
    
    def set_lambda_values(self,lambda_values):
        
        self.lambda_values = lambda_values
    
    def set_detector_lambda_values(self,data_loader):
    
        # Function: set_detector_lambda_values
        # Inputs:   data_loader  (Pytorch dataloader)
        # Process: sets the lambda values for the detector
        # Used Functions: calculate_lambda
        # Output:   none
        
        x = np.zeros([10,len(data_loader)])
        n = 0

        percent_done = 10

        model_labels = np.zeros([len(data_loader)])
            
        class_size = self.get_class_size()
        model      = self.get_model()
        device     = self.get_device()
        
        model.eval()
        
        for image_batch, _ in data_loader:

            x[:,n] = model(image_batch.to(device)).cpu().data.numpy().squeeze()
            model_labels[n] = x[:,n].argmax()
            n += 1

            if n % (len(data_loader)/10) == 0:
                print(str(percent_done) + '% Percent Done')
                percent_done += 10

        lambda_values_model = self.calculate_lambda(model_labels,torch.Tensor(np.arange(0,10)).int())
        
        self.set_lambda_values(lambda_values_model)
    
    def calculate_lambda(self,labels,label_vals):

        # Function: calculate_lambda
        # Inputs:   labels        (Pytorch Tensor) size=(batch)
        #           label_vals    (Pytorch Tensor) size=(class_size)
        # Process: finds the frequency a label shows up in the data set
        # Output:   lambda_vals   (numpy array)    size=(class_size)

        lambda_values = np.zeros([len(label_vals)])

        for val in range(len(label_vals)):

            lambda_values[val] = (val == labels).sum().item()/len(labels)
                
        return lambda_values

    def LambdaPredictionTransition(self,x):

        # Function: LambdaPredictionTransition
        # Inputs:   x           (numpy array) size=(class_size)
        #           lambda_vals (numpy array)    size=(class_size)
        # Process: iterates next process
        # Output:   x           (numpy array) size=(class_size)

        lambda_values = self.get_lambda_values()
        
        x = x + lambda_values
        return x

    def LambdaFilterTransition(self,x,y,epsilon):

        # Function: LambdaFilterTransition
        # Inputs:   x           (numpy array) size=(class_size)
        #           y           (numpy array) size=(class_size)
        #           epsilon     (float) 
        # Process: adjusts process error
        # Output:   x           (numpy array) size=(class_size)

        epsilon = self.get_epsilon()
        
        x = x + epsilon*(y-x)
        return x

    def LambdaObservation(self,y,label_pred):

        # Function: LambdaObservation
        # Inputs:   y           (numpy array) size=(class_size)
        #           label_pred  (Pytorch Tensor) size=(class_size)
        # Process: observation equation for lambda observation
        # Used Functions: calculate_lambda
        # Output:   y           (numpy array) size=(class_size)

        class_size = self.get_class_size()

        observation_lambda_values = self.calculate_lambda(label_pred,torch.Tensor(np.arange(0,class_size)).int())
        y = y + observation_lambda_values
        return y

    def Residual(x,y):

        # Function: LambdaObservation
        # Inputs:   x           (numpy array) size=(class_size)
        #           y           (numpy array) size=(class_size)
        # Process: finds error between true label frequency and predicted label frequency
        # Output:   r           (float) 

        r = np.linalg.norm(x-y)
        return r

    def runDetector(self,data_loader,numIter):

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
        #           accuracy    (float) 

        k = 0

        x = np.zeros([len(lambda_values),numIter])
        y = np.zeros([len(lambda_values),numIter])
        r = np.zeros([numIter])

        percent_done  = 10
        correct_array = np.zeros(numIter)
        
        lambda_values = self.get_lambda_values()
        epsilon       = self.get_epsilon()
        device        = self.get_device()
        
        for image_batch, label_batch in data_loader:

            model.eval()

            if len(image_batch.size()) == 4:
                output       = self.model(image_batch.to(device))
            elif len(image_batch.size()) == 3:
                output       = self.model(image_batch.unsqueeze(0).to(device))

            _, predicted = torch.max(output.data, 1)

            x[:,k+1] = self.LambdaPredictionTransition(x[:,k],lambda_values) #(1)
            y[:,k+1] = self.LambdaObservation(y[:,k],predicted)              #(2)
            r[k]     = self.Residual(x[:,k+1],y[:,k+1])                      #(3)
            x[:,k+1] = self.LambdaFilterTransition(x[:,k+1],y[:,k+1],epsilon)#(4)

            k += 1

            correct_array[k] = (predicted.cpu() == label_batch).sum().item()

            if k % (numIter/10) == 0:
                print(str(percent_done) + '% Percent Done')
                percent_done += 10

            if k-1 == numIter:
                break

        accuracy = np.mean(correct_array)

        return r,correct_array,accuracy
    
    