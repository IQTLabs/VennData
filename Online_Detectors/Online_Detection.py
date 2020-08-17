import numpy as np
import torch 
import math 

from scipy.stats import chi2

class Detector():
    
    def __init__(self,epsilon,model,device,class_size):
        
        self.set_epsilon(epsilon)
        self.set_model(model)
        self.set_device(device)
        self.set_class_size(class_size)
        
        self.set_lambda_values(math.nan)
                
        self.x = 0
        self.y = 0
        
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
    
    def set_covariance(self):
        
        eps           = self.get_epsilon()
        lambda_values = self.get_lambda_values()
        class_size    = self.get_class_size()
        
        self.P = (((1-eps)**2)/(1-(1-eps)**2))*(lambda_values-lambda_values**2)*np.identity(class_size)
       
    def get_covariance(self):
        
        return self.P
    
    def set_threshold(self,false_alarm_rate):
        
        class_size = self.get_class_size()
        
        bins = np.linspace(0,5*class_size,int(1e4))
        
        p_g  = chi2.pdf(bins,class_size)
        P_g  = 1-np.cumsum(p_g)/np.sum(p_g)
        
        self.thresh = bins[np.where(np.abs(P_g-false_alarm_rate)==np.abs(P_g-false_alarm_rate).min())]
        
    def get_threshold(self):
        
        return self.thresh
       
    def calculate_dataset_lambda_values(self,data_loader):
    
        # Function: calculate_dataset_lambda_values
        # Inputs:   data_loader  (Pytorch dataloader)
        # Process: calculates the lambda values for the dataset
        # Used Functions: calculate_lambda
        # Output:   lambda (array)
        
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
        
        return lambda_values_model
    
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

    def LambdaFilterTransition(self,x,y):

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

    def Residual(self,x,y):

        # Function: LambdaObservation
        # Inputs:   x           (numpy array) size=(class_size)
        #           y           (numpy array) size=(class_size)
        # Process: finds error between true label frequency and predicted label frequency
        # Output:   r           (float) 

        P = self.get_covariance()
        r = np.matmul(np.matmul((x-y),np.linalg.inv(P)),(x-y))
        
        return r
    
    def ResidualClasses(self,x,y):

        # Function: LambdaObservation
        # Inputs:   x           (numpy array) size=(class_size)
        #           y           (numpy array) size=(class_size)
        # Process: finds error between true label frequency and predicted label frequency
        # Output:   r           (float) 

        P = self.get_covariance()
        r = np.matmul(np.linalg.inv(P),np.power(x-y,2))
        
        return r

    def runDetector(self,data_loader,numIter,numEpoch):

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
        #           correct_array (array)
        #           accuracy    (float) 

        lambda_values = self.get_lambda_values()
        device        = self.get_device()
        model         = self.get_model()
        
        accuracy = []
        
        x = np.zeros([len(lambda_values)])
        y = np.zeros([len(lambda_values)])
        r = []

        for epoch in range(numEpoch):
            
            percent_done  = 10
            correct_array = np.zeros(numIter)
            
            k = 0
            
            for image_batch, label_batch in data_loader:

                model.eval()

                if len(image_batch.size()) == 4:
                    output       = model(image_batch.to(device))
                elif len(image_batch.size()) == 3:
                    output       = model(image_batch.unsqueeze(0).to(device))

                _, predicted = torch.max(output.data, 1)

                x = self.LambdaPredictionTransition(x)       #(1)
                y = self.LambdaObservation(y,predicted)      #(2)
                
                r.append(self.Residual(x,y))                 #(3)
                
                x = self.LambdaFilterTransition(x,y)         #(4)
                
                k += 1

                correct_array[k] = (predicted.cpu() == label_batch).sum().item()

                if k % (numIter/10) == 0:
                    print(str(percent_done) + '% Percent Done')
                    percent_done += 10

                if k == numIter-1:
                    break
                    
            accuracy.append(np.mean(correct_array))

        return r,correct_array,accuracy
    
    def analyzeSignal(self,x,y,input_img):

        # Function: analyzeSignal
        # Inputs:   x         (Pytorch Neural Network) 
        #           y        (Pytorch Device)
        #           input_img   (Pytorch Tensor)    size=(1,features,height,width)
        # Process:  tests if there is a shift in label distribution
        # Used Functions:  LambdaPredictionTransition (1)
        #                  LambdaObservation          (2)
        #                  Residual                   (3)
        #                  LambdaFilterTransition     (4)
        # Output:   xPredict   (float) 
        #           xFilter    (float) 
        #           y          (float) 
        #           r          (float) 
        #           detection  (boolean) 
        #           label_detection  (int array) 

        lambda_values = self.get_lambda_values()
        device        = self.get_device()
        model         = self.get_model()
        thresh        = self.get_threshold()
        
        model.eval()

        if len(input_img.size()) == 4:
            output = model(input_img.to(device))
        elif len(input_img.size()) == 3:
            output = model(input_img.unsqueeze(0).to(device))

        _, predicted = torch.max(output.data, 1)

        xPredict = self.LambdaPredictionTransition(x)       #(1)
        y        = self.LambdaObservation(y,predicted)      #(2)

        g        = self.Residual(xPredict,y)                #(3)
        r        = self.ResidualClasses(xPredict,y)         #(3)
        
        if g > thresh:
            detection = True
        else:
            detection = False
            
        label_detections = np.array(r > 1,dtype=int)
        
        xFilter  = self.LambdaFilterTransition(xPredict,y)  #(4)

        return xPredict,xFilter,y,g,r,detection,label_detections
    
    