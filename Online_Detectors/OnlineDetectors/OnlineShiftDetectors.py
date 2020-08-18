import torch
import numpy as np
import math

class ClassifierDetector():
    
    def __init__(self,epsilon,class_size):
        
        self.epsilon    = epsilon
        self.class_size = class_size
            
        self.x = torch.zeros([class_size,1])
        self.y = torch.zeros([class_size,1])
                            
    def set_label_distribution(self,label_distribution):
        
        # Function: set_label_distribution
        # Inputs:   label_distribution  (Pytorch tensor) (class_size)
        # Process:  sets the label distribution from the dataset
        # Output:   none
        
        self.label_distribution = label_distribution.unsqueeze(1)
        self.calculate_covariance()
        
    def set_label_pred_distribution(self,model,data_loader,device):
        
        # Function: set_label_distribution
        # Inputs:   label_distribution  (Pytorch tensor) (class_size)
        # Process:  sets the model predicted label distribution from the dataset
        # Used Functions: calculate_label_pred_distribution
        # Output:   none
        
        self.label_distribution = self.calculate_label_pred_distribution(model,data_loader,device).unsqueeze(1)
        self.calculate_covariance()
        
    def calculate_label_pred_distribution(self,model,data_loader,device):
        
        # Function: calculate_label_pred_distribution
        # Inputs:   model  (Pytorch model)
        #           data_loader  (Pytorch dataloader)
        #           device (Pytorch device)
        # Process:  finds the predicted label distribution of the model
        # Output:   pred_label_distribution (Pytorch tensor) (class_size)
        
        k            = 0
        percent_done = 10

        label_prediction_list = np.array([])
        class_size            = self.class_size
        
        model.eval()
       
        for image_batch, _ in data_loader:
            
            output            = model(image_batch.to(device))
            label_predictions = output.argmax(1)
            
            label_prediction_list = np.append(label_prediction_list,label_predictions.cpu().numpy())
            
            k += 1

            if k % (len(data_loader)/10) == 0:
                print(str(percent_done) + '% Percent Done')
                percent_done += 10

        label_prediction_list   = torch.Tensor(label_prediction_list).unsqueeze(0)
        pred_label_distribution = (label_prediction_list==torch.Tensor(list(range(class_size))).unsqueeze(1)).float().sum(1)/float(k*len(image_batch))
                
        return pred_label_distribution
    
    def calculate_covariance(self):
        
        # Function: calculate_covariance
        # Inputs:   none
        # Process:  finds the error covariance of the filter
        # Output:   none
        
        epsilon            = self.epsilon
        label_distribution = self.label_distribution
        class_size         = self.class_size
        
        self.P = (((1-epsilon)**2)/(1-(1-epsilon)**2))*(label_distribution-label_distribution**2)*torch.eye(class_size)
        
    def shift_filter(self,model_output,batch_size):

        # Function: calculate_covariance
        # Inputs:   model_output (pytorch Tensor) (batch_size x class_size)
        #           batch_size   (int)
        # Process:  calculates the detection signal using filters
        # Output:   r (pytorch Tensor) (class specific detection signal) (class_size)
        #           g (pytorch Tensor) (general detection signal) (1)
        
        try:
            label_distribution = self.label_distribution
        except AttributeError:
            raise AttributeError('Error: Missing Label Distribution, either set the distribution with set_label_distribution or calculate with set_label_pred_distribution')
            
        class_size   = self.class_size
        model_output = (model_output.argmax(1).unsqueeze(0)==torch.Tensor(list(range(class_size))).unsqueeze(1)).float().sum(1).unsqueeze(1)
        
        x = self.x
        y = self.y
        P = self.P
        epsilon = self.epsilon
        
        y = y + model_output
        x = x + label_distribution*batch_size
        
        residual = y-x
        
        r = torch.matmul(torch.inverse(P),residual)*residual
        g = torch.matmul(residual.transpose(0,1),torch.matmul(torch.inverse(P),residual))
        
        x = x + epsilon*(y-x)
        
        self.x = x
        self.y = y
        
        return r.squeeze(),g.squeeze()
    
class VariationalDetector():
    
    def __init__(self,epsilon,latent_dim):
    
        self.epsilon    = epsilon
        self.latent_dim = latent_dim
            
        self.x = 0
        self.y = 0
        
    def set_latent_distribution(self,model,data_loader,device,batch_size):
        
        # Function: set_latent_distribution
        # Inputs:   model  (Pytorch tensor) (class_size)
        #           data_loader  (Pytorch dataloader)
        #           device (Pytorch device)
        #           batch_size (int)
        # Process:  sets the latent distribution from the dataset
        # Used Functions: calculate_latent_distribution
        # Output:   latent_variable_list (pytorch Tensor)
        
        self.latent_distribution,latent_variable_list = self.calculate_latent_distribution(model,data_loader,device,batch_size)
        
        return latent_variable_list
    
    def calculate_latent_distribution(self,model,data_loader,device,batch_size):
        
        # Function: calculate_latent_distribution
        # Inputs:   model  (Pytorch tensor) (class_size)
        #           data_loader  (Pytorch dataloader)
        #           device (Pytorch device)
        #           batch_size (int)
        # Process:  calculates the latent distribution from the dataset
        # Output:   latent_distribution (pytorch Tensor)
        #           latent_variable_list (pytorch Tensor)
        
        k            = 0
        percent_done = 10

        latent_variable_list = np.array([])
        latent_dim           = self.latent_dim
        
        model.eval()
       
        for image_batch, _ in data_loader:
            
            _, embed_out, _, _   = model(image_batch.to(device))
            
            for bb in range(batch_size):
            
                embed_var = torch.dot(embed_out[bb,:,:,:].squeeze(),embed_out[bb,:,:,:].squeeze())/self.latent_dim
                latent_variable_list = np.append(latent_variable_list,embed_var.detach().cpu().numpy())
                        
            k += 1

            if k % (len(data_loader)/10) == 0:
                print(str(percent_done) + '% Percent Done')
                percent_done += 10

        latent_variable_list   = torch.Tensor(latent_variable_list)
        latent_distribution    = latent_variable_list.mean()
        
        return latent_distribution,latent_variable_list
    
    def shift_filter(self,embed_out,batch_size):
       
        # Function: calculate_latent_distribution
        # Inputs:   embed_out  (Pytorch tensor) (batch_size x latent_dim)
        #           batch_size (int)
        # Process:  calculates the detection signal using filters
        # Output:   r (pytorch Tensor) (detection signal)
        
        try:
            latent_distribution = self.latent_distribution
        except AttributeError:
            raise AttributeError('Error: Missing Latent Distribution, set the distribution with calculate_latent_distribution')
            
        latent_dim = self.latent_dim
        
        embed_var = embed_out.squeeze().pow(2).sum(1)/self.latent_dim
        embed_var = embed_var.detach().cpu().sum()/batch_size

        x = self.x
        y = self.y
        epsilon = self.epsilon

        y = y + embed_var
        x = x + latent_distribution

        residual = y-x

        r = residual.pow(2)

        x = x + epsilon*(y-x)

        self.x = x
        self.y = y

        return r.squeeze()
    
    