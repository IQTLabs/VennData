import torch
import numpy as np
import math
import torchvision
from torch import nn

from pytorch_lightning import Trainer

import time
import os

import OnlineShiftDetectors

from pytorch_lightning.core.lightning import LightningModule
import functools
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_up_data():
    
    # Function: set_up_data
    # Inputs:   none
    # Process:  returns datasets for testing
    # Output:   cifar10_trainloader (pytorch dataloader)
    #           cifar10_testloader (pytorch dataloader)
    #           gen_testloader (pytorch dataloader)

    n_epochs   = 150
    batch_size = int(1e2)
    lr         = 0.01

    # define series of transforms to pre process images 
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    # load training set 
    cifar10_trainset = torchvision.datasets.CIFAR10('/home/fmejia/fmejia/Cypercat/cyphercat/datasets//', train=True, transform=transform, download=True)
    cifar10_trainloader = torch.utils.data.DataLoader(cifar10_trainset, batch_size=batch_size, shuffle=True, num_workers=16)

    # load test set 
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

    return cifar10_trainloader,cifar10_testloader,gen_testloader



class ResnetGenerator(LightningModule):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self,learning_rate=0.001):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        use_dropout = True
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        # norm_layer = functools.partial(nn.InstanceNorm2d)
        input_nc = 3
        output_nc = 3
        ngf = 64
        ndf = 64

        z_dim = 256

        n_blocks=6
        padding_type='reflect'
        
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.LeakyReLU(0.2, inplace = True)]

        n_downsampling = 5
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if i < (n_downsampling-1):
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.LeakyReLU(0.2, inplace = True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias)]
                     
                    
        ## variance model
        model2 = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.LeakyReLU(0.2, inplace = True)]

        n_downsampling = 5
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if i < (n_downsampling-1):
                model2 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.LeakyReLU(0.2, inplace = True)]
            else:
                model2 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias)]
                     
               
        model_upsample1 = []
        
        for i in range(n_downsampling-2):
            mult = 2 ** (n_downsampling-i)
            model_upsample1 += [                      
#                     nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, padding =1, stride = 1),            
#                     nn.Upsample(scale_factor=2, mode='bilinear'),
                
                      nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.LeakyReLU(0.2, inplace = True)]
        n_downsampling = 2
        mult = 2 ** (n_downsampling)        
        model_resnet = []
        for i in range(n_blocks):       # add ResNet blocks

            model_resnet += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        
        model_upsample = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model_upsample += [
#                       nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, padding =1, stride = 1),            
#                       nn.Upsample(scale_factor=2, mode='bilinear'),
                    nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.LeakyReLU(0.2, inplace = True)]
        model_upsample += [nn.ReflectionPad2d(3)]
        model_upsample += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model_upsample += [nn.Tanh()]

        self.model = nn.Sequential(*model)
        self.model_variance = nn.Sequential(*model2)
        self.model_resnet = nn.Sequential(*model_resnet)
        self.model_upsample1 = nn.Sequential(*model_upsample1)    
        self.model_upsample = nn.Sequential(*model_upsample)
        
        self.mean = torch.tensor((0.4914, 0.4822, 0.4465))#.to(device)
        self.mean = self.mean.view(-1,1,1)
        self.var = torch.tensor((0.2023, 0.1994, 0.2010))#.to(device)
        self.var = self.var.view(-1,1,1)

        self.loss_function = nn.SmoothL1Loss()
        self.learning_rate = learning_rate
        
    def forward(self, input, decode = False):
        """Standard forward"""
        if decode:                 
                x = self.model_upsample1(input)            
                x = self.model_resnet(x)
                x = self.model_upsample(x)
                x = x / 2 + 0.5
                x = (x - self.mean)/self.var
                return x
        mean = self.model(input)
        variance = self.model_variance(input)

        sample = Variable(torch.randn(mean.size()).type(torch.cuda.FloatTensor))
        x1 = mean + (variance * sample)
        x = self.model_upsample1(x1)
        x = self.model_resnet(x)
        
        x = self.model_upsample(x)
        x = x / 2 + 0.5
        x = (x - self.mean.cuda())/self.var.cuda()
        return x, x1, mean, variance

    def validation_step(self, batch, batch_idx):
        
        imgs, labels = batch
        
        out_img, embed_out, mean, variance = self(imgs)
        
        AE_loss = self.loss_function(out_img, imgs)
        kl_loss = (mean ** 2 + variance **2 - torch.log(variance ** 2) - 1).mean()
        loss    = AE_loss + kl_loss/10
        
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):

        avg_loss         = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}
    
    def training_step(self, batch, batch_idx):
        
        imgs, labels = batch
        
        out_img, embed_out, mean, variance = self(imgs)
        AE_loss = self.loss_function(out_img, imgs)
        kl_loss = (mean ** 2 + variance **2 - torch.log(variance ** 2) - 1).mean()
        loss    = AE_loss + kl_loss/10
        
        tensorboard_logs = {'train_loss': loss}
        
        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        
        beta1   = 0.5
        lr_adam = 1e-04
        #optimizer_g = torch.optim.Adam(Generator.parameters(), lr = lr_adam, betas = (beta1, 0.999))

        return torch.optim.Adam(self.parameters(), lr=(self.learning_rate), betas = (beta1, 0.999))
    
class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.LeakyReLU(0.2, inplace = True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.2)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

def run_test(iteration,cifar10_trainloader,cifar10_testloader,gen_testloader):

    # Function: run_test
    # Inputs:   iteration (int)
    #           cifar10_trainloader (pytorch dataloader)
    #           cifar10_testloader (pytorch dataloader)
    #           gen_testloader (pytorch dataloader)
    # Process:  runs test for VAE covariate detector
    # Output:   none
    
    stats_path = '/home/jgornet/Covariate_Testing/statistics'
    model_path = '/home/jgornet/Covariate_Testing/models'
    
    model_file = os.path.join(model_path,'model_num_' + str(iteration) + '.ckpt')
    
    train_file = os.path.join(stats_path,'train_num_' + str(iteration) + '.npy')
    test_file  = os.path.join(stats_path,'test_num_' + str(iteration) + '.npy')
    gen_file   = os.path.join(stats_path,'gen_num_' + str(iteration) + '.npy')
        
    model = ResnetGenerator()

    start_time = time.time()

    trainer = Trainer(gpus=4, num_nodes=1, distributed_backend='dp',auto_lr_find=True,profiler=True,max_epochs=3,checkpoint_callback=False)
    trainer.fit(model, cifar10_trainloader, cifar10_testloader)

    #trainer.save_checkpoint(model_file)
    
    #model = ResnetGenerator().load_from_checkpoint(checkpoint_path=model_file)
    model.to(device)

    variational_detector = OnlineShiftDetectors.VariationalDetector(0.1,2048)

    latent_variable_list = variational_detector.set_latent_distribution(model,cifar10_trainloader,device,100)

    r_train = np.array([])

    for epoch in range(1):

        for i, batch in enumerate(cifar10_trainloader, 0):

            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)

            _, embed_out, _, _   = model(imgs.to(device))
            r_var                = variational_detector.shift_filter(embed_out.cpu(),100)
            r_train              = np.append(r_train,r_var.cpu().detach().numpy())    

    r_test = np.array([])

    for epoch in range(5):

        for i, batch in enumerate(cifar10_testloader, 0):

            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)

            _, embed_out, _, _   = model(imgs.to(device))
            r_var                = variational_detector.shift_filter(embed_out.cpu(),100)
            r_test               = np.append(r_test,r_var.cpu().detach().numpy())    

    r_gen = np.array([])

    for epoch in range(5):

        for i, batch in enumerate(gen_testloader, 0):

            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)

            _, embed_out, _, _   = model(imgs.to(device))
            r_var                = variational_detector.shift_filter(embed_out.cpu(),100)
            r_gen                = np.append(r_gen,r_var.cpu().detach().numpy())    

    np.save(train_file,r_train)
    np.save(test_file,r_test)
    np.save(gen_file,r_gen)
    
if __name__ == '__main__':

    use_dropout = True
    norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    # norm_layer = functools.partial(nn.InstanceNorm2d)
    input_nc = 3
    output_nc = 3
    ngf = 64
    ndf = 64

    z_dim = 256
    
    cifar10_trainloader,cifar10_testloader,gen_testloader = set_up_data()

    for iter_num in range(250):
    
        run_test(iter_num,cifar10_trainloader,cifar10_testloader,gen_testloader)