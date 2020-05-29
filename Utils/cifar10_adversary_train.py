import sys 
import numpy as np 
import matplotlib.pyplot as plt

from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torchvision 
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import pickle 
sys.path.insert(0, '../../Utils')

import models
from train import *
from metrics import * 

print("Python: %s" % sys.version)
print("Pytorch: %s" % torch.__version__)

n_epochs = 100
batch_size = 128

# define series of transforms to pre process images 
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),    
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
    

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


# load training set 
cifar10_trainset = torchvision.datasets.CIFAR10('../../Datasets/', train=True, transform=transform, download=True)
cifar10_trainloader = torch.utils.data.DataLoader(cifar10_trainset, batch_size=batch_size, shuffle=True, num_workers=2)

# load test set 
cifar10_testset = torchvision.datasets.CIFAR10('../../Datasets/', train=False, transform=transform, download=True)
cifar10_testloader = torch.utils.data.DataLoader(cifar10_testset, batch_size=batch_size, shuffle=False, num_workers=2)

# helper function to unnormalize and plot image 
def imshow(img):
    mean = torch.tensor((0.4914, 0.4822, 0.4465)).to(device)
    mean = mean.view(-1,1,1).cpu().detach().numpy()
    var = torch.tensor((0.2023, 0.1994, 0.2010)).to(device)
    var = var.view(-1,1,1).cpu().detach().numpy()
    
    img = np.array(img)
    img = (img*var) + mean
    img = np.moveaxis(img, 0, -1)
    plt.imshow(img)
    plt.show()
    

# determine device to run network on (runs on gpu if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

n_classes = 10
class_criterion = nn.CrossEntropyLoss()

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
#         self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
#         out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv1(F.leaky_relu(self.bn1(x), 0.1))
        out = self.conv2(F.leaky_relu(self.bn2(out), 0.1))
        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = int((depth-4)/6)
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.leaky_relu(self.bn1(out), 0.1)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out



classifier=Wide_ResNet(28, 10, 0.3, 10)
classifier = classifier.to(device)

def ga_PGD(image, net, iterations, lr,category, verbose = True):        
    if iterations == 0:
        return image
    category = category.to(device)
        
    image = image.cuda()
    
    input_orig = Variable(image, requires_grad=False).to(device)
    
    losses = []
    losses2 = []
    modifier = torch.tensor(np.random.uniform(-10/255*2.78, 10/255*2.78, image.size())).float().to(device)
    modifier_var = Variable(modifier, requires_grad=True)
    var = torch.tensor((0.2023, 0.1994, 0.2010)).to(device)
    var = var.view(-1,1,1)
    
    optimizer = optim.Adam([modifier_var], lr)
    alpha = 2/255*2.78
    for j in range(iterations):      
        
        net.zero_grad()
        input_adv = torch.clamp(torch.clamp(modifier_var,-10/255*2.78, 10/255*2.78) + input_orig, -2.78, 2.78)
#         input_adv = torch.clamp(modifier_var + input_orig, -2.78, 2.78)        
        out = net(input_adv)
        loss_class = class_criterion(out,category)
        loss_class.backward()
        modifier_var.data = modifier_var + alpha*modifier_var.grad.detach().sign()                
        modifier_var.grad.zero_()
        
    if verbose:
        losses.append(loss_class.data)
        plt.plot(losses)
        plt.show()
    
    return input_adv

ae_criterion = nn.MSELoss()

def adv_train(net, data_loader, test_loader, optimizer, criterion, n_epochs, classes=None, verbose=False):
    losses = []
    
    for epoch in range(n_epochs):
        net.train()
        total = 0
        correct = 0
        for i, batch in enumerate(data_loader):

            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()

            im_fool= ga_PGD(imgs,net,10,1e-1,labels,False).detach()
            outputs = net(im_fool)
            
            ## accuracy calc
            predicted = outputs.argmax(dim=1)
            total += imgs.size(0)
            correct += predicted.eq(labels).sum().item()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.data)
            if verbose:
                print("[%d/%d][%d/%d] loss = %f" % (epoch, n_epochs, i, len(data_loader), loss.item()))

        # evaluate performance on testset at the end of each epoch
        print("[%d/%d]" %(epoch, n_epochs))

        print("Train Accuracy %f" %(correct/total*100))
    return losses
optimizer = optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0002)  
try:
    losses
except:       
    losses = []
    

losses = losses + adv_train(classifier, cifar10_trainloader, cifar10_testloader, optimizer, class_criterion, n_epochs= 10, verbose=True)

torch.save(classifier, 'CIFAR10_VGG_trained_robust10.pt')

def save_checkpoint(model=None, optimizer=None, epoch=None,
                    data_descriptor=None, loss=None, accuracy=None, path='./',
                    filename='checkpoint', ext='.pth.tar'):
    state = {
        'epoch': epoch,
        'arch': str(model.type),
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'dataset': data_descriptor
        }
    torch.save(state, path+filename+ext)
save_checkpoint(classifier, optimizer, n_epochs, filename = 'vgg_checkpoint_robust10')

file_Name = "loss10.p"
# open the file for writing
# with open(file_Name,'wb') as fileObject:
#     pickle.dump(im_hist, fileObject)
    
    
    
    
losses = losses + adv_train(classifier, cifar10_trainloader, cifar10_testloader, optimizer, class_criterion, n_epochs= 40, verbose=True)

torch.save(classifier, 'CIFAR10_VGG_trained_robust50.pt')

save_checkpoint(classifier, optimizer, n_epochs, filename = 'vgg_checkpoint_robust50')

file_Name = "loss50.p"
# open the file for writing
# with open(file_Name,'wb') as fileObject:
#     pickle.dump(im_hist, fileObject)
    
    
    
    
losses = losses + adv_train(classifier, cifar10_trainloader, cifar10_testloader, optimizer, class_criterion, n_epochs= 50, verbose=True)

torch.save(classifier, 'CIFAR10_VGG_trained_robust100.pt')

save_checkpoint(classifier, optimizer, n_epochs, filename = 'vgg_checkpoint_robust100')

file_Name = "loss100.p"
# open the file for writing
# with open(file_Name,'wb') as fileObject:
#     pickle.dump(im_hist, fileObject)
    
    
optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0002)  
losses = losses + adv_train(classifier, cifar10_trainloader, cifar10_testloader, optimizer, class_criterion, n_epochs= 50, verbose=True)
    
torch.save(classifier, 'CIFAR10_VGG_trained_robust150.pt')   

save_checkpoint(classifier, optimizer, n_epochs, filename = 'vgg_checkpoint_robust150')
file_Name = "loss150.p"
# open the file for writing
# with open(file_Name,'wb') as fileObject:
#     pickle.dump(im_hist, fileObject)
    
optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0002)  
losses = losses + adv_train(classifier, cifar10_trainloader, cifar10_testloader, optimizer, class_criterion, n_epochs= 50, verbose=True)

torch.save(classifier, 'CIFAR10_VGG_trained_robust200.pt')

save_checkpoint(classifier, optimizer, n_epochs, filename = 'vgg_checkpoint_robust200')
file_Name = "loss200.p"
# open the file for writing
# with open(file_Name,'wb') as fileObject:
#     pickle.dump(im_hist, fileObject)































