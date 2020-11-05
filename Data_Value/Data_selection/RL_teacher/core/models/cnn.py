'''Mini CNN in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

class CNNMini(nn.Module):
    """ a CNN model with fewer layers than the full one defined above """
    def __init__(self, num_classes=10, fc_in=54080):
        super(CNNMini, self).__init__()

        # batchnorm after activation: https://blog.paperspace.com/busting-the-myths-about-batch-normalization/
        self.conv_base = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=10),

            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=20),

            nn.AvgPool2d(kernel_size=3, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(fc_in, 256),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.Linear(128, num_classes))

    def forward(self, input):
        output = self.conv_base(input)
        output = output.view(output.size(0), -1)
        #print(output.shape)
        output = self.fc(output)

        return output

