import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, dim):
        super(UnFlatten, self).__init__()
        self.dim = dim
    def forward(self, input):
        return input.view(input.size(0), self.dim, 1, 1)

class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=256, z_dim=128, device='cpu'):
        super(VAE, self).__init__()
        self.dev = device
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            #nn.ReLU(),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            #nn.ReLU(),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            #nn.ReLU(),
            nn.LeakyReLU(),
            nn.Conv2d(128, h_dim, kernel_size=3, stride=2),
            nn.BatchNorm2d(h_dim),
            #nn.ReLU(),
            nn.LeakyReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(dim=h_dim),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            #nn.ReLU(),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            #nn.ReLU(),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            #nn.ReLU(),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(self.dev)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x, bottleneck_only=False):

        h = self.encoder(x)
        #print('output encoder shape = {}'.format(h.shape))
        z, mu, logvar = self.bottleneck(h)
        z = self.fc3(z)
        return self.decoder(z), mu, logvar

