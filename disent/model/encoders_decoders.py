import torch.nn as nn


# ========================================================================= #
# HELPER                                                                    #
# ========================================================================= #


class View(nn.Module):
    """From: https://github.com/1Konny/Beta-VAE/blob/master/model.py"""
    def __init__(self, size):
        super().__init__()
        self.size = size
    def forward(self, tensor):
        return tensor.view(self.size)

class Unsqueeze3D(nn.Module):
    """From: https://github.com/amir-abdi/disentanglement-pytorch"""
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = x.unsqueeze(-1)
        return x

class Flatten3D(nn.Module):
    """From: https://github.com/amir-abdi/disentanglement-pytorch"""
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x



class BaseModule(nn.Module):
    @property
    def z_dim(self):
        raise NotImplementedError
    @property
    def x_dim(self):
        raise NotImplementedError


# ========================================================================= #
# simple 28x28 fully connected models                                       #
# ========================================================================= #


class EncoderSimpleFC(nn.Module):
    def __init__(self, x_dim=28*28, h_dim1=512, h_dim2=256, z_dim=2):
        super().__init__()
        self.z_dim = z_dim  # REQUIRED
        self.model = nn.Sequential(
            Flatten3D(),
            nn.Linear(x_dim, h_dim1),
            nn.ReLU(True),
            nn.Linear(h_dim1, h_dim2),
            nn.ReLU(True),
        )
        self.enc3mean = nn.Linear(h_dim2, z_dim)
        self.enc3logvar = nn.Linear(h_dim2, z_dim)

    def forward(self, x):
        # encoder | p(z|x)
        x = self.model(x)
        return self.enc3mean(x), self.enc3logvar(x)

class DecoderSimpleFC(nn.Module):
    def __init__(self, x_dim=28*28, h_dim1=512, h_dim2=256, z_dim=2):
        super().__init__()
        self.z_dim = z_dim # REQUIRED
        self.model = nn.Sequential(
            nn.Linear(z_dim, h_dim2),
            nn.ReLU(True),
            nn.Linear(h_dim2, h_dim1),
            nn.ReLU(True),
            nn.Linear(h_dim1, x_dim),
        )

    def forward(self, z):
        # decoder | p(x|z)
        return self.model(z)


# ========================================================================= #
# simple 64x64 convolutional models                                         #
# ========================================================================= #


class EncoderSimpleConv64(nn.Module):
    """From: https://github.com/amir-abdi/disentanglement-pytorch"""

    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__()
        assert image_size == 64, 'This model only works with image size 64x64.'
        self.z_dim = latent_dim  # REQUIRED

        self.model = nn.Sequential(
            nn.Conv2d(num_channels, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 4, 2, 1),
            nn.ReLU(True),
            Flatten3D(),
        )
        self.enc3mean = nn.Linear(256, self.z_dim)
        self.enc3logvar = nn.Linear(256, self.z_dim)

    def forward(self, x):
        # encoder | p(z|x)
        x = self.model(x)
        return self.enc3mean(x), self.enc3logvar(x)


class DecoderSimpleConv64(nn.Module):
    """From: https://github.com/amir-abdi/disentanglement-pytorch"""

    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__()
        assert image_size == 64, 'This model only works with image size 64x64.'
        self.z_dim = latent_dim  # REQUIRED

        self.model = nn.Sequential(
            Unsqueeze3D(),
            nn.Conv2d(latent_dim, 256, 1, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 128, 4, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, num_channels, 3, 1)
        )
        # output shape = bs x 3 x 64 x 64

    def forward(self, x):
        return self.model(x)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
