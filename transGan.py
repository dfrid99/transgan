import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from vit_pytorch import ViT
import torchvision.datasets as dset
import numpy as np
import matplotlib.pyplot as plt
from gener import Generator
import torch.optim as optim
from torch_position_embedding import position_embedding
image_size = 32
batch_size = 8

def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

discriminator = ViT(
    image_size = image_size,
    patch_size =  4,
    num_classes = 1,
    dim = 384,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
    )


class mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, act=nn.ReLU()):
        super(mlp, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            act,
            nn.Linear(hidden_features, out_features),
            act,
        )

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.mlp = mlp(100, 384 * 12, 384 * 64)
        encoder_layer1 = nn.TransformerEncoderLayer(d_model=384, nhead=8)
        self.enc1 = nn.TransformerEncoder(encoder_layer1, num_layers=2)
        self.pix1 = nn.PixelShuffle(2)
        encoder_layer2 = nn.TransformerEncoderLayer(d_model=96, nhead=8)
        self.enc2 = nn.TransformerEncoder(encoder_layer2, num_layers=2)
        self.pix2 = nn.PixelShuffle(2)
        encoder_layer3 = nn.TransformerEncoderLayer(d_model=24, nhead=8)
        self.enc3 = nn.TransformerEncoder(encoder_layer3, num_layers=2)
        self.fin_lin = nn.Linear(24, 3)
        self.unflat = nn.Unflatten(1, (32, 32))

    def forward(self, x):
        x = self.mlp(x)
        x = x.view(-1, int(x.shape[1] / 384), 384)
        x = self.enc1(x)
        dim = int(math.sqrt(x.shape[1]))
        x = x.view(-1, x.shape[2], dim, dim)
        x = self.pix1(x)
        x = x.view(x.shape[0], -1, x.shape[1])
        x = self.enc2(x)
        dim = int(math.sqrt(x.shape[1]))
        x = x.view(-1, x.shape[2], dim, dim)
        x = self.pix2(x)
        x = x.view(x.shape[0], -1, x.shape[1])
        x = self.enc3(x)
        x = self.fin_lin(x)
        x = self.unflat(x)
        x = x.permute(0, 3, 1, 2)
        return x
 



    

    


