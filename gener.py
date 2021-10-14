import torch
import torch.nn as nn
import math
from Vit import Block, Mlp



class Generator(nn.Module):
    def __init__(self, initial_dim, initial_patch):
        super(Generator, self).__init__()
        self.initial_dim = initial_dim
        self.initial_patch = initial_patch
        self.mlp = Mlp(100, int(initial_dim*(initial_patch**2)/4), initial_dim*(initial_patch**2))
        self.pos_encoding = nn.Parameter(torch.randn(1,initial_patch**2, initial_dim))
        self.enc1 = nn.Sequential(*[Block(initial_dim,4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.1, attn_drop=0.1,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm) for i in range(2)])
        self.pix1 = nn.PixelShuffle(2)
        self.enc2 = nn.Sequential(*[Block(int(initial_dim/4), 4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.1, attn_drop=0.1,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm) for i in range(2)])
        self.pix2 = nn.PixelShuffle(2)
        self.enc3 = nn.Sequential(*[Block(int(initial_dim/16), 4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.1, attn_drop=0.1,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm) for i in range(2)])
        self.fin_lin = nn.Linear(int(initial_dim/16), 3)
        self.unflat = nn.Unflatten(1,(32,32))
    def forward(self, x):
        bs = x.shape[0]
        x = x.view(bs, -1)
        x = self.mlp(x)
        x = x.view(bs,-1,self.initial_dim)
        pos_vector = self.pos_encoding.repeat(bs, 1, 1)
        x = x+ pos_vector
        x = self.enc1(x)
        dim = int(math.sqrt(x.shape[1]))
        x = x.view(bs, -1, dim, dim)
        x = self.pix1(x)
        x = x.view(x.shape[0],-1,x.shape[1])
        x = self.enc2(x)
        dim = int(math.sqrt(x.shape[1]))
        x = x.view(bs, -1, dim, dim)
        x = self.pix2(x)
        x = x.view(bs,-1,x.shape[1])
        x = self.enc3(x)
        x = self.fin_lin(x)
        x = self.unflat(x)
        x = x.permute(0, 3, 1, 2)
        return x
