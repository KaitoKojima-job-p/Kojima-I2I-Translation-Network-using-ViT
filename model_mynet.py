import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# パラメータ
patch_size = 8
projection_dim = 64
num_patches = (256 // patch_size ) ** 2
embed_dim = 64
num_heads = 2
ff_dim = 32

class Patches(nn.Module):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def forward(self, images):
        batch_size = images.shape[0]
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(batch_size, -1, self.patch_size * self.patch_size * images.shape[1])
        return patches

class PatchEncoder(nn.Module):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = nn.Linear(patch_size*patch_size*3, projection_dim)
        self.position_embedding = nn.Embedding(num_patches, projection_dim)

    def forward(self, patches):
        positions = torch.arange(0, self.num_patches, device=patches.device).unsqueeze(0).expand(patches.size(0), -1)
        encoded = self.projection(patches) + self.position_embedding(positions)
        return encoded

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super(TransformerBlock, self).__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = x
        x = self.layernorm1(x)
        x, _ = self.att(x, x, x)
        x = self.dropout(x)
        x = x + h

        h = x
        x = self.layernorm2(x)
        x = self.ffn(x)
        x = x + h
        return x, _

class Residual_Block(nn.Module):
	def __init__(self, channel_num):
		super(Residual_Block, self).__init__()
		self.conv_block1 = nn.Sequential(
			nn.Conv2d(channel_num, channel_num, 3, padding=1),
			nn.BatchNorm2d(channel_num),
			nn.ReLU(),
		) 
		self.conv_block2 = nn.Sequential(
			nn.Conv2d(channel_num, channel_num, 3, padding=1),
			nn.BatchNorm2d(channel_num),
		)
		self.relu = nn.ReLU()
	
	def forward(self, x):
		residual = x
		x = self.conv_block1(x)
		x = self.conv_block2(x)
		#x = x 
		#x = self.relu(x)
		return x + residual
     
    
class Residual_Block_v1(nn.Module):
	def __init__(self, channel_num):
		super(Residual_Block_v1, self).__init__()
		self.conv_block1 = nn.Sequential(
			nn.Conv2d(channel_num, channel_num, 3, padding=1),
			nn.BatchNorm2d(channel_num),
			nn.ReLU(),
		) 
	
	def forward(self, x):
		x = self.conv_block1(x)
		return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1, dropout=0.0):
        super(Up, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=False)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU()
        self.residual_block = Residual_Block(out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batchnorm(x)
        x = self.leakyrelu(x)
        x = self.residual_block(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class Generator(nn.Module):
    def __init__(self, patch_size=patch_size, num_patches=num_patches, projection_dim=projection_dim, num_heads=num_heads, ff_dim=ff_dim):
        super(Generator, self).__init__()
        print("teian syuhou ver2.0")
        self.patches = Patches(patch_size)
        self.patch_encoder = PatchEncoder(num_patches, projection_dim)
        self.transformer_block1 = TransformerBlock(projection_dim, num_heads, ff_dim, dropout=0.2)
        self.transformer_block2 = TransformerBlock(projection_dim, num_heads, ff_dim, dropout=0.2)
        self.transformer_block3 = TransformerBlock(projection_dim, num_heads, ff_dim, dropout=0.2)
        self.transformer_block4 = TransformerBlock(projection_dim, num_heads, ff_dim, dropout=0.2)
        
        self.up1 = Up(1024, 512, dropout=0.5)
        self.up2 = Up(512, 256, dropout=0.5)
        self.up3 = Up(256, 128, dropout=0.5)
        self.up4 = Up(128, 64)
        self.up5 = Up(64, 32)

        self.conv2d = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2d_ = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)

        self.residual_block1 = Residual_Block(32)
        self.residual_block2 = Residual_Block(32)
        self.residual_block3 = Residual_Block(32)

        self.tanh = nn.Tanh()

    def forward(self, x):
        input = x
        patches = self.patches(x)

        encoded_patches = self.patch_encoder(patches)

        x, _ = self.transformer_block1(encoded_patches)
        x, _ = self.transformer_block2(x)
        x, _ = self.transformer_block3(x)
        x, _ = self.transformer_block4(x)

        x = x.view(x.size(0), 1024, 8, 8)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)

        r = self.residual_block1(self.conv2d_(input))
        r = self.residual_block2(r)
        r = self.residual_block3(r)
        x = x + r

        x = self.conv2d(x)

        x = self.tanh(x)

        return x