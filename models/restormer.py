import torch
import torch.nn as nn
import torch.nn.functional as F

class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, stride=1, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(b, self.num_heads, c // self.num_heads, h * w)
        k = k.view(b, self.num_heads, c // self.num_heads, h * w)
        v = v.view(b, self.num_heads, c // self.num_heads, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = out.view(b, c, h, w)

        out = self.project_out(out)
        return out

class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor=2.66):
        super(GDFN, self).__init__()
        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, stride=1, padding=1, groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels)

    def forward(self, x):
        b, c, h, w = x.shape
        
        # LayerNorm expects (B, H, W, C)
        x_norm = x.permute(0, 2, 3, 1).contiguous()
        x_norm = self.norm1(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2).contiguous()
        x = x + self.attn(x_norm)

        x_norm = x.permute(0, 2, 3, 1).contiguous()
        x_norm = self.norm2(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2).contiguous()
        x = x + self.ffn(x_norm)

        return x

class Restormer(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dim=32, num_blocks=[2, 2, 2, 2], num_heads=[1, 2, 4, 8]):
        super(Restormer, self).__init__()
        self.embed = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1, bias=False)
        
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim, num_heads[0]) for _ in range(num_blocks[0])])
        self.down1 = nn.Conv2d(dim, dim*2, kernel_size=4, stride=2, padding=1, bias=False)
        
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim*2, num_heads[1]) for _ in range(num_blocks[1])])
        self.down2 = nn.Conv2d(dim*2, dim*4, kernel_size=4, stride=2, padding=1, bias=False)
        
        self.latent = nn.Sequential(*[TransformerBlock(dim*4, num_heads[2]) for _ in range(num_blocks[2])])
        
        self.up2 = nn.ConvTranspose2d(dim*4, dim*2, kernel_size=2, stride=2, bias=False)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim*2, num_heads[1]) for _ in range(num_blocks[1])])
        
        self.up1 = nn.ConvTranspose2d(dim*2, dim, kernel_size=2, stride=2, bias=False)
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim, num_heads[0]) for _ in range(num_blocks[0])])
        
        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        inp_enc_level1 = self.embed(x)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        
        inp_latent = self.down2(out_enc_level2)
        out_latent = self.latent(inp_latent)
        
        inp_dec_level2 = self.up2(out_latent) + out_enc_level2
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        
        inp_dec_level1 = self.up1(out_dec_level2) + out_enc_level1
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out = self.output(out_dec_level1) + x # Global residual learning
        
        return out
