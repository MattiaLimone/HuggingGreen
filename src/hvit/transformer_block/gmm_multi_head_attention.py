import torch
import torch.nn.functional as func
from torch import nn
from torch import Tensor
from einops import rearrange
import math
from torch.distributions import Normal, MixtureSameFamily, Categorical


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 12, dropout: float = 0, dim_head: int = 64):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        inner_dim = dim_head * num_heads
        self.scale = dim_head ** -0.5
        project_out = not (num_heads == 1 and dim_head == emb_size)
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, inner_dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, emb_size),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        # extra parameters for GMM
        self.num_mixtures = emb_size
        self.mean = nn.Parameter(torch.randn(num_heads, dim_head))
        self.logvar = nn.Parameter(torch.randn(num_heads, dim_head))
        self.mix_coef = nn.Parameter(torch.randn(dim_head))

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        '''
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        '''
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
        # compute Gaussian mixture for keys

        # sum up over the last axis
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        # calculate probability density function using the attention weights as mixture coefficients
        product = torch.matmul(attn, v)
        mean = torch.mean(product, dim=1, keepdim=True)
        variance = torch.var(product, dim=1, keepdim=True)

        log_probs = -0.5 * (torch.log(2 * torch.tensor(math.pi)) + torch.log(variance) - 0.5 *(product - mean) ** 2 / variance)

        log_probs = torch.sum(log_probs, dim=-1) - torch.sum(torch.log(torch.sqrt(variance)), dim=-1)

        out = torch.matmul(torch.mul(log_probs.unsqueeze(-1), attn), v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

