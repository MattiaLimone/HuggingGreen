import torch
import torch.nn.functional as func
from torch import nn
from torch import Tensor
from einops import rearrange
from torch.distributions import Normal, MixtureSameFamily, Categorical


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        # This parameter are necessary for the Guassian mixture model
        # define parameters for Gaussian mixture model
        self.num_mixtures = emb_size
        self.mean = nn.Parameter(torch.randn(self.num_mixtures, self.emb_size))
        self.logvar = nn.Parameter(torch.randn(self.num_mixtures, self.emb_size))
        self.mix_coef = nn.Parameter(torch.randn(self.num_mixtures))

    
    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # compute Gaussian mixture for keys
        mix = MixtureSameFamily(
            mixture_distribution=Categorical(logits=self.mix_coef),
            component_distribution=Normal(loc=self.mean, scale=torch.exp(self.logvar / 2))
        )
        keys = rearrange(keys, "b h n d -> b n (h d)")
        key_mix = mix.log_prob(keys)
        key_mix = rearrange(key_mix, "b n (h d) -> b h n d", h=self.num_heads)
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, key_mix)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = func.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

