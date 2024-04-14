import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
import numpy as np

from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
from functions import to_price_paths

SCALE_FACTOR = 45

class Generator(nn.Module):
    def __init__(self, seq_len=42, conditions_dim=0, patch_size=7, channels=1, latent_dim=100, embed_dim=10, depth=3,
                 num_heads=5, forward_drop_rate=0.5, attn_drop_rate=0.5):
        super(Generator, self).__init__()
        self.channels = channels
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.conditions_dim = conditions_dim
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.depth = depth
        self.attn_drop_rate = attn_drop_rate
        self.forward_drop_rate = forward_drop_rate

        self.l1 = nn.Sequential(
            nn.Linear(self.latent_dim, self.seq_len*self.embed_dim),
            nn.Tanh()
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, self.embed_dim))
        self.blocks = Gen_TransformerEncoder(
            depth=self.depth,
            seq_len = self.seq_len,
            emb_size = self.embed_dim,
            conditions_dim = self.conditions_dim,
            drop_p = self.attn_drop_rate,
            forward_drop_p=self.forward_drop_rate
        )
        self.deconv = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.channels, 1, 1, 0)
        )

    def forward(self, z, conditions):
        conditions = torch.reshape(conditions, (conditions.shape[0], -1))
        z = self.l1(z).view(-1, self.seq_len, self.embed_dim)
        z = z + self.pos_embed
        out = self.blocks(z, conditions)
        out = out.reshape(out.shape[0], 1, out.shape[1], out.shape[2])
        out = self.deconv(out.permute(0, 3, 1, 2))
        out = out.view(-1, self.channels, 1, self.seq_len)
        out = out / SCALE_FACTOR
        out = to_price_paths(out)
        return out
    
class Gen_TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 seq_len,
                 emb_size,
                 conditions_dim,
                 num_heads=5,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        # super().__init__(
        #     ResidualAdd(nn.Sequential(
        #         nn.LayerNorm(emb_size),
        #         MultiHeadAttention(emb_size, num_heads, drop_p),
        #         nn.Dropout(drop_p)
        #     )),
        #     ResidualAdd(nn.Sequential(
        #         nn.LayerNorm(emb_size),
        #         FeedForwardBlock(
        #             emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
        #         nn.Dropout(drop_p)
        #     ))
        # )
        super().__init__()
        self.conditional_norm1 = ConditionalLayerNorm2d(seq_len, emb_size, conditions_dim)
        self.multi_head_attention = MultiHeadAttention(emb_size, num_heads, drop_p)
        self.dropout1 = nn.Dropout(drop_p)
        self.conditional_norm2 = ConditionalLayerNorm2d(seq_len, emb_size, conditions_dim)
        self.feed_forward = FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p)
        self.dropout2 = nn.Dropout(drop_p)

    def forward(self, x, conditions):
        # First Conditional LayerNorm and MultiHead Attention + Residual
        out = self.conditional_norm1(x, conditions)
        out = self.multi_head_attention(out)
        out = self.dropout1(out)
        out = out + x  # Residual connection

        # Second Conditional LayerNorm and FeedForward + Residual
        out = self.conditional_norm2(out, conditions)
        out = self.feed_forward(out)
        out = self.dropout2(out)
        out = out + x  # Residual connection

        return out

        
class Gen_TransformerEncoder(nn.Sequential):
    def __init__(self, depth=8, **kwargs):
        super().__init__()
        self.blocks = nn.Sequential(*[Gen_TransformerEncoderBlock(**kwargs) for _ in range(depth)])
        # super().__init__(*[Gen_TransformerEncoderBlock(**kwargs) for _ in range(depth)])
    
    def forward(self, x, conditions):
        for block in self.blocks:
            x = block(x, conditions)
        return x
        
        
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ConditionalLayerNorm2d(nn.Module):
    def __init__(self, seq_len, emb_size, conditions_dim): # num_features refer to C-dimension in conditions
        super().__init__()
        # Assuming the input x to be in shape (B, seq_len, emb_size)
        # LayerNorm will be applied across (seq_len, emb_size) for each element in the batch
        self.ln = nn.LayerNorm([seq_len, emb_size])

        self.embed_gamma = nn.Linear(conditions_dim, emb_size, bias=False)
        self.embed_beta = nn.Linear(conditions_dim, emb_size, bias=False)

    def forward(self, x, conditions):
        b, seq_len, emb_size = x.shape
        # Apply LayerNorm
        out = self.ln(x)  # Shape: (B, seq_len, emb_size)

        gamma = self.embed_gamma(conditions).view(b, 1, emb_size)  # Shape: (b, 1, emb_size)
        beta = self.embed_beta(conditions).view(b, 1, emb_size)  # Shape: (b, 1, emb_size)

        # Apply conditional normalization adjustments
        out = out * gamma + beta
        return out

    
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
    
    
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )
        
        
class Dis_TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size=100,
                 num_heads=5,
                 drop_p=0.,
                 forward_expansion=4,
                 forward_drop_p=0.):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            ))
        )


class Dis_TransformerEncoder(nn.Sequential):
    def __init__(self, depth=8, **kwargs):
        super().__init__(*[Dis_TransformerEncoderBlock(**kwargs) for _ in range(depth)])
         
        
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size=100, n_classes=1):
        super().__init__()

        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):
        out = self.clshead(x)
        return out

    
class PatchEmbedding_Linear(nn.Module):
    #what are the proper parameters set here?
    def __init__(self, in_channels=1, patch_size=21, emb_size=50, seq_len=42, conditions_dim=0):
        # self.patch_size = patch_size
        super().__init__()
        #change the conv2d parameters here
        self.project_x = nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=1, s2=patch_size),
            nn.Linear(patch_size*in_channels, emb_size),
            nn.Tanh()
        )
        # self.project_conds = nn.Sequential(
        #     Rearrange('b c h w -> b (h w) c'),
        #     nn.Linear(in_channels, emb_size),
        #     nn.Tanh()
        # )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_emb = nn.Parameter(torch.randn(seq_len//patch_size + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.project_x(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_emb
        return x
        
        
class Discriminator(nn.Sequential):
    def __init__(self, 
                 in_channels=1,
                 patch_size=7,
                 emb_size=50, 
                 seq_len=42,
                 conditions_dim=0,
                 depth=3, 
                 n_classes=1, 
                 **kwargs):
        assert seq_len%patch_size == 0
        super().__init__()
        self.patch_embedding = PatchEmbedding_Linear(in_channels, patch_size, emb_size, seq_len, conditions_dim)
        self.encoder = Dis_TransformerEncoder(depth, emb_size=emb_size, drop_p=0.5, forward_drop_p=0.5, **kwargs)
        self.conditional_norm = ConditionalLayerNorm2d(patch_size, emb_size, conditions_dim)
        self.classification_head = ClassificationHead(emb_size, n_classes)
        
    def forward(self, x, conditions):
        x = x[:,:,:,1:]
        x = self.patch_embedding(x)
        x = self.encoder(x)
        x = self.conditional_norm(x, conditions)
        x = self.classification_head(x)
        return x