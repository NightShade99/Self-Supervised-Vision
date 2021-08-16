
import math
import torch 
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F


class MultiheadSelfAttention(nn.Module):

    def __init__(self, hidden_dim, num_heads):
        super(MultiheadSelfAttention, self).__init__()
        self.heads = num_heads
        self.hidden_dim = hidden_dim 
        self.head_size = hidden_dim // num_heads
        self.softmax = nn.Softmax(dim=-1)
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        bs, n = x.shape[0], x.shape[1]
        identity = self.layer_norm(x)
        q = self.query(x).view(bs, n, self.heads, self.head_size).transpose(1, 2).contiguous()
        k = self.key(x).view(bs, n, self.heads, self.head_size).transpose(1, 2).contiguous()
        v = self.value(x).view(bs, n, self.heads, self.head_size).transpose(1, 2).contiguous()
        attn_scores = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(self.head_size)
        attn_probs = self.softmax(attn_scores)
        out = torch.einsum("bhij,bhjd->bhid", attn_probs, v).transpose(1, 2).contiguous().view(bs, n, self.hidden_dim)
        return out + identity, attn_probs


class Feedforward(nn.Module):

    def __init__(self, hidden_dim, intermediate_dim):
        super(Feedforward, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, intermediate_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(intermediate_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        identity = self.layer_norm(x)
        out = self.fc2(self.gelu(self.fc1(x)))
        return out + identity


class TransformerLayer(nn.Module):

    def __init__(self, hidden_dim, intermediate_dim, num_attention_heads):
        super(TransformerLayer, self).__init__()
        self.attention = MultiheadSelfAttention(hidden_dim, num_attention_heads)
        self.feedfwd = Feedforward(hidden_dim, intermediate_dim)

    def forward(self, x):
        out, attn_probs = self.attention(x)
        out = self.feedfwd(out)
        return out, attn_probs


class EmbeddingLayer(nn.Module):

    def __init__(self, num_global_patches, num_local_patches, input_dim, embedding_dim):
        super(EmbeddingLayer, self).__init__()
        self.num_global_patches = num_global_patches
        self.num_local_patches = num_local_patches
        self.cls_embedding = nn.Embedding(1, input_dim)
        self.pos_embedding_global = nn.Embedding(num_global_patches+1, embedding_dim)
        self.pos_embedding_local = nn.Embedding(num_local_patches+1, embedding_dim)

    def forward(self, x):
        indices = np.repeat(np.arange(x.size(1)+1).reshape(1, -1), repeats=x.shape[0], axis=0)
        if x.size(1) == self.num_global_patches:
            pos_embeds = self.pos_embedding_global(torch.from_numpy(indices).long().to(x.device))
        elif x.size(1) == self.num_local_patches:
            pos_embeds = self.pos_embedding_local(torch.from_numpy(indices).long().to(x.device))
        else:
            raise RuntimeError(f"Num patches {x.size(1)} not matching global {self.num_global_patches} or local {self.num_local_patches} patches")        
        cls_embeds = self.cls_embedding(torch.LongTensor(x.size(0), 1).zero_().to(x.device))
        x = torch.cat([cls_embeds, x], dim=1)
        x = torch.cat([x, pos_embeds], dim=-1)
        return x


class TransformerEncoder(nn.Module):

    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        self.hidden_dim = config["hidden_dim"]
        self.embedding_dim = config["embedding_dim"]
        self.intermediate_dim = config["intermediate_dim"]
        self.num_attention_heads = config["num_attention_heads"]
        self.patch_size = config["patch_size"]
        self.num_layers = config["num_encoder_layers"]
        self.num_global_patches = config["num_global_patches"]
        self.num_local_patches = config["num_local_patches"]

        self.unfold_image = nn.Unfold(kernel_size=(self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size))
        self.projection_fc = nn.Linear(3 * (self.patch_size**2) + self.embedding_dim, self.hidden_dim)
        self.embedding = EmbeddingLayer(self.num_global_patches, self.num_local_patches, 3*(self.patch_size**2), self.embedding_dim)
        self.enc_layers = nn.ModuleList([TransformerLayer(self.hidden_dim, self.intermediate_dim, self.num_attention_heads) for _ in range(self.num_layers)])

    def forward(self, img, return_attn=False):
        x = self.unfold_image(img).transpose(1, 2).contiguous()
        x = self.embedding(x)
        x = self.projection_fc(x)

        attn_probs = {}
        for i in range(self.num_layers):
            x, attn = self.enc_layers[i](x)
            attn_probs[f"layer_{i}"] = attn 
        
        if return_attn:
            return x[:, 0, :], attn_probs               
        else:   
            return x[:, 0, :]                                       # Only embeddings corresponding to [CLS] are returned