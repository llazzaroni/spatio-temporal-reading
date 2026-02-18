import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Define all the model components / submodules

class TransformerBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.AttentionBlock = AttentionBlock(d_model)
        self.ffn = ffn(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, tokens):
        # First the self attention
        x = self.AttentionBlock(tokens)

        # Then add and norm
        x = self.norm1(tokens + x)

        # Then the ffn
        y = self.ffn(x)

        # Finally, the add and norm
        z = self.norm2(y + x)

        return z


class AttentionBlock(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.keys_linear = nn.Linear(d_model, d_model)
        self.queries_linear = nn.Linear(d_model, d_model)
        self.values_linear = nn.Linear(d_model, d_model)
        self.distance_scale = nn.Parameter(torch.tensor(0.01))
        self.IGNORE = float("-inf")

    def forward(self, tokens):
        # tokens have dimensions (batches, len_sequence, d_model)
        B, T, D = tokens.shape

        # First compute keys, queries and values
        keys = self.keys_linear(tokens) # (batches, len_sequences, d_model)
        queries = self.queries_linear(tokens) # (batches, len_sequences, d_model)
        values = self.values_linear(tokens) # (batches, len_sequences, d_model)

        # Second compute the attention scores
        scores = torch.matmul(queries, torch.transpose(keys, 1, 2)) # (batches, len_sequences, d_model)

        # Give a bias towards close tokens
        positions = torch.arange(T, device=tokens.device)
        distance = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs().float()
        distance_bias = -self.distance_scale * distance 
        scores = scores + distance_bias.unsqueeze(0) # (batches, len_sequences, len_sequences)

        # Apply the causal mask
        scores = self.apply_causal_mask(scores)
        
        # Third normalize by the model dimension
        scores = scores / (self.d_model ** 0.5)

        # Fourth compute the softmax
        attn = scores.softmax(dim=-1) # (batches, len_sequences, len_sequences)

        # Finally, multiply with the values
        out = torch.matmul(attn, values) # (batches, len_sequences, d_model)

        return out
    
    def apply_causal_mask(self, scores):
        ones = torch.ones_like(scores) # (n_batches, len_sequences, len_sequences)
        mask = torch.triu(ones, diagonal=1)
        mask = mask.to(torch.bool)
        return scores.masked_fill(mask, self.IGNORE)
    

class ffn(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)

    def forward(self, tokens):
        # Tokens is of dimensions (n_batches, len_sequences, d_model)
        x = self.linear1(tokens)
        x = F.relu(x)
        x = self.linear2(x)
        return x
    


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()

        self.alpha = nn.Parameter(torch.tensor(0.1))

        positions = torch.arange(0, max_len, dtype = torch.float32).unsqueeze(1) # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)) 

        pe_raw = positions * div_term # (max_len, d_model)

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pe_raw)
        pe[:, 1::2] = torch.cos(pe_raw)

        pe = pe.unsqueeze(0) # (1, max_len, d_model)

        self.pe = pe

    def forward(self, x):
        return x + self.pe[:, :x.shape[1], :] * self.alpha
    
class gptProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(768, 64)
        self.linear2 = nn.Linear(64, 4)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        return self.linear2(x)

        
class TransformerBlockMultiHeadAttn(nn.Module):
    def __init__(self, d_model, H):
        super().__init__()
        self.MultiHeadAttentionBlock = MultiHeadAttentionBlock(d_model, H)
        self.ffn = ffn(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, tokens):
        # First the self attention
        x = self.MultiHeadAttentionBlock(tokens)

        # Then add and norm
        x = self.norm1(tokens + x)

        # Then the ffn
        y = self.ffn(x)

        # Finally, the add and norm
        z = self.norm2(y + x)

        return z
    

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model, H):
        super().__init__()
        self.d_model = d_model
        self.keys_linear = nn.Linear(d_model, d_model)
        self.queries_linear = nn.Linear(d_model, d_model)
        self.values_linear = nn.Linear(d_model, d_model)
        self.distance_scale = nn.Parameter(torch.tensor(0.01))
        self.IGNORE = float("-inf")
        self.H = H
        self.W_out = nn.Linear(d_model, d_model)

    def forward(self, tokens):
        # tokens have dimensions (batches, len_sequence, d_model)
        B, T, D = tokens.shape

        # First compute keys, queries and values
        keys = self.keys_linear(tokens) # (batches, len_sequences, d_model)
        queries = self.queries_linear(tokens) # (batches, len_sequences, d_model)
        values = self.values_linear(tokens) # (batches, len_sequences, d_model)

        keys = keys.view(B, T, self.H, D // self.H) # (batches, len_sequences, heads, d_model/heads)
        queries = queries.view(B, T, self.H, D // self.H) # (batches, len_sequences, heads, d_model/heads)
        values = values.view(B, T, self.H, D // self.H) # (batches, len_sequences, heads, d_model/heads)

        # Second compute the attention scores
        S = torch.einsum("bthd,buhd->btuh", queries, keys) # (batches, len_sequences, len_sequences, heads)
        

        # Give a bias towards close tokens
        positions = torch.arange(T, device=tokens.device)
        distance = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs().float()
        distance_bias = -self.distance_scale * distance 

        S = S + distance_bias.unsqueeze(0).unsqueeze(-1) # (batches, len_sequences, len_sequences, d_model/heads)

        # Apply the causal mask
        S = self.apply_causal_mask(S)
        
        # Third normalize by the model dimension
        S = S / ((D // self.H) ** 0.5)

        # Fourth compute the softmax
        S = S.softmax(dim=2) # (batches, len_sequences, len_sequences, d_model/heads)

        # Finally, multiply with the values
        A = torch.einsum("btuh,buhd->bthd", S, values)

        A = A.reshape(B, T, D)
        out = self.W_out(A)

        return out   

    def apply_causal_mask(self, scores):
        ones = torch.ones_like(scores) # (n_batches, len_sequences, len_sequences, d_model/heads)
        mask = torch.triu(ones, diagonal=1)
        mask = mask.to(torch.bool)
        return scores.masked_fill(mask, self.IGNORE) 
    

class Cov2DHead(nn.Module):
    def __init__(self, d_model, admixture_components, eps=1e-6):
        super().__init__()
        self.proj = nn.Linear(d_model, 3 * admixture_components)
        self.eps = eps
        self.admixture_components = admixture_components

    def forward(self, x):
        sigmas = self.proj(x) # (batches, len_sequence, 3 * admixture_components)
        B, T, O = sigmas.shape
        sigmas = sigmas.view(B, T, self.admixture_components, 3) # (batches, len_sequence, admixture_components, 3)

        a11 = sigmas[:, :, :, 0]
        a22 = sigmas[:, :, :, 1]
        a21 = sigmas[:, :, :, 2]

        a11 = F.softplus(a11) + self.eps
        a22 = F.softplus(a22) + self.eps

        L = torch.zeros([B, T, self.admixture_components, 2, 2], device=x.device, dtype=x.dtype) # (batches, len_sequence, admixture_components, 2, 2)
        L[:, :, :, 0, 0] = a11
        L[:, :, :, 1, 1] = a22
        L[:, :, :, 1, 0] = a21

        Sigma = L @ L.transpose(-1, -2)
        return Sigma # (batches, len_sequence, admixture_components, 2, 2)


class CovSaccHead(nn.Module):
    def __init__(self, d_model, admixture_components, eps=1e-6):
        super().__init__()
        self.proj = nn.Linear(d_model, admixture_components)
        self.eps = eps
        self.admixture_components = admixture_components

    def forward(self, x):
        sigmas = self.proj(x) # (batches, len_sequence, admixture_components)

        sigmas = F.softplus(sigmas) + self.eps

        return sigmas # (batches, len_sequence, admixture_components)
