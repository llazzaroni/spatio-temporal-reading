import torch
import torch.nn as nn
import torch.nn.functional as F

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
        positions = torch.arange(T)
        distance = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs().float()  # (T, T)
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

        

