import torch
from torch import nn 
import math


SEQUENCE_LENGTH = 24

class Probsparse_Attention_Head(nn.Module):
    def __init__(self, q_dim, kv_dim, d_model, c, share_kv=False, mask=None):
        super().__init__()
        self.q_dim = q_dim 
        self.kv_dim = kv_dim 
        self.d_model = d_model
        # Using notation from the paper (sampling factor ideally set to 5 in the paper) 
        self.c = c 
        self.share_kv = share_kv
        self.mask = mask
    
        self.register_buffer('tril', torch.tril(torch.ones(SEQUENCE_LENGTH, SEQUENCE_LENGTH))) 

    def _sampler(self, k):
        '''
        k is the forwarded versions of the linear layers defined above in the class
        Shape: (batch_size, kv_dim, d_model)
        '''
        batch_size = k.shape[0]
        u = int(self.c * math.log(self.q_dim))
        U = int(self.q_dim * math.log(self.kv_dim))
        
        # Randomly selecting 'U' dot-product pairs from keys for each batch
        k_bar_list = []
        for b in range(batch_size):
            # Select random indices for each batch
            index_sample = torch.randint(0, self.kv_dim, (U,))
            k_bar_list.append(k[b, index_sample, :])
        
        k_bar = torch.stack(k_bar_list)
        return k_bar, u

    def forward(self, q, k, v):
        batch_size = q.shape[0]
        
        # Getting the sample score S_bar 
        k_bar, u = self._sampler(k)  
    
        s_bar = torch.bmm(q, k_bar.transpose(1, 2))
        M = torch.max(s_bar, 2).values - torch.mean(s_bar, 2)  # Shape: (batch_size, q_dim)
        
        # Initialize output with zeros
        s = torch.zeros_like(q) 

        for b in range(batch_size):
            # Top-u sampling of the queries for each batch
            top_u_indices = torch.topk(M[b], u, sorted=False)[1]
            q_bar = q[b, top_u_indices, :]  # Shape: (u, d_model)
            
            # Performing attention computation
            attn_weights = (q_bar @ k[b].transpose(0, 1)) / (k[b].shape[0] ** 0.5)
            attn_weights = torch.softmax(attn_weights, dim=-1)

            s_1 = attn_weights @ v[b]
            
            # Place S1 in the corresponding positions
            s[b, top_u_indices, :] = s_1
            
            # Get indices not in top_u
            all_indices = torch.arange(q.shape[1], device=q.device)
            mask = torch.ones(q.shape[1], dtype=torch.bool, device=q.device)
            mask[top_u_indices] = False
            remaining_indices = all_indices[mask]
            
            # Compute S0 (mean of values) for remaining positions
            s_0 = torch.mean(v[b], dim=0, keepdim=True)  # Shape: (1, d_model)
            s_0 = s_0.expand(len(remaining_indices), -1)  # Shape: (remaining_count, d_model)
            s[b, remaining_indices, :] = s_0
        
        return s if not self.share_kv else (s, k, v)


class Self_Attention_Head(nn.Module):
    def __init__(self, q_dim, kv_dim, d_model, receive_kv=False):
        super().__init__()
        self.q_dim = q_dim 
        self.kv_dim = kv_dim 
        self.d_model = d_model
        self.receive_kv = receive_kv
        
        self.queries = nn.Linear(self.q_dim, self.d_model)
        self.keys = nn.Linear(self.kv_dim, self.d_model) 
        self.values = nn.Linear(self.kv_dim, self.d_model)

    def forward(self, x, k=None, v=None):
        # x: (batch_size, seq_len, q_dim)
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        q = self.queries(x)  # (batch_size, seq_len, d_model)
        if not self.receive_kv:
            k = self.keys(x)  
            v = self.values(x) 
        
        attn_weights = torch.bmm(q, k.transpose(1, 2)) / (k.shape[1] ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        attn_scores = torch.bmm(attn_weights, v)
        
        return attn_scores


class MultiHead_Attention(nn.Module):
    def __init__ (self, n_heads, d_model, head_type, q_dim, kv_dim, c, receive_kv=False, share_kv=False):
        super().__init__()
        self.n_heads = n_heads 
        self.d_model = d_model 
        self.head_type = head_type    # Can be Self_Attention_Head or Prob_Sparse_Attention_Head
        self.head_size = self.d_model // self.n_heads
        self.q_dim = q_dim 
        self.kv_dim = kv_dim 
        self.receive_kv = receive_kv
        self.c = c 
        self.share_kv = share_kv

        # Initialize either the self attention or the prob-sparse attention 
        if self.head_type == "sa":
            # Getting the 'x' 
            self.heads = nn.ModuleList([Self_Attention_Head(self.q_dim, self.kv_dim, self.head_size, self.receive_kv) for _ in range(self.n_heads)])
        else:
            self.heads = nn.ModuleList([Probsparse_Attention_Head(self.q_dim, self.kv_dim, self.head_size, self.c, self.share_kv) for _ in range(self.n_heads)])

        self.projection = nn.Linear(self.n_heads * self.head_size, self.d_model)

    def forward(self, *args):
        # Conditions to be checked for 
        '''
        1. Self attention layer Normal 
        2. Self attention layer receive kv 
        3. Prob-Sparse layer Normal 
        4. Prob-Sparse layer share kv
        '''
        attn_out = None
        if self.head_type == "sa" and not self.receive_kv:
            x = args[0]  
            outputs = [head(x) for head in self.heads]  
            attn_out = torch.cat(outputs, dim=-1)  
            return self.projection(attn_out)
        if self.head_type == "sa" and self.receive_kv:
            x, k, v = args[0], args[1], args[2]
            outputs = [head(x, k, v) for head in self.heads]  
            attn_out = torch.cat(outputs, dim=-1)  
            return self.projection(attn_out) 
        if self.head_type != "sa" and not self.share_kv:
            q, k, v = args[0], args[1], args[2]
            outputs = [head(q, k, v) for head in self.heads]
            attn_out = torch.cat(outputs, dim=-1)
            return self.projection(attn_out)
        if self.head_type != "sa" and self.share_kv:
            q, k, v = args[0], args[1], args[2]
            outputs = [head(q, k, v) for head in self.heads]
            output_s = [outputs[i][0] for i in range(len(outputs))]
            attn_out = torch.cat(output_s, dim=-1)  
            return self.projection(attn_out), k, v
        


class FeedForward(nn.Module):
    def __init__(self, fan_in, ff_dim, fan_out):
        super().__init__()
        self.fan_in = fan_in
        # As per the paper the ff_dim = 4 * fan_in
        self.ff_dim = ff_dim 
        self.fan_out = fan_out

        self.l1 = nn.ReLU(nn.Linear(self.fan_in, self.ff_dim))
        self.l2 = nn.Linear(self.ff_dim, self.fan_out)

    def forward(self, x):   
        return self.l2(self.l1(x))



class PositionEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model 

        position_embed = torch.zeros(SEQUENCE_LENGTH, self.d_model)
        for pos in range(len(position_embed)):
            for i in range(len(position_embed[pos])):
                div_term = 10000 ** (2 * i / self.d_model)
                if i % 2 == 0:
                    position_embed[pos][i]  = math.sin(pos / div_term)
                else:
                    position_embed[pos][i]  = math.cos(pos / div_term)

        # Persists this in model training 
        self.register_buffer('position_embed', position_embed)

    
    def forward(self, token_embed):
        return token_embed + self.position_embed
    


