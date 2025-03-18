'''
Notes from the paper 

Use a fixed size window i.e a fixed number of days of historic data before a prediction is made 
Use the KL-Divergence to measure the similarity between the distributions 'p' and 'q' 
'''


import torch
from torch import nn 
import math

class Probsparse_Attention_Head(nn.Module):
	def __init__(self, q_dim, kv_dim, d_model, c, share_kv=False):
		super().__init__()
		self.q_dim = q_dim 
		self.kv_dim = kv_dim 
		self.d_model = d_model
		# Using notation from the paper (:TODO figure out what exactly 'c' is)
		self.c = c 
		self.share_kv = share_kv
		
		# Initializing the query, key and value vectors with the shape (L, d_model)
		self.queries = nn.Linear(self.q_dim, self.d_model) 
		self.keys = nn.Linear(self.kv_dim, self.d_model) 
		self.values = nn.Linear(self.kv_dim, self.d_model) 

	def _sampler(self, k):
		'''
		k is the forwarded versions of the linear layers defined above in the class
		'''
		u = self.c * math.log(self.q_dim)
		U = int(self.q_dim * math.log(self.kv_dim))
		
		# Randomly selecting 'U' dot-product pairs from self.keys as k_bar
		index_sample = torch.randint(0, self.kv_dim, (U,))
		k_bar = k[:, index_sample, :]
		return k_bar, u

	def forward(self, x):
		q = self.queries(x)
		k = self.keys(x)
		v = self.values(x)
		
		# Getting the sample score S_bar 
		k_bar, u = self._sampler(k)
		s_bar = torch.matmul(q, k_bar.transpose(1, 0))
		
		# Computing the 'M' value 
		M = torch.max(s_bar, 1).values - torch.mean(s_bar, 1)
		
		# Top-k sampling of the queries 
		top_u_indices = torch.topk(M, u, sorted=False)[1]
		q_bar = q[top_u_indices]
		
		# Performing attention computation (:TODO masking not implemented) 
		attn_weights = q_bar @ k.transpose(1, 0)
		attn_weights = torch.softmax(attn_weights, dim=-1)
		s_1 = atnn_weights @ v
		s_0 = torch.mean(v, dim=0).unsqueeze(0)
		s_0 = s_0.expand(q_bar.shape[0], -1)

		s = torch.zeros_like(q)  # Initialize output with zeros
		s[top_u_indices] = s_1  # Place S1 in the corresponding positions
		remaining_indices = torch.tensor([i for i in range(q.shape[0]) if i not in top_u_indices])
		s[remaining_indices] = s_0  # Place S0 for remaining rows

		return s if not self.share_kv else s, k, v


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
		q = self.queries(x)
		if not self.receive_kv:
			k = self.keys(x)
			v = self.values(x)
		attn_weights = q @ k.transpose(1, 0)
		attn_weights = torch.softmax(attn_weights, dim=-1)
		attn_scores = attn_weights @ v
		return attn_scores
			


class MultiHead_Attention(nn.Module):
	def __init__(self, n_heads, d_model, head_type, q_dim, kv_dim, receive_kv=False, c, share_kv=False):
		super().__init__()
		self.n_heads = n_heads 
		self.d_model = d_model 
		self.head_type = head_type	# Can be Self_Attention_Head or Prob_Sparse_Attention_Head
		self.head_size = self.d_model // self.n_heads
		self.q_dim = q_dim 
		self.kv_dim = kv_dim 	
		self.receive_kv = receive_kv
		self.c = c 
		self.share_kv = share_kv

		self.projection = nn.Linear(self.num_heads * self.head_size, sself.d_model)
		if self.head_type == "sa":
			self.heads = nn.ModuleList([Self_Attention_Head(self.q_dim, self.kv_dim, self.d_model, self.receive_kv) for _ in range(self.n_heads)])
		else:
			self.heads = nn.ModuleList([Probsparse_Attention_Head(self.q_dim, self.kv_dim, self.d_model, self.c, self.share_kv) for _ in range(self.n_heads)])
