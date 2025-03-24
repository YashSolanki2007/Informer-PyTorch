import torch
from torch import nn
import math
from main import *

# Test function
def test_probsparse_attention():
    print("Testing ProbSparse Attention...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Define parameters
    batch_size = 2
    q_dim = 16      # Number of query elements
    kv_dim = 24     # Number of key/value elements
    d_model = 32    # Model dimension
    c = 5           # Sampling factor
    n_heads = 4     # Number of attention heads
    
    # Create random input tensors
    q = torch.randn(batch_size, q_dim, d_model // n_heads)
    k = torch.randn(batch_size, kv_dim, d_model // n_heads)
    v = torch.randn(batch_size, kv_dim, d_model // n_heads)
    
    print(f"Input shapes - Q: {q.shape}, K: {k.shape}, V: {v.shape}")
    
    # Test single attention head (without sharing KV)
    print("\nTesting single ProbSparse attention head (without share_kv)...")
    head = Probsparse_Attention_Head(q_dim, kv_dim, d_model, c, share_kv=False)
    output = head(q, k, v)
    print(f"Single head output shape: {output.shape}")
    
    # Test single attention head (with sharing KV)
    print("\nTesting single ProbSparse attention head (with share_kv)...")
    head_share_kv = Probsparse_Attention_Head(q_dim, kv_dim, d_model, c, share_kv=True)
    output_tuple = head_share_kv(q, k, v)
    print(f"Single head output type: {type(output_tuple)}")
    print(f"Single head output[0] shape: {output_tuple[0].shape}")
    print(f"Single head output[1] shape: {output_tuple[1].shape}")
    print(f"Single head output[2] shape: {output_tuple[2].shape}")
    
    # Test multi-head attention (without sharing KV)
    print("\nTesting multi-head ProbSparse attention (without share_kv)...")
    mha = MultiHead_Attention(n_heads, d_model, "ps", q_dim, kv_dim, c, share_kv=False)
    mha_output = mha(q, k, v)
    print(f"Multi-head output shape: {mha_output.shape}")
    
    # Test multi-head attention (with sharing KV)
    print("\nTesting multi-head ProbSparse attention (with share_kv)...")
    mha_share_kv = MultiHead_Attention(n_heads, d_model, "ps", q_dim, kv_dim, c, share_kv=True)
    mha_output_share_kv = mha_share_kv(q, k, v)
    print(f"Multi-head output shape (with share_kv): {mha_output_share_kv[0].shape}")
    

if __name__ == "__main__":
    test_probsparse_attention()