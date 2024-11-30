import torch
from torch import nn
from torch.nn import functional as F
import math

class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias = True, out_proj_bias = True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed,d_embed, bias = in_proj_bias)
        self.k_proj = nn.Linear(d_cross,d_embed, bias = in_proj_bias)
        self.v_proj = nn.Linear(d_cross,d_embed, bias = in_proj_bias)
        self.out_proj = nn.Linear(d_embed,d_embed,bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed//n_heads

    def forward(self, x, y):
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        q = q.view(interim_shape).transpose(1,2)
        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        k = k.view(interim_shape).transpose(1, 2) 
        v = v.view(interim_shape).transpose(1, 2) 
        #(Batch_size, H, seq_len_q, seq_len_kv)
        weight = q @ k.transpose(-1,-2)/math.sqrt(self.d_head)
        weight = F.softmax(weight,dim=-1)

        #(Batch_size, H, seq_len_q, Dim_Q/ H)
        output = weight @ v

        output = output.transpose(1,2).contiguous()

        output = output.view(input_shape)
        
        # (Batch_Size, Seq_Len_Q, Dim_Q)
        output = self.out_proj(output)

        return output


