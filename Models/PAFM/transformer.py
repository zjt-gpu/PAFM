import math
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from einops import rearrange, reduce, repeat
from Models.PAFM.model_utils import LearnablePositionalEncoding, Conv_MLP,\
                                                       AdaLayerNorm, GELU2

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale

class CrossAttention(nn.Module):
    def __init__(self,
                 n_embd,       
                 condition_embd,    
                 n_head,            
                 attn_pdrop=0.1,     
                 resid_pdrop=0.1,    
                 max_len=None):     
        super().__init__()

        self.n_head = n_head
        self.head_dim = n_embd // n_head

        # linear projections
        self.query = nn.Linear(n_embd, n_embd)
        self.key = nn.Linear(condition_embd, n_embd)
        self.value = nn.Linear(condition_embd, n_embd)

        # normalization
        self.q_norm = RMSNorm(n_embd)
        self.k_norm = RMSNorm(n_embd)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x, encoder_output, mask=None):
        
        B, T, C = x.size()
        T_enc = encoder_output.size(1)

        q = self.query(x)                
        k = self.key(encoder_output)     
        v = self.value(encoder_output)   

        q = self.q_norm(q) + q
        k = self.k_norm(k) + k

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)     
        k = k.view(B, T_enc, self.n_head, self.head_dim).transpose(1, 2) 
        v = v.view(B, T_enc, self.n_head, self.head_dim).transpose(1, 2)  

        scale = self.head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale       

        attn = F.softmax(attn_scores, dim=-1)                              
        attn = self.attn_drop(attn)

        y = torch.matmul(attn, v)                                          
        y = y.transpose(1, 2).contiguous().view(B, T, C)               
        y = self.resid_drop(self.proj(y))

        attn_mean = attn.mean(dim=1)  

        return y, attn_mean
    
class MoEFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts=4, dropout=0.1, activate='GELU'):
        super().__init__()
        self.num_experts = num_experts
        self.dropout = nn.Dropout(dropout)

        act_fn = {
            'GELU': nn.GELU(),
            'RELU': nn.ReLU(),
            'SiLU': nn.SiLU()
        }
        act = act_fn[activate] if isinstance(activate, str) else activate

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                act,
                nn.Linear(hidden_dim, input_dim)
            ) for _ in range(num_experts)
        ])

        self.gate = nn.Linear(input_dim, num_experts)

        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        B, T, D = x.shape
        gate_logits = self.gate(x) 
        top2_vals, top2_idx = torch.topk(gate_logits, k=2, dim=-1) 
        gate_scores = torch.softmax(top2_vals, dim=-1)  

        output = torch.zeros_like(x)
        load_count = torch.zeros(self.num_experts, device=x.device)

        for i in range(2):
            idx = top2_idx[:, :, i]   
            weight = gate_scores[:, :, i].unsqueeze(-1)  

            for expert_id, expert in enumerate(self.experts):
                mask = (idx == expert_id).unsqueeze(-1).float() 
                if mask.sum() == 0:
                    continue

                routed_input = x * mask
                expert_out = expert(routed_input) * weight * mask
                output += expert_out
                load_count[expert_id] += mask.sum()

        output = self.dropout(output)
        return self.norm(x + output)

class DecoderBlock(nn.Module):
    def __init__(self,
                 n_channel,
                 n_feat,
                 n_embd=1024,
                 n_head=16,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 mlp_hidden_times=4,
                 activate='GELU',
                 condition_dim=1024,
                 max_len = None
                 ):
        super().__init__()
        
        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = AdaLayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)

        self.attn1 = FullAttention(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop, 
                resid_pdrop=resid_pdrop
                )
        
        self.attn2 = CrossAttention(
                n_embd=n_embd,
                condition_embd=condition_dim,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                max_len = max_len
                )
        
        self.mlp = MoEFeedForward(n_embd, mlp_hidden_times * n_embd, num_experts=4, dropout=resid_pdrop, activate=activate)

    def forward(self, x, encoder_output, timestep, mask=None, label_emb=None):
        a, _ = self.attn1(self.ln1(x, timestep, label_emb), mask=mask)
        x = x + a

        a, _ = self.attn2(self.ln2(x, timestep), encoder_output, mask=mask)
        x = x + a

        x = x + self.mlp(self.ln3(x))

        return x

class Decoder(nn.Module):
    def __init__(
        self,
        n_channel,
        n_feat,
        n_embd=1024,
        n_head=16,
        n_layer=10,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        mlp_hidden_times=4,
        block_activate='GELU',
        condition_dim=512,
        max_len = None
    ):
      super().__init__()
      self.d_model = n_embd
      self.n_feat = n_feat
      self.blocks = nn.Sequential(*[DecoderBlock(
                n_feat=n_feat,
                n_channel=n_channel,
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                mlp_hidden_times=mlp_hidden_times,
                activate=block_activate,
                condition_dim=condition_dim,
                max_len = max_len
        ) for _ in range(n_layer)])
      
    def forward(self, x, t, enc, padding_masks=None, label_emb=None):
        b, c, _ = x.shape
        for block_idx in range(len(self.blocks)):
            x = self.blocks[block_idx](x, enc, t, mask=padding_masks, label_emb=label_emb)
        
        return x

class FullAttention(nn.Module):
    def __init__(self,
                 n_embd,
                 n_head,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1):
        super().__init__()
        assert n_embd % n_head == 0

        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.query = nn.Linear(n_embd, n_embd)
        self.key = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

    def forward(self, x, mask=None):
        B, T, C = x.size()

        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        y = self.proj(attn_output)
        y = self.resid_drop(y)

        attn_avg = attn_weights.mean(dim=1)
        return y, attn_avg

class EncoderBlock(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self,
                 n_embd=1024,
                 n_head=16,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 mlp_hidden_times=4,
                 activate='GELU'
                 ):
        super().__init__()

        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = FullAttention(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
            )
        
        assert activate in ['GELU', 'GELU2']
        act = nn.GELU() if activate == 'GELU' else GELU2()

        self.mlp = nn.Sequential(
                nn.Linear(n_embd, mlp_hidden_times * n_embd),
                act,
                nn.Linear(mlp_hidden_times * n_embd, n_embd),
                nn.Dropout(resid_pdrop),
            )
        
    def forward(self, x, timestep, mask=None, label_emb=None):
        a, att = self.attn(self.ln1(x, timestep, label_emb), mask=mask)
        x = x + a
        x = x + self.mlp(self.ln2(x))   # only one really use encoder_output
        return x, att


class Encoder(nn.Module):
    def __init__(
        self,
        n_layer=14,
        n_embd=1024,
        n_head=16,
        attn_pdrop=0.,
        resid_pdrop=0.,
        mlp_hidden_times=4,
        block_activate='GELU',
    ):
        super().__init__()

        self.blocks = nn.Sequential(*[EncoderBlock(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                mlp_hidden_times=mlp_hidden_times,
                activate=block_activate,
        ) for _ in range(n_layer)])

    def forward(self, input, t, padding_masks=None, label_emb=None):
        x = input
        for block_idx in range(len(self.blocks)):
            x, _ = self.blocks[block_idx](x, t, mask=padding_masks, label_emb=label_emb)
        return x

class MoEBlock(nn.Module):
    def __init__(self, input_dim, expert_dim, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([nn.Linear(input_dim, expert_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, input_feat, delta):
        B, T, D = input_feat.shape

        gate_logits = self.gate(input_feat)
        topk_vals, topk_idx = torch.topk(gate_logits, self.top_k, dim=-1)

        topk_weights = F.softmax(topk_vals, dim=-1) 

        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(delta))  

        expert_outputs = torch.stack(expert_outputs, dim=-1)  

        topk_outputs = torch.gather(
            expert_outputs, 
            dim=-1, 
            index=topk_idx.unsqueeze(2).expand(-1, -1, D, -1)
        )  

        weighted_output = (topk_outputs * topk_weights.unsqueeze(2)).sum(dim=-1)  # (B, T, D)

        return weighted_output


class Transformer(nn.Module):
    def __init__(
        self,
        n_feat,
        n_channel,
        n_layer_enc=5,
        n_layer_dec=14,
        n_embd=1024,
        n_heads=16,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        mlp_hidden_times=4,
        block_activate='GELU',
        max_len=2048,
        conv_params=None,
        **kwargs
    ):
        super().__init__()
        self.emb = Conv_MLP(n_feat, n_embd, resid_pdrop=resid_pdrop)
        self.inverse = Conv_MLP(n_embd, n_feat, resid_pdrop=resid_pdrop)
        self.pos_enc = LearnablePositionalEncoding(n_embd, dropout=resid_pdrop, max_len=max_len)

        self.encoder = Encoder(n_layer_enc, n_embd, n_heads, attn_pdrop, resid_pdrop, mlp_hidden_times, block_activate)
        self.decoder = Decoder(n_channel, n_feat, n_embd, n_heads, n_layer_dec, attn_pdrop, resid_pdrop, mlp_hidden_times,
                               block_activate, condition_dim=n_embd, max_len = max_len)

        self.weight_mlp = nn.Sequential(
            nn.Linear(n_feat, n_feat),
            nn.Sigmoid() 
        )
        self.noise_sigma = 0.1
        self.noise_alpha = 0.1

    def forward(self, input, t, padding_masks=None, return_res=False):
        emb = self.emb(input)
        enc_cond = self.encoder(self.pos_enc(emb), t, padding_masks=padding_masks)

        noise = torch.randn_like(input) * self.noise_sigma
        input_noise = input + noise
        emb_noise = self.emb(input_noise)
        enc_noise = self.encoder(self.pos_enc(emb_noise), t, padding_masks=padding_masks)
        enc_decode = self.decoder(enc_cond, t, enc_noise, padding_masks=padding_masks)
        noise_decode = self.decoder(enc_noise, t, enc_noise, padding_masks=padding_masks)
        output = self.inverse(enc_decode)  
        noise_output = self.inverse(noise_decode)
        delta = noise_output - output

        weight = self.weight_mlp(output)
        a = delta * weight  
        return output + a * self.noise_alpha
        
if __name__ == '__main__':
    pass