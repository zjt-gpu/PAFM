import math
import torch
import torch.nn.functional as F
from torch import nn
from einops import reduce
from tqdm.auto import tqdm
from Models.PAFM.transformer import Transformer
import os

class FM_TS(nn.Module):
    def __init__(
            self,
            seq_length,
            feature_size,
            n_layer_enc=3,
            n_layer_dec=6,
            d_model=None,
            n_heads=4,
            mlp_hidden_times=4,
            attn_pd=0.,
            resid_pd=0.,
            kernel_size=None,
            padding_size=None,
            patch_len=4,
            stride=2,
            emb_drop=0.1,
            gpu=0,
            **kwargs
    ):
        super(FM_TS, self).__init__()

        self.seq_length = seq_length
        self.feature_size = feature_size

        self.model = Transformer(n_feat=feature_size, n_channel=seq_length, n_layer_enc=n_layer_enc, n_layer_dec=n_layer_dec,
                                 n_heads=n_heads, attn_pdrop=attn_pd, resid_pdrop=resid_pd, mlp_hidden_times=mlp_hidden_times,
                                 max_len=seq_length, n_embd=d_model, conv_params=[kernel_size, padding_size], patch_len=patch_len, stride=stride, emb_drop=emb_drop, **kwargs)
        
        self.device = torch.device(f"cuda:{gpu}")
        
        self.alpha = 3  
        self.time_scalar = 1000  

        self.num_timesteps = int(os.environ.get('hucfg_num_steps', '100'))

    def output(self, x, t, padding_masks=None):
        model_output = self.model(x, t, padding_masks=padding_masks)
        return model_output

    @torch.no_grad()
    def sample(self, shape):
        self.eval()

        zt = torch.randn(shape).to(self.device)#cuda()  
        
        timesteps = torch.linspace(0, 1, self.num_timesteps+1)
        t_shifted = 1-(self.alpha * timesteps) / (1 + (self.alpha - 1) * timesteps)
        t_shifted = t_shifted.flip(0)

        for t_curr, t_prev in zip(t_shifted[:-1], t_shifted[1:]):
            step = t_prev - t_curr
            v = self.output(zt.clone(), torch.tensor([t_curr*self.time_scalar]).unsqueeze(0).repeat(zt.shape[0], 1).to(self.device).squeeze(), padding_masks=None)                  
            zt = zt.clone() + step * v 

        return zt 

    def generate_mts(self, batch_size=16):
        feature_size, seq_length = self.feature_size, self.seq_length
        return self.sample((batch_size, seq_length, feature_size))
    
    def _train_loss(self, x_start):
        
        z0 = torch.randn_like(x_start) 
        z1 = x_start

        t = torch.rand(z0.shape[0], 1, 1).to(z0.device)
        if str(os.environ.get('hucfg_t_sampling', 'uniform')) == 'logitnorm':
            t = torch.sigmoid(torch.randn(z0.shape[0], 1, 1)).to(z0.device)
        z_t =  t * z1 + (1.-t) * z0

        target = z1 - z0

        model_out = self.output(z_t, t.squeeze()*self.time_scalar, None)
        train_loss = F.mse_loss(model_out, target, reduction='none')

        train_loss = reduce(train_loss, 'b ... -> b (...)', 'mean')
        train_loss = train_loss.mean()
        return train_loss.mean()

    def forward(self, x):
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        return self._train_loss(x_start=x)
    
    def fast_sample_infill(self, shape, target, partial_mask=None):

        z0 = torch.randn(shape).to(self.device)
        z1 = zt = z0
        for t in range(self.num_timesteps):
            target = target.to(self.device)
            t = t/self.num_timesteps 
            t = t**(float(os.environ.get('hucfg_Kscale', '1.0')))  

            z0 = torch.randn(shape).to(self.device)

            target_t = target*t + z0*(1-t) 
            zt = z1*t + z0*(1-t) 
            partial_mask = partial_mask.to(zt.device)

            zt[partial_mask] = target_t[partial_mask] 
            v = self.output(zt, torch.tensor([t*self.time_scalar]).to(self.device), None) 

            z1 = zt.clone() + (1 - t) * v  
            z1 = torch.clamp(z1, min=-1, max=1) 


        return z1






