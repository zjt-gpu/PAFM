import os
import torch
import numpy as np

from tqdm.auto import tqdm
from torch.utils.data import Dataset

from Models.PAFM.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from Utils.masking_utils import noise_mask


class SineDataset(Dataset):
    def __init__(
        self, 
        window=128, 
        num=30000, 
        dim=12, 
        save2npy=True, 
        neg_one_to_one=True, 
        seed=123,
        period='train',
        output_dir='./OUTPUT',
        predict_length=None,
        missing_ratio=None,
        style='separate', 
        distribution='geometric', 
        mean_mask_length=3
    ):
        super(SineDataset, self).__init__()
        assert period in ['train', 'test'], 'period must be train or test.'
        if period == 'train':
            assert ~(predict_length is not None or missing_ratio is not None), ''

        self.pred_len, self.missing_ratio = predict_length, missing_ratio
        self.style, self.distribution, self.mean_mask_length = style, distribution, mean_mask_length

        self.dir = os.path.join(output_dir, 'samples')
        os.makedirs(self.dir, exist_ok=True)

        self.rawdata = self.sine_data_generation(no=num, seq_len=window, dim=dim, save2npy=save2npy, 
                                                 seed=seed, dir=self.dir, period=period)
        self.auto_norm = neg_one_to_one
        self.samples = self.normalize(self.rawdata)
        self.var_num = dim
        self.sample_num = self.samples.shape[0]
        self.window = window

        self.period, self.save2npy = period, save2npy
        if period == 'test':
            if missing_ratio is not None:
                self.masking = self.mask_data(seed)
            elif predict_length is not None:
                masks = np.ones(self.samples.shape)
                masks[:, -predict_length:, :] = 0
                self.masking = masks.astype(bool)
            else:
                raise NotImplementedError()

    def normalize(self, rawdata):
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(rawdata)
        return data

    def unnormalize(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        return data
    
    @staticmethod
    def sine_data_generation(no, seq_len, dim, save2npy=True, seed=123, dir="./", period='train'):
        st0 = np.random.get_state()
        np.random.seed(seed)

        data = list()
        for i in tqdm(range(0, no), total=no, desc="Sampling sine-dataset"):
            temp = list()
            for k in range(dim):
                freq = np.random.uniform(0, 0.1)            
                phase = np.random.uniform(0, 0.1)
          
                temp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
                temp.append(temp_data)
        
            temp = np.transpose(np.asarray(temp))
            temp = (temp + 1)*0.5
            data.append(temp)
            
        np.random.set_state(st0)
        data = np.array(data)
        if save2npy:
            np.save(os.path.join(dir, f"sine_ground_truth_{seq_len}_{period}.npy"), data) 
            np.save(os.path.join(dir, f"sine_norm_truth_{seq_len}_{period}.npy"), unnormalize_to_zero_to_one(data))

        return data
    
    def mask_data(self, seed=2023):
        masks = np.ones_like(self.samples)
        st0 = np.random.get_state()
        np.random.seed(seed)

        for idx in range(self.samples.shape[0]):
            x = self.samples[idx, :, :]  
            mask = noise_mask(x, self.missing_ratio, self.mean_mask_length, self.style,
                              self.distribution)  
            masks[idx, :, :] = mask

        if self.save2npy:
            np.save(os.path.join(self.dir, f"sine_masking_{self.window}.npy"), masks)

        np.random.set_state(st0)
        return masks.astype(bool)

    def __getitem__(self, ind):
        if self.period == 'test':
            x = self.samples[ind, :, :]  
            m = self.masking[ind, :, :]  
            return torch.from_numpy(x).float(), torch.from_numpy(m)
        x = self.samples[ind, :, :]  
        return torch.from_numpy(x).float()

    def __len__(self):
        return self.sample_num
