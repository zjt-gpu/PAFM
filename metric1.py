import os
import torch
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname('__file__'), '../'))
from Utils.context_fid import Context_FID
from Utils.metric_utils import display_scores
from Utils.cross_correlation import CrossCorrelLoss

iterations = 5

folders = ['ETTh', 'fMRI', 'MuJoCo', 'Stocks', 'Energy']
names = {'ETTh': 'etth', 'fMRI': 'fMRI', 'MuJoCo': 'mujoco', 'Stocks': 'stock', 'Energy': 'energy'}
#folders = ['Sines']
#names = {'Sines': 'sine'}

for folder in folders:
    name = names[folder]
    #ori_data = np.load(f'OUTPUT/{folder}/samples/{name}_ground_truth_24_train.npy') Sines
    ori_data = np.load(f'OUTPUT/{folder}/samples/{name}_norm_truth_24_train.npy')
    fake_data = np.load(f'OUTPUT/{folder}/flow_fake_{folder}.npy')
    discriminative_score = []

    print(ori_data.shape, fake_data.shape)

    context_fid_score = []

    for i in range(iterations):
        context_fid = Context_FID(ori_data[:], fake_data[:ori_data.shape[0]])
        context_fid_score.append(context_fid)
        print(f'Iter {i}: ', 'context-fid =', context_fid, '\n')
        
    display_scores(context_fid_score, f'result/{folder}', 'context_id')

    def random_choice(size, num_select=100):
        select_idx = np.random.randint(low=0, high=size, size=(num_select,))
        return select_idx

    x_real = torch.from_numpy(ori_data)
    x_fake = torch.from_numpy(fake_data)

    correlational_score = []
    size = int(x_real.shape[0] / iterations)

    for i in range(iterations):
        real_idx = random_choice(x_real.shape[0], size)
        fake_idx = random_choice(x_fake.shape[0], size)
        corr = CrossCorrelLoss(x_real[real_idx, :, :], name='CrossCorrelLoss')
        loss = corr.compute(x_fake[fake_idx, :, :])
        correlational_score.append(loss.item())
        print(f'Iter {i}: ', 'cross-correlation =', loss.item(), '\n')

    display_scores(correlational_score, f'result/{folder}', 'correlational_id')
