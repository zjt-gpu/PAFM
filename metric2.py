import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.append(os.path.join(os.path.dirname('__file__'), '../'))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from Utils.metric_utils import display_scores
from Utils.discriminative_metric import discriminative_score_metrics
from Utils.predictive_metric import predictive_score_metrics

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

    for i in range(iterations):
        temp_disc, fake_acc, real_acc = discriminative_score_metrics(ori_data[:], fake_data[:ori_data.shape[0]])
        discriminative_score.append(temp_disc)
        print(f'Iter {i}: ', temp_disc, ',', fake_acc, ',', real_acc, '\n')
        
    print('sine:')
    display_scores(discriminative_score, f'result/{folder}', 'discriminative_id')
    print()

    predictive_score = []
    for i in range(iterations):
        temp_pred = predictive_score_metrics(ori_data, fake_data[:ori_data.shape[0]])
        predictive_score.append(temp_pred)
        print(i, ' epoch: ', temp_pred, '\n')
        
    print('sine:')
    display_scores(predictive_score, f'result/{folder}', 'predictive_id')
    print()