import scipy.stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from matplotlib.font_manager import FontProperties

import os


def display_scores(results, dir_path, type, len='24'):
   mean = np.mean(results)
   sigma = scipy.stats.sem(results)
   sigma = sigma * scipy.stats.t.ppf((1 + 0.95) / 2., 5-1)
   print('Final Score: ', f'{mean} \xB1 {sigma}')
   os.makedirs(dir_path, exist_ok=True)
   file_path = os.path.join(dir_path, 'result.txt')
   with open(file_path, "a") as f:
        f.write(f'Final length of {len} {type} Score:  {mean} \xB1 {sigma} \n')


def train_test_divide (data_x, data_x_hat, data_t, data_t_hat, train_rate=0.8):
  
  no = len(data_x)
  idx = np.random.permutation(no)
  train_idx = idx[:int(no*train_rate)]
  test_idx = idx[int(no*train_rate):]
    
  train_x = [data_x[i] for i in train_idx]
  test_x = [data_x[i] for i in test_idx]
  train_t = [data_t[i] for i in train_idx]
  test_t = [data_t[i] for i in test_idx]      
    
  no = len(data_x_hat)
  idx = np.random.permutation(no)
  train_idx = idx[:int(no*train_rate)]
  test_idx = idx[int(no*train_rate):]
  
  train_x_hat = [data_x_hat[i] for i in train_idx]
  test_x_hat = [data_x_hat[i] for i in test_idx]
  train_t_hat = [data_t_hat[i] for i in train_idx]
  test_t_hat = [data_t_hat[i] for i in test_idx]
  
  return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


def extract_time (data):
  
  time = list()
  max_seq_len = 0
  for i in range(len(data)):
    max_seq_len = max(max_seq_len, len(data[i][:,0]))
    time.append(len(data[i][:,0]))
    
  return time, max_seq_len


def visualization(ori_data, generated_data, analysis, compare=3000, pic_path='toy_exp/pic'):
    
    anal_sample_no = min([compare, ori_data.shape[0]])
    idx = np.random.permutation(ori_data.shape[0])[:anal_sample_no]


    ori_data = ori_data[idx]
    generated_data = generated_data[idx]

    no, seq_len, dim = ori_data.shape

    steps = 10
    anal_sample_num = anal_sample_no // steps + 1

    for i in range(0, anal_sample_no, steps):
        if (i == 0):
            prep_data = np.reshape(np.mean(ori_data[0, :, :], 1), [1, seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data[0, :, :], 1), [1, seq_len])
        else:
            prep_data = np.concatenate((prep_data,
                                        np.reshape(np.mean(ori_data[i, :, :], 1), [1, seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat,
                                            np.reshape(np.mean(generated_data[i, :, :], 1), [1, seq_len])))

    red_rgba = to_rgba("red", alpha=0.2)
    blue_rgba = to_rgba("blue", alpha=0.2)

    colors = [red_rgba] * anal_sample_num + [blue_rgba] * anal_sample_num 

    font_path = 'Fonts/times.ttf'
    font_prop = FontProperties(fname=font_path, size=22)
    xy_font_prop = FontProperties(fname=font_path, size=20)

    if analysis == 'pca':
        pca = PCA(n_components=2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)

        f, ax = plt.subplots(1)
        plt.scatter(pca_results[:, 0], pca_results[:, 1],
                    c=colors[:anal_sample_num], label="Original")
        plt.scatter(pca_hat_results[:, 0], pca_hat_results[:, 1],
                    c=colors[anal_sample_num:], label="Ours")
        
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontproperties(xy_font_prop)

        ax.legend(prop=font_prop, loc='best')
        plt.savefig(f'{pic_path}/PCA.svg', bbox_inches='tight')
        plt.savefig(f'{pic_path}/PCA.png', bbox_inches='tight')
        plt.close()

    elif analysis == 'tsne':
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(prep_data_final)

        f, ax = plt.subplots(1)
        plt.scatter(tsne_results[:anal_sample_num, 0], tsne_results[:anal_sample_num, 1],
                    c=colors[:anal_sample_num], label="Original")
        plt.scatter(tsne_results[anal_sample_num:, 0], tsne_results[anal_sample_num:, 1],
                    c=colors[anal_sample_num:], label="Synthetic")

        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontproperties(xy_font_prop)

        ax.legend(prop=font_prop, loc='best')
        # plt.title('t-SNE plot', fontproperties=font_prop)
        # plt.xlabel('x-tsne', fontproperties=font_prop)
        # plt.ylabel('y_tsne', fontproperties=font_prop)
        plt.savefig(f'{pic_path}/t-SNE.svg', bbox_inches='tight')
        plt.savefig(f'{pic_path}/t-SNE.png', bbox_inches='tight')
        plt.show()
        plt.close()

    elif analysis == 'kernel':
        f, ax = plt.subplots(1)
        sns.kdeplot(prep_data.reshape(-1), label='Original', linewidth=3, color="Orange")
        sns.kdeplot(prep_data_hat.reshape(-1), label='Synthetic', linewidth=3, linestyle='--', color="blue")

        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontproperties(xy_font_prop)

        ax.legend(prop=font_prop, loc='best')
        plt.savefig(f'{pic_path}/Value.svg', bbox_inches='tight')
        plt.savefig(f'{pic_path}/Value.png', bbox_inches='tight')
        plt.show()
        plt.close()

if __name__ == '__main__':
   pass