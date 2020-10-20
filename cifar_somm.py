# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 16:20:32 2020

@author: admin
"""


import torch
import matplotlib.pyplot as plt 
import numpy as np
import warnings
import scipy.io as sio
from torch import nn, optim
from torch.autograd import Variable
import scipy.io as scio
from torchvision.utils import save_image
import os
from tqdm import tqdm
from minisom import MiniSom


data_a = sio.loadmat('./data/result/feature_epoch_30_(loss0.5512).mat')
feature = data_a['feature']
data_b = sio.loadmat('./data/result/label_epoch_30_(loss348.1304).mat')
target = data_b['label'].transpose(1,0).squeeze()

label_names = {0:'airplane', 1:'car', 2:'bird', 3:'cat', 4:'deer',
                5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}

n_neurons = 10
m_neurons = 10
som = MiniSom(n_neurons, m_neurons, feature.shape[1], sigma=1.5, learning_rate=.5, 
              neighborhood_function='gaussian', activation_distance='cosine', random_seed=0)

som.pca_weights_init(feature)
som.train(feature, 5000, verbose=True)  # random training


colors = ['k','r','peru','orange','y','g','dodgerblue','b','m','pink']

w_x, w_y = zip(*[som.winner(d) for d in feature])
w_x = np.array(w_x)
w_y = np.array(w_y)

plt.figure(figsize=(10, 9))
plt.pcolor(som.distance_map().T, cmap='bone_r', alpha=.2)
plt.colorbar()


for c in tqdm(np.unique(target)):
    idx_target = target==c
    plt.scatter(w_x[idx_target]+.5+(np.random.rand(np.sum(idx_target))-.5)*.8,
                w_y[idx_target]+.5+(np.random.rand(np.sum(idx_target))-.5)*.8, 
                s=50, c=colors[c-1], label=label_names[c])
plt.legend(loc='upper right',fontsize = 10)
plt.grid()
plt.savefig('./resulting_images/som_seed.png')
plt.show()

import matplotlib.gridspec as gridspec

labels_map = som.labels_map(feature, [label_names[t] for t in target])

fig = plt.figure(figsize=(9, 9))
the_grid = gridspec.GridSpec(n_neurons, m_neurons, fig)
for position in labels_map.keys():
    label_fracs = [labels_map[position][l] for l in label_names.values()]
    plt.subplot(the_grid[n_neurons-1-position[1],
                          position[0]], aspect=1)
    patches, texts = plt.pie(label_fracs)

plt.legend(patches, label_names.values(), bbox_to_anchor=(3.5, 6.5), ncol=3)
plt.savefig('resulting_images/som_seed_pies.png')
plt.show()