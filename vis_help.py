import time
import pandas as pd
import matplotlib.patheffects as PathEffects
#matplotlib inline
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import torch
import networks

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']



def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(10):
        #ax = fig.add_subplot(111, projection='3d')
        inds = np.where(targets==i)[0]
        ax.scatter(embeddings[inds,0], embeddings[inds,1], embeddings[inds,2], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(mnist_classes)

def extract_embeddings(dataloader, model, pretrain):
    with torch.no_grad():
        model.eval()
        embeddings_1 = np.zeros((len(dataloader.dataset), networks.vis_size))
        embeddings_2 = np.zeros((len(dataloader.dataset), networks.vis_size))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            
            images = images.cuda()
            emb_1, emb_2= model.get_embedding(images, pretrain)
            emb_1, emb_2 = emb_1.cpu(), emb_2.cpu()
            embeddings_1[k:k+len(images)] = emb_1
            embeddings_2[k:k+len(images)] = emb_2
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings_1, embeddings_2, labels