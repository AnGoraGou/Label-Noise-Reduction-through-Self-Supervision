import os
import glob
import torch
import math
import random
import argparse
import pandas as pd
from PIL import Image
from pathlib import Path
from byol_pytorch import BYOL
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from torchvision import transforms
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torch.nn as nn
import numpy as np



img_paths = glob.glob('/workspace/Data/solo_train/*/*.tif')  #sftp://subrat@10.107.47.139 /workspace/Data/solo_train_copy/  ####solo_train_copy/*/*.tif'
img_paths = './subrat/Data/solo_train/'
csv_file = './subrat/Data/Clean_train_data_encd.csv'
IMAGE_EXTS = ['.jpg', '.png', '.jpeg','.tif']
# print(f'Number of images: {len(img_paths)}')
BATCH_SIZE = 32
EPOCHS     = 15
LR         = 6e-2
NUM_GPUS   = 2
NUM_WORKERS = 2  #multiprocessing.cpu_count
n_label = 4
IMAGE_SIZE = 1024

tsne_path = "/workspace/subrat/tsne_dir_test/"

if not os.path.exists(tsne_path):
    os.mkdir(tsne_path)
    print(f"Directory created: {tsne_path}")


class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        self.learner = BYOL(net, **kwargs)
        # wandb.log({'Learner':self.learner})

    def forward(self, images):
        # print(f"learning parameters is {self.learner(images)}")
        return self.learner(images)

    def training_step(self, images, _):
        loss = self.forward(images)
        # print(f"Loss value is {loss}")
        # wandb.log({'Loss': loss})
        return {'loss': loss}



    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=LR)

    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()

for NUM_Clust in range(9):

    # Choose the number of clusters
    num_clusters = NUM_Clust+2
    print(f"The number of cluster is: {NUM_Clust}")

    img_features = np.array(img_features)

    # print(img_features)
    # exit()
    # Reshape the 1D array to 2D array
    # img_features = np.reshape(img_features, (img_features.shape[0], 1))

    # Run K-means algorithm
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(img_features)
    print(f'KKmeans is {kmeans}')

    # Evaluate the clusters
    wcss = kmeans.inertia_
    wcss_list.append(wcss)
    print(f"Within-cluster sum of squares is {wcss} for {num_clusters} clusters")

    # Visualize the clusters
    clust_labels = kmeans.labels_
    print(f' length of labels is {len(clust_labels)}')
    # print(f'hhhhhhhhhhhhhhhhhhhhhhhhhhhhhh is {labels}')
    centroids = kmeans.cluster_centers_
    # labels_list.append(labels)
    centroids_list.append(centroids) 
    #print(f'label is {labels}')
    #print(f'centroid is {centroids}')  

    # Evaluate the algorithm using the silhouette score
    silhouette_avg = silhouette_score(img_features, clust_labels)
    silhouette_list.append(silhouette_avg)
    print(f"The average silhouette score is {silhouette_avg} for {num_clusters} clusters")
############################################################################################################
    # # Apply t-SNE to reduce the dimensionality to 2D space
    # tsne = TSNE(n_components=2, perplexity=30.0, random_state=0)
    # X_tsne = tsne.fit_transform(img_features)

    # # Plot the t-SNE results with different colors for each cluster
    # plt.figure(figsize=(10, 8))
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clust_labels, cmap='rainbow')
    # plt.title(f't-SNE visualization with {num_clusters} clusters')
    # plt.xlabel('t-SNE component 1')
    # plt.ylabel('t-SNE component 2')
    # plt.colorbar()
    # # plt.show()
    # plt.savefig(os.path.join(tsne_path,("tsne_cluster_"+ str(NUM_Clust)+".jpg")))


    # # Plot the t-SNE results with different colors for each cluster
    # plt.figure(figsize=(10, 8))
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=label_list, cmap='rainbow')
    # plt.title(f't-SNE visualization with {num_clusters} clusters')
    # plt.xlabel('t-SNE component 1')
    # plt.ylabel('t-SNE component 2')
    # plt.colorbar()

    # plt.savefig(os.path.join(tsne_path,("tsne_GTlabel_"+ str(NUM_Clust)+".jpg")))


##############################################################################################################


# print(f'len of img {len(img_name_list)}')
# print(f'len of ground_truth label is  {len(label_list)}')
# print(f'len of cluster label {len(labels)}')
# print(labels)


    # # Plot the t-SNE results with different colors for each cluster


###  Block for pseudo labels from cluster label and ground truth
k_df = pd.DataFrame(list(zip(img_name_list, label_list, labels)),columns =['Name', 'gt_label','cl_label'])
k_df['Name'] = k_df['Name'].apply(lambda x: str(x).replace(',', '').replace('(', '').replace(')', '').replace("'", ""))
k_df['gt_label'] = k_df['gt_label'].apply(lambda x: x.numpy())
k_df['gt_label'] = k_df['gt_label'].apply(lambda x: str(x).replace('[', '').replace(']', ''))
k_df["ps_label"] = ""

empty_col = pd.DataFrame(['']*len(k_df))
k_df = k_df.insert(3, 'ps_label', empty_col)



cluster_list = k_df.cl_label.unique()
cluster_list = sorted(cluster_list)

print(cluster_list)
#df_k = k_df.loc[k_df['cl_label'] == 4]
#ps = df_k['gt_label'].mode()

#print(ps.values)

for cluster_ in cluster_list:
    print(f'Working on cluster: {cluster_}')
    cl_df = k_df.loc[k_df['cl_label'] == cluster_]  # here SUBRAT change cl_label to gt_label
    ps = cl_df['gt_label'].mode().values
    #print(ps[0])
    k_df.loc[k_df['cl_label'] == cluster_, 'ps_label'] = ps[0]
    # print(f'pseudo label for cluster {cluster_} is {ps_label_c}')

print(k_df)

k_df['noise'] = k_df['gt_label'] ==k_df['ps_label']


print(f"Value counts of noise: {k_df['noise'].value_counts()}")
