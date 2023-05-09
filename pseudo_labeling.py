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



# parser = argparse.ArgumentParser(description='byol-lightning-test')

# parser.add_argument('--image_folder', type=str, required = True,
#                        help='path to your folder of images for self-supervised learning')

# args = parser.parse_args()




# img_paths = glob.glob('/workspace/Data/solo_train/*/*.tif')  #sftp://subrat@10.107.47.139 /workspace/Data/solo_train_copy/  ####solo_train_copy/*/*.tif'
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
   
print(f'len of img {len(img_name_list)}')
print(f'len of ground_truth label is  {len(label_list)}')
print(f'len of cluster label {len(clust_labels)}')
# print(labels_)

# exit()

###  Block for pseudo labels from cluster label and ground truth
k_df = pd.DataFrame(list(zip(img_name_list, label_list, clust_labels)),columns =['Name', 'gt_label','cl_label'])
k_df['Name'] = k_df['Name'].apply(lambda x: str(x).replace(',', '').replace('(', '').replace(')', '').replace("'", ""))
k_df['gt_label'] = k_df['gt_label'].apply(lambda x: x.numpy())
k_df['gt_label'] = k_df['gt_label'].apply(lambda x: str(x).replace('[', '').replace(']', ''))
# k_df["ps_label"] = ""
print(k_df)

empty_col = pd.DataFrame(['']*len(k_df))
k_df = k_df.insert(3, 'ps_label', empty_col.to_numpy().flatten())

#k_df = pd.read_csv('subrat/data_manp.csv')
print(k_df)
# k_df = k_df.insert(3, column = "ps_label", value = '0')
k_df.to_csv('/subrat/Data/file1.csv')
exit()
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
