import os
import json
from PIL import Image
import pandas as pd

import numpy as np
import torch 
from torch.utils.data import Dataset
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import open3d as o3d
import torchvision.transforms as transforms
from scipy import stats


class CARLA_Data(Dataset):
    def __init__(self, root, root_csv, config):

        self.dataframe = pd.read_csv(root+root_csv)
        self.root=root
        self.seq_len = config.seq_len

    def __len__(self):
        """Returns the length of the dataset. """
        return self.dataframe.shape[0]

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        data['fronts'] = []
        data['lidars'] = []
        data['beam'] = []
        PT=[]
        

        add_fronts = self.dataframe['unit1_rgb_5'][index]
        file_sep = '/'
        add_fronts_split = add_fronts.split(file_sep)
        if '33' in add_fronts_split[1] or '34' in add_fronts_split[1]: 
            add_fronts_split[-2]=add_fronts_split[-2]+'_raw'
            add_fronts = file_sep.join(add_fronts_split)
        
        add_lidars = self.dataframe['unit1_lidar_5'][index]
        beamidx=self.dataframe['unit1_beam'][index]-1
        
        
        for i in range(self.seq_len):
            data['fronts'].append(torch.from_numpy(np.transpose(np.array(Image.open(self.root+add_fronts).resize((256,256))),(2,0,1))))
            

            PT = np.asarray(o3d.io.read_point_cloud(self.root+add_lidars).points)
            PT = lidar_to_histogram_features(PT)
            data['lidars'].append(PT)
            data['beam'].append(beamidx)
        return data
    
class CARLA_Data_Test(Dataset):
    def __init__(self, root, root_csv, config):

        self.dataframe = pd.read_csv(root+root_csv)
        self.root=root
        self.seq_len = config.seq_len

    def __len__(self):
        """Returns the length of the dataset. """
        return self.dataframe.shape[0]

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        data['fronts'] = []
        data['lidars'] = []
        data['idx'] = index
        PT=[]

        add_fronts = self.dataframe['unit1_rgb_5'][index]
        add_lidars = self.dataframe['unit1_lidar_5'][index]
        
        for i in range(self.seq_len):
            data['fronts'].append(torch.from_numpy(np.transpose(np.array(Image.open(self.root+add_fronts).resize((256,256))),(2,0,1))))
            

            PT = np.asarray(o3d.io.read_point_cloud(self.root+add_lidars).points)
            PT = lidar_to_histogram_features(PT)
            data['lidars'].append(PT)

        return data
def lidar_to_histogram_features(lidar, crop=256):
    """
    Convert LiDAR point cloud into 2-bin histogram over 256x256 grid
    """
    def splat_points(point_cloud):
        # 256 x 256 grid
        pixels_per_meter = 8
        hist_max_per_pixel = 5
        x_meters_max = 100
        y_meters_max = 100
        xbins = np.linspace(-x_meters_max, x_meters_max+1, 257)
        ybins = np.linspace(-y_meters_max, y_meters_max, 257)
        hist = np.histogramdd(point_cloud[...,:2], bins=(xbins, ybins))[0]
        hist[hist>hist_max_per_pixel] = hist_max_per_pixel
        overhead_splat = hist/hist_max_per_pixel
        return overhead_splat

    below = lidar[lidar[...,2]<=-2.0]
    above = lidar[lidar[...,2]>-2.0]
    below_features = splat_points(below)
    above_features = splat_points(above)
    features = np.stack([below_features, above_features], axis=-1)
    features = np.transpose(features, (2, 0, 1)).astype(np.float32)
    return features