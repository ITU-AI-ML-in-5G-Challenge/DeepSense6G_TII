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
import cv2

class CARLA_Data(Dataset):
    def __init__(self, root, root_csv, config, test=False):

        self.dataframe = pd.read_csv(root+root_csv)
        self.root=root
        self.seq_len = config.seq_len
        self.test=test
        self.add_mask = config.add_mask
        self.enhanced = config.enhanced
        self.add_velocity = config.add_velocity
    def __len__(self):
        """Returns the length of the dataset. """
        # return 50
        return self.dataframe.shape[0]


    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        data['fronts'] = []
        data['lidars'] = []

        data['radars'] = []
        data['idx'] = index
        PT=[]
        file_sep = '/'
        add_fronts = self.dataframe['unit1_rgb_5'][index]
        add_lidars = self.dataframe['unit1_lidar_5'][index]
        add_radars = self.dataframe['unit1_radar_5'][index]
        add_radars_split = add_radars.split(file_sep)
        add_radars_split[-2]=add_radars_split[-2]+'_ang'
        #add_radars_split[-2]=add_radars_split[-2]+'_vel'
        #add_radars_split[-2]=add_radars_split[-2]+'_cube'

        add_radars = file_sep.join(add_radars_split)

        for i in range(self.seq_len):
            if 'scenario31' in add_fronts or 'scenario32' in add_fronts:
                imgs = np.array(Image.open(self.root + add_fronts).resize((256, 256)))
                seg = np.array(Image.open(self.root+add_fronts[:30]+'_seg'+add_fronts[30:]).resize((256,256)))
                imgs = cv2.addWeighted(imgs, 0.8, seg, 0.2, 0)
            else:
                if self.add_mask & self.enhanced:
                    raise Exception("mask or enhance, both are not possible")
                if self.add_mask:
                    imgs = np.array(
                        Image.open(self.root + add_fronts[:30] + '_mask' + add_fronts[30:]).resize((256, 256)))
                elif self.enhanced:
                    imgs = np.array(
                        Image.open(self.root + add_fronts).resize((256, 256)))
                else:
                    imgs = np.array(Image.open(self.root + add_fronts[:30]+'_raw'+add_fronts[30:]).resize((256, 256)))

            data['fronts'].append(torch.from_numpy(np.transpose(imgs,(2,0,1))))
            add_radars_vel = add_radars.replace('ang','vel')
            radar_ang=np.expand_dims(np.load(self.root + add_radars), 0)
            if self.add_velocity:
                radar_vel = np.expand_dims(np.load(self.root + add_radars_vel), 0)
                data['radars'].append(torch.from_numpy(np.concatenate([radar_ang, radar_vel], 0)))
            else:
                data['radars'].append(torch.from_numpy(radar_ang))

            PT = np.asarray(o3d.io.read_point_cloud(self.root+add_lidars).points)
            PT = lidar_to_histogram_features(PT)
            data['lidars'].append(PT)
        if not self.test:
            data['beam'] = []
            data['beamidx'] = []
            beamidx=self.dataframe['unit1_beam'][index]-1
            x_data = range(max(beamidx-5,0),min(beamidx+5,63)+1)
            y_data = stats.norm.pdf(x_data, beamidx, 0.5)
            data_beam=np.zeros((64))
            data_beam[x_data]=y_data*1.25
            data['beam'].append(data_beam)
            data['beamidx'].append(beamidx)
        return data


def lidar_to_histogram_features(lidar, crop=256):
    """
    Convert LiDAR point cloud into 2-bin histogram over 256x256 grid
    """
    def splat_points(point_cloud):
        # 256 x 256 grid
        pixels_per_meter = 8
        hist_max_per_pixel = 5
        x_meters_max = 50
        y_meters_max = 50
        xbins = np.linspace(-x_meters_max, 0, 257)
        ybins = np.linspace(-y_meters_max, y_meters_max, 257)
        hist = np.histogramdd(point_cloud[...,:2], bins=(xbins, ybins))[0]
        hist[hist>hist_max_per_pixel] = hist_max_per_pixel
        overhead_splat = hist/hist_max_per_pixel
        return overhead_splat
    # idx=lidar[:,2]<2
    # lidar = lidar[idx, :]

    # below = lidar[lidar[...,2]<=-2.0]
    # above = lidar[lidar[...,2]>-2.0]
    # below_features = splat_points(below)
    # above_features = splat_points(above)
    # features = np.stack([below_features, above_features], axis=-1)
    # features = np.transpose(features, (2, 0, 1)).astype(np.float32)
    lidar_feature = splat_points(lidar)
    lidar_feature = lidar_feature[np.newaxis,:,:]
    return lidar_feature



