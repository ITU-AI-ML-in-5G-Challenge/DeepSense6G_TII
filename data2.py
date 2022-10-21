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
        data['beamidx'] = []
        data['radars'] = []
        data['idx'] = index
        PT=[]
        file_sep = '/'
        add_fronts = self.dataframe['unit1_rgb_5'][index]
        add_lidars = self.dataframe['unit1_lidar_5'][index]
        add_radars = self.dataframe['unit1_radar_5'][index]
        add_radars_split = add_radars.split(file_sep)
        #add_radars_split[-2]=add_radars_split[-2]+'_ang'
        add_radars_split[-2]=add_radars_split[-2]+'_vel'
        #add_radars_split[-2]=add_radars_split[-2]+'_cube'

        add_radars = file_sep.join(add_radars_split)
        
        beamidx=self.dataframe['unit1_beam'][index]-1
        x_data = range(max(beamidx-5,0),min(beamidx+5,63)+1)
        y_data = stats.norm.pdf(x_data, beamidx, 0.5)
        data_beam=np.zeros((64))
        data_beam[x_data]=y_data*1.25
#         print(y_data*1.25)
        
        
        
        for i in range(self.seq_len):
            data['fronts'].append(torch.from_numpy(np.transpose(np.array(Image.open(self.root+add_fronts).resize((256,256))),(2,0,1))))
            data['radars'].append(torch.from_numpy(np.expand_dims(np.load(self.root+add_radars),0)))
            

            PT = np.asarray(o3d.io.read_point_cloud(self.root+add_lidars).points)
            PT = lidar_to_histogram_features(PT)
            data['lidars'].append(PT)
            data['beam'].append(data_beam)
            data['beamidx'].append(beamidx)
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
        data['radars'] = []
        PT=[]
        file_sep = '/'
        add_fronts = self.dataframe['unit1_rgb_5'][index]
        add_lidars = self.dataframe['unit1_lidar_5'][index]
        add_radars = self.dataframe['unit1_radar_5'][index]
        add_radars_split = add_radars.split(file_sep)
        #add_radars_split[-2]=add_radars_split[-2]+'_ang'
        add_radars_split[-2]=add_radars_split[-2]+'_vel'
        #add_radars_split[-2]=add_radars_split[-2]+'_cube'
        add_radars = file_sep.join(add_radars_split)
        
        for i in range(self.seq_len):
            data['fronts'].append(torch.from_numpy(np.transpose(np.array(Image.open(self.root+add_fronts).resize((256,256))),(2,0,1))))
            data['radars'].append(torch.from_numpy(np.expand_dims(np.load(self.root+add_radars),0)))
            

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
        x_meters_max = 50
        y_meters_max = 50
        xbins = np.linspace(-x_meters_max, x_meters_max+1, 257)
        ybins = np.linspace(-y_meters_max, y_meters_max, 257)
        hist = np.histogramdd(point_cloud[...,:2], bins=(xbins, ybins))[0]
        hist[hist>hist_max_per_pixel] = hist_max_per_pixel
        overhead_splat = hist/hist_max_per_pixel
        return overhead_splat
    idx=lidar[:,2]<2
    lidar = lidar[idx, :]

    below = lidar[lidar[...,2]<=-2.0]
    above = lidar[lidar[...,2]>-2.0]
    below_features = splat_points(below)
    above_features = splat_points(above)
    features = np.stack([below_features, above_features], axis=-1)
    features = np.transpose(features, (2, 0, 1)).astype(np.float32)
    return features


def scale_and_crop_image(image, scale=1, crop=256):
    """
    Scale and crop a PIL image, returning a channels-first numpy array.
    """
    # image = Image.open(filename)
    (width, height) = (int(image.width // scale), int(image.height // scale))
    im_resized = image.resize((width, height))
    image = np.asarray(im_resized)
    start_x = height//2 - crop//2
    start_y = width//2 - crop//2
    cropped_image = image[start_x:start_x+crop, start_y:start_y+crop]
    cropped_image = np.transpose(cropped_image, (2,0,1))
    return cropped_image


def transform_2d_points(xyz, r1, t1_x, t1_y, r2, t2_x, t2_y):
    """
    Build a rotation matrix and take the dot product.
    """
    # z value to 1 for rotation
    xy1 = xyz.copy()
    xy1[:,2] = 1

    c, s = np.cos(r1), np.sin(r1)
    r1_to_world = np.matrix([[c, s, t1_x], [-s, c, t1_y], [0, 0, 1]])

    # np.dot converts to a matrix, so we explicitly change it back to an array
    world = np.asarray(r1_to_world @ xy1.T)

    c, s = np.cos(r2), np.sin(r2)
    r2_to_world = np.matrix([[c, s, t2_x], [-s, c, t2_y], [0, 0, 1]])
    world_to_r2 = np.linalg.inv(r2_to_world)

    out = np.asarray(world_to_r2 @ world).T
    
    # reset z-coordinate
    out[:,2] = xyz[:,2]

    return out
