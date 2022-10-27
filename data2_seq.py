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
from sklearn.preprocessing import normalize
import utm
import cv2
import re

class CARLA_Data(Dataset):
    def __init__(self, root, root_csv, config, test=False, augment={'camera':0, 'lidar':0, 'radar':0}):

        self.dataframe = pd.read_csv(root+root_csv)
        self.root=root
        self.seq_len = config.seq_len
        self.gps_data = []
        self.pos_input_normalized = Normalize_loc(root,self.dataframe,angle_norm=config.angle_norm)
        self.test = test
        self.add_velocity = config.add_velocity
        self.add_mask = config.add_mask
        self.enhanced = config.enhanced
        self.filtered = config.filtered
        self.augment = augment
        self.custom_FoV_lidar = config.custom_FoV_lidar
        self.add_mask_seg = config.add_mask_seg

    def __len__(self):
        """Returns the length of the dataset. """
        return self.dataframe.shape[0]

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        data['fronts'] = []
        data['lidars'] = []

        data['radars'] = []
        data['gps'] = self.pos_input_normalized[index,:,:]
        data['scenario'] = []
        data['loss_weight'] = []

        PT=[]
        file_sep = '/'
        add_fronts = []
        add_lidars = []
        add_radars = []
        # instanceidx=['1','2','5']
        instanceidx=['1','2', '3', '4', '5']

        for stri in instanceidx:

            camera_dir = self.dataframe['unit1_rgb_'+stri][index]
            if self.augment['camera'] > 0: # and 'scenario31' in camera_dir:
                camera_dir = re.sub('camera_data/', 'camera_data_aug/', camera_dir)
                camera_dir = camera_dir[:-4] + '_' + str(self.augment['camera']) + '.jpg'
                add_fronts.append(camera_dir)
            else:
                add_fronts.append(self.dataframe['unit1_rgb_'+stri][index])

            lidar_dir = self.dataframe['unit1_lidar_'+stri][index]
            if self.augment['lidar'] > 0:  # and 'scenario31' in lidar_dir:
                lidar_dir = re.sub('lidar_data/', 'lidar_data_aug/', lidar_dir)
                lidar_dir = lidar_dir[:-4] + '_' + str(self.augment['lidar']) + '.ply'
                add_lidars.append(lidar_dir)
            elif self.filtered:     # for non augmented lidar, check if applying filtered
                lidar_dir = re.sub('lidar_data/', 'lidar_data_filtered/', lidar_dir)
                add_lidars.append(lidar_dir)
            else:
                add_lidars.append(self.dataframe['unit1_lidar_'+stri][index])

            radar_dir = self.dataframe['unit1_radar_'+stri][index]
            if self.augment['radar'] > 0:
                radar_dir = re.sub('radar_data/', 'radar_data_ang_aug/', radar_dir)
            else:
                radar_dir = re.sub('radar_data/', 'radar_data_ang/', radar_dir)
            add_radars.append(radar_dir)

            #add_radars.append(add_radars1[:29] + '_vel' + add_radars1[29:])
            #add_radars.append(add_radars1[:29] + '_cube' + add_radars1[29:])

        self.seq_len = len(instanceidx)

        # check which scenario is the data sample associated 
        scenarios = ['scenario31', 'scenario32', 'scenario33', 'scenario34']
        loss_weights = [1.0, 1.0, 1.0, 1.0]
        # loss_weights = [227.2727, 3.5804, 2.9112, 2.6824]

        for i in range(len(scenarios)):
            s = scenarios[i]
            if s in self.dataframe['unit1_rgb_5'][index]:
                data['scenario'] = s
                data['loss_weight'] = loss_weights[i]
                break

        for i in range(self.seq_len):
            if 'scenario31' in add_fronts[i] or 'scenario32' in add_fronts[i]:

                # if self.add_mask:
                #     imgs = np.array(
                #         Image.open(self.root + add_fronts[i][:30] + '_mask' + add_fronts[i][30:]).resize((256, 256)))
                # else:
                #     imgs = np.array(
                #         Image.open(self.root + add_fronts[i]).resize((256, 256)))
                # imgs = np.array(Image.open(self.root + add_fronts[i]).resize((256, 256)))
                # seg = np.array(Image.open(self.root+add_fronts[i][:30]+'_seg'+add_fronts[i][30:]).resize((256,256)))
                # imgs = cv2.addWeighted(imgs, 0.8, seg, 0.2, 0)

                if self.augment['camera'] == 0:  # segmentation added to non augmented data
                    if self.add_mask_seg:
                        imgs = np.array(
                            Image.open(self.root + add_fronts[i][:30] + '_mask' + add_fronts[i][30:]).resize(
                                (256, 256)))
                    else:
                        imgs = np.array(Image.open(self.root + add_fronts[i]).resize((256, 256)))
                        seg = np.array(Image.open(self.root+add_fronts[i][:30]+'_seg'+add_fronts[i][30:]).resize((256,256)))
                        imgs = cv2.addWeighted(imgs, 0.8, seg, 0.2, 0)

            else:
                if self.add_mask & self.enhanced:
                    raise Exception("mask or enhance, both are not possible")
                if self.add_mask:
                    imgs = np.array(
                        Image.open(self.root + add_fronts[i][:30] + '_mask' + add_fronts[i][30:]).resize((256, 256)))
                elif self.enhanced:
                    imgs = np.array(
                        Image.open(self.root + add_fronts[i]).resize((256, 256)))
                else:
                    imgs = np.array(Image.open(self.root + add_fronts[i][:30]+'_raw'+add_fronts[i][30:]).resize((256, 256)))


            data['fronts'].append(torch.from_numpy(np.transpose(np.array(Image.open(self.root+add_fronts[i]).resize((256,256))),(2,0,1))))
            radar_ang = np.expand_dims(np.load(self.root + add_radars[i]), 0)

            if self.add_velocity:
                radar_vel = np.expand_dims(np.load(self.root + add_radars[i].replace('ang','vel')), 0)
                data['radars'].append(torch.from_numpy(np.concatenate([radar_ang, radar_vel], 0)))
            else:
                data['radars'].append(torch.from_numpy(radar_ang))
            
            # data['radars'].append(torch.from_numpy(np.expand_dims(np.load(self.root+add_radars[i]),0)))
            #lidar data
            PT = np.asarray(o3d.io.read_point_cloud(self.root+add_lidars[i]).points)
            PT = lidar_to_histogram_features(PT, add_lidars[i],custom_FoV=self.custom_FoV_lidar)
            data['lidars'].append(PT)

        if not self.test:
            data['beam'] = []
            data['beamidx'] = []
            beamidx = self.dataframe['unit1_beam'][index] - 1
            x_data = range(max(beamidx - 5, 0), min(beamidx + 5, 63) + 1)
            y_data = stats.norm.pdf(x_data, beamidx, 0.5)
            data_beam = np.zeros((64))
            data_beam[x_data] = y_data * 1.25
            data['beam'].append(data_beam)
            data['beamidx'].append(beamidx)

        return data

        

def lidar_to_histogram_features(lidar, address,custom_FoV):
    """
    Convert LiDAR point cloud into 2-bin histogram over 256x256 grid
    """
    def splat_points(point_cloud,addr):
        # 256 x 256 grid
        pixels_per_meter = 8
        hist_max_per_pixel = 5
        x_meters_max = 50
        y_meters_max = 50
        xbins = np.linspace(-x_meters_max, 0, 257)
        ybins = np.linspace(-y_meters_max, y_meters_max, 257)
        if custom_FoV:
            if 'scenario31' in addr:
                xbins = np.linspace(-70, 0, 257)
                ybins = np.linspace(-25, 14, 257)
            elif 'scenario32' in addr:
                xbins = np.linspace(-60, 0, 257)
                ybins = np.linspace(-40, 5.5, 257)
            elif 'scenario33' in addr:
                xbins = np.linspace(-50, 0, 257)
                ybins = np.linspace(-12, 7, 257)
            elif 'scenario34' in addr:
                xbins = np.linspace(-50, 0, 257)
                ybins = np.linspace(-20, 10, 257)

        hist = np.histogramdd(point_cloud[...,:2], bins=(xbins, ybins))[0]
        hist[hist>hist_max_per_pixel] = hist_max_per_pixel
        overhead_splat = hist/hist_max_per_pixel
        return overhead_splat

    lidar_feature = splat_points(lidar,address)
    lidar_feature = lidar_feature[np.newaxis, :, :]
    return lidar_feature

def xy_from_latlong(lat_long):
    """
    Requires lat and long, in decimal degrees, in the 1st and 2nd columns.
    Returns same row vec/matrix on cartesian (XY) coords.
    """
    # utm.from_latlon() returns: (EASTING, NORTHING, ZONE_NUMBER, ZONE_LETTER)
    x, y, *_ = utm.from_latlon(lat_long[:,0], lat_long[:,1])
    return np.stack((x,y), axis=1)


def Normalize_loc(root, dataframe,angle_norm):
    n_samples = dataframe.index.stop
    pos1_rel_paths = dataframe['unit2_loc_1'].values
    pos2_rel_paths = dataframe['unit2_loc_2'].values
    pos_bs_rel_paths = dataframe['unit1_loc'].values
    pos1_abs_paths = [os.path.join(root, path[2:]) for path in pos1_rel_paths]
    pos2_abs_paths = [os.path.join(root, path[2:]) for path in pos2_rel_paths]
    pos_bs_abs_paths = [os.path.join(root, path[2:]) for path in pos_bs_rel_paths]
    pos_input = np.zeros((n_samples, 2, 2))
    pos_bs = np.zeros((n_samples, 2))
    for sample_idx in tqdm(range(n_samples)):
        # unit2 (UE) positions
        pos_input[sample_idx, 0, :] = np.loadtxt(pos1_abs_paths[sample_idx])
        pos_input[sample_idx, 1, :] = np.loadtxt(pos2_abs_paths[sample_idx])
        # unit1 (BS) position
        pos_bs[sample_idx] = np.loadtxt(pos_bs_abs_paths[sample_idx])
    pos_ue_stacked = np.vstack((pos_input[:, 0, :], pos_input[:, 1, :]))
    pos_bs_stacked = np.vstack((pos_bs, pos_bs))
    pos_ue_stacked = np.vstack((pos_input[:, 0, :], pos_input[:, 1, :]))
    pos_bs_stacked = np.vstack((pos_bs, pos_bs))

    pos_ue_cart = xy_from_latlong(pos_ue_stacked)
    pos_bs_cart = xy_from_latlong(pos_bs_stacked)

    pos_diff = pos_ue_cart - pos_bs_cart

    # pos_min = np.min(pos_diff, axis=0)
    # pos_max = np.max(pos_diff, axis=0)
    pos_max = np.array([40.20955233, 52.31386139])
    pos_min = np.array([ -7.18029715, -97.55563452])

    # Normalize and unstack
    pos_stacked_normalized = (pos_diff - pos_min) / (pos_max - pos_min)
    if angle_norm:
        pos_stacked_normalized = normalize(pos_diff, axis=1)
    pos_input_normalized = np.zeros((n_samples, 2, 2))
    pos_input_normalized[:, 0, :] = pos_stacked_normalized[:n_samples]
    pos_input_normalized[:, 1, :] = pos_stacked_normalized[n_samples:]
    return pos_input_normalized