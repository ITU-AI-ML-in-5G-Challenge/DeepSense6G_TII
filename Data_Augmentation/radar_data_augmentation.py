import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import random

def range_angle_map(data, fft_size=256):
    data = np.fft.fft(data, axis=1)  # Range FFT
    data -= np.mean(data, 2, keepdims=True)
    data = np.fft.fft(data, fft_size, axis=0)  # Angle FFT
    # data = np.fft.fftshift(data, axes=0)
    data = np.abs(data).sum(axis=2)  # Sum over velocity
    return data.T
def range_velocity_map(data, fft_size=256):
    data = np.fft.fft(data, axis=1)  # Range FFT
    # data -= np.mean(data, 2, keepdims=True)
    data = np.fft.fft(data, fft_size, axis=2)  # Velocity FFT
    # data = np.fft.fftshift(data, axes=2)
    data = np.abs(data).sum(axis=0)  # Sum over antennas
    # data = np.log(1 + data)
    return data
def radar_cube_map(data, fft_size = 4):
    data = np.fft.fft(data, axis=1) # Range FFT
    data = np.fft.fft(data, axis=2) # Velocity FFT
    data = np.fft.fft(data, n=fft_size, axis=0) # Angle FFT
    data = np.abs(data)
    return data
def minmax(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())



path_root=["/efs/data/Adaptation_dataset_multi_modal/scenario31","/efs/data/Adaptation_dataset_multi_modal/scenario32",
"/efs/data/Adaptation_dataset_multi_modal/scenario33"]

path_aug_root=["/efs/data/radar_data_augmentation/Adaptation_dataset_multi_modal/scenario31/",
"/efs/data/radar_data_augmentation/Adaptation_dataset_multi_modal/scenario32/",
"/efs/data/radar_data_augmentation/Adaptation_dataset_multi_modal/scenario33/"]

for path_idx in range(len(path_root)):

    path= path_root[path_idx] + "/unit1/radar_data/"
    path_aug_ang = path_aug_root[path_idx] + "/radar_data_ang/"
    path_aug_vel= path_aug_root[path_idx] + "/radar_data_vel/"
    #path_aug_cube= path_aug_root[path_idx] + "/radar_data_cube/"


    radarfiles=os.listdir(path)


    if not os.path.isdir(path_aug_ang):
        os.mkdir(path_aug_ang)

    if not os.path.isdir(path_aug_vel):
        os.mkdir(path_aug_vel)
    
    #if not os.path.isdir(path_aug_cube):
    #    os.mkdir(path_aug_cube)

    for filename in tqdm(radarfiles):
        if ".npy" in filename:
            data=np.load(path+filename)
            radar_range_ang_data = range_angle_map(data)
            radar_range_vel_data = range_velocity_map(data)

            ##################### Shift disturbance ########################
            #range_ang_max=np.amax(radar_range_ang_data)*0.1
            #range_vel_max=np.amax(radar_range_vel_data)*0.1

            #range_ang_shift=random.uniform(range_ang_max*0.25,range_ang_max)
            #range_vel_shift=random.uniform(range_vel_max*0.25,range_vel_max)

            
            #################### Radar range angle augmentation ############
            radar_range_ang_data_aug=[]

            for x_idx in range(len(radar_range_ang_data)):
                row_aug=[]

                for y_idx in range(len(radar_range_ang_data[x_idx])):
                    
                    random_shift=radar_range_ang_data[x_idx][y_idx]*0.1
                    row_aug.append(radar_range_ang_data[x_idx][y_idx]+random.uniform(random_shift*0.25,random_shift))
                    
                    #row_aug.append(radar_range_ang_data[x_idx][y_idx]+range_ang_shift)

                radar_range_ang_data_aug.append(row_aug)

            #################### Radar range velocity augmentation ############
            radar_range_vel_data_aug=[]

            for x_idx in range(len(radar_range_vel_data)):
                row_aug=[]

                for y_idx in range(len(radar_range_vel_data[x_idx])):
                    
                    random_shift=radar_range_vel_data[x_idx][y_idx]*0.1
                    row_aug.append(radar_range_vel_data[x_idx][y_idx]+random.uniform(random_shift*0.25,random_shift))
                    
                    #row_aug.append(radar_range_vel_data[x_idx][y_idx]+range_vel_shift)

                radar_range_vel_data_aug.append(row_aug)


            np.save(path_aug_ang+filename,minmax(np.asarray(radar_range_ang_data_aug)))
            np.save(path_aug_vel+filename, minmax(np.asarray(radar_range_vel_data_aug)))
            
            #deb1=minmax(radar_range_ang_data)
            #deb2=minmax(np.asarray(radar_range_ang_data_aug))
            #deb11=minmax(radar_range_vel_data)
            #deb22=minmax(np.asarray(radar_range_vel_data_aug))
            #print(deb1[200][120])
            #print(deb2[200][120])
            #print(deb11[200][120])
            #print(deb22[200][120])
            #a=1

