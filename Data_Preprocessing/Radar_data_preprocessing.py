import glob
import os
import numpy as np
import matplotlib.pyplot as plt


def range_angle_map(data, fft_size=256):
    data = np.fft.fft(data, axis=1)  # Range FFT
    data -= np.mean(data, 2, keepdims=True)
    data = np.fft.fft(data, fft_size, axis=0)  # Angle FFT
    # data = np.fft.fftshift(data, axes=0)
    data = np.abs(data).sum(axis=2)  # Sum over velocity
    return data.T
def range_velocity_map(data):
    data = np.fft.fft(data, axis=1)  # Range FFT
    # data -= np.mean(data, 2, keepdims=True)
    data = np.fft.fft(data, 256, axis=2)  # Velocity FFT
    # data = np.fft.fftshift(data, axes=2)
    data = np.abs(data).sum(axis=0)  # Sum over antennas
    # data = np.log(1 + data)
    return data
def minmax(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())
folder= 'Adaptation_dataset_multi_modal/scenario34'# Adaptation_dataset_multi_modal 0Multi_Modal Multi_Modal_Test
path="./Dataset/"+folder+"/unit1/radar_data/"
path_ang="./Dataset/"+folder+"/unit1/radar_data_ang/"
path_vel="./Dataset/"+folder+"/unit1/radar_data_vel/"

radarfiles=os.listdir(path)
if not os.path.isdir(path_ang):
    os.mkdir(path_ang)

if not os.path.isdir(path_vel):
    os.mkdir(path_vel)
from joblib import Parallel, delayed
def process(filename):
    print(filename)
    data = np.load(path + filename)
    radar_range_ang_data = range_angle_map(data)
    radar_range_vel_data = range_velocity_map(data)
    np.save(path_ang + filename, minmax(radar_range_ang_data))
    np.save(path_vel + filename, minmax(radar_range_vel_data))
Parallel(n_jobs=100)(delayed(process)(filename) for filename in radarfiles if ".npy" in filename)
