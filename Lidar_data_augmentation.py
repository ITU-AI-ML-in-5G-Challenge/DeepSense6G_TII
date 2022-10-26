import os
import random
import math
import numpy as np
import open3d as o3d

lidar_path="/efs/data/Adaptation_dataset_multi_modal/scenario31/unit1/lidar_data/"
lidar_augmentation_path="/efs/data/Adaptation_dataset_multi_modal/scenario31/unit1/lidar_data_aug/"

lidar_list=os.listdir(lidar_path)


###################  Lidar data augmentation: downsampling (0.9%) and adding noise (-+0.4)
noise_range=0.4

for lidar_item in lidar_list: 

    lidar=lidar_path+lidar_item
    pcd = o3d.io.read_point_cloud(lidar)
    
    ############ Test
    downpcd1 = pcd.voxel_down_sample(voxel_size=2)
    downpcd2 = pcd.random_down_sample(0.8)
    downpcd3 = pcd.random_down_sample(0.7)

    lidar_aug_path1=lidar_augmentation_path+lidar_item[:-4]+"_1.ply"
    lidar_aug_path2=lidar_augmentation_path+lidar_item[:-4]+"_2.ply"
    lidar_aug_path3=lidar_augmentation_path+lidar_item[:-4]+"_3.ply"

    o3d.io.write_point_cloud(lidar_aug_path1,downpcd1,write_ascii=True)
    o3d.io.write_point_cloud(lidar_aug_path2,downpcd2,write_ascii=True)
    o3d.io.write_point_cloud(lidar_aug_path3,downpcd3,write_ascii=True)
    #############

    downpcd1 = pcd.random_down_sample(0.9)
    lidar_aug_path1=lidar_augmentation_path+lidar_item[:-4]+"_1.ply"
    o3d.io.write_point_cloud(lidar_aug_path1,downpcd1,write_ascii=True)

    
    for point_item in pcd.points:
        point_item[0]=point_item[0]+random.uniform(-noise_range,noise_range)
        point_item[1]=point_item[1]+random.uniform(-noise_range,noise_range)
        point_item[2]=point_item[2]+random.uniform(-noise_range,noise_range)

    lidar_aug_path2=lidar_augmentation_path+lidar_item[:-4]+"_2.ply"
    o3d.io.write_point_cloud(lidar_aug_path2,pcd,write_ascii=True)



