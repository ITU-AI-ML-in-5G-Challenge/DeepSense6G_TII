import os
import random
import math
import numpy as np
import open3d as o3d

lidar_path=["/efs/data/preprocess_lidar/Adaptation_dataset_multi_modal/scenario31/",
"/efs/data/preprocess_lidar/Adaptation_dataset_multi_modal/scenario32/",
"/efs/data/preprocess_lidar/Adaptation_dataset_multi_modal/scenario33/"]

lidar_augmentation_path=["/efs/data/preprocess_lidar/Adaptation_dataset_multi_modal/scenario31_aug/",
"/efs/data/preprocess_lidar/Adaptation_dataset_multi_modal/scenario32_aug/",
"/efs/data/preprocess_lidar/Adaptation_dataset_multi_modal/scenario33_aug/"]


###################  Lidar data augmentation: downsampling (0.9%) and adding noise (-+0.4)
noise_range=0.4

for scenario_idx in range(len(lidar_path)):


    lidar_list=os.listdir(lidar_path[scenario_idx])
    for lidar_item in lidar_list: 

        lidar=lidar_path[scenario_idx]+lidar_item
        pcd = o3d.io.read_point_cloud(lidar)

        downpcd1 = pcd.random_down_sample(0.9)
        lidar_aug_path1=lidar_augmentation_path[scenario_idx]+lidar_item[:-4]+"_1.ply"
        o3d.io.write_point_cloud(lidar_aug_path1,downpcd1,write_ascii=True)

        
        for point_item in pcd.points:
            point_item[0]=point_item[0]+random.uniform(-noise_range,noise_range)
            point_item[1]=point_item[1]+random.uniform(-noise_range,noise_range)
            point_item[2]=point_item[2]+random.uniform(-noise_range,noise_range)

        lidar_aug_path2=lidar_augmentation_path[scenario_idx]+lidar_item[:-4]+"_2.ply"
        o3d.io.write_point_cloud(lidar_aug_path2,pcd,write_ascii=True)

'''
########################### Preprocessing augmented data ##########################################
scenario_list=["scenario31"]
lidar_aug_path=[["/efs/data/Adaptation_dataset_multi_modal/scenario31/unit1/lidar_data_aug/"]]
preprocessed_lidar_aug_path=[["/efs/data/preprocess_lidar/Adaptation_dataset_multi_modal/scenario31_aug/"]]


for scenario_idx in range(len(scenario_list)):

    lidar_background="/efs/data/preprocess_lidar/Background/"+scenario_list[scenario_idx]+"_background.ply"
    background_pcl = o3d.io.read_point_cloud(lidar_background)

    for lidar_path_idx in range(len(lidar_aug_path[scenario_idx])):

        lidar_path_item=lidar_aug_path[scenario_idx][lidar_path_idx]
        preprocessed_lidar_aug_path_item=preprocessed_lidar_aug_path[scenario_idx][lidar_path_idx]
        lidar_list=os.listdir(lidar_path_item)
        
        for lidar_item in lidar_list: 
            lidar=lidar_path_item+lidar_item     
            pcd = o3d.io.read_point_cloud(lidar)

            pcd_tree = o3d.geometry.KDTreeFlann(background_pcl)
            filtered_pcl_list=[]


            filter_distance_min=0.3
            filter_distance_max=5
            lidar_distance_min=40
            lidar_distance_cst=30

            for point_item in pcd.points:

                [k, idx, _]=pcd_tree.search_knn_vector_3d(point_item, 1)
                
                dx=point_item[0]-background_pcl.points[idx[0]][0]
                dy=point_item[1]-background_pcl.points[idx[0]][1]
                dz=point_item[2]-background_pcl.points[idx[0]][2]
                distance=math.sqrt((dx*dx)+(dy*dy))

                px=point_item[0]
                py=point_item[1]
                pz=point_item[2]
                point_distance=math.sqrt((px*px)+(py*py))


                filter_distance=filter_distance_min+(filter_distance_max-filter_distance_min)*(point_distance/lidar_distance_cst)**4    


                if (distance>=filter_distance):

                    filtered_pcl_list.append(point_item)


            pcd.points = o3d.utility.Vector3dVector(np.array(filtered_pcl_list))
            preprocessed_lidar=preprocessed_lidar_aug_path_item+lidar_item  
            o3d.io.write_point_cloud(preprocessed_lidar,pcd,write_ascii=True)

'''




