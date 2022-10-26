import os
import random
import math
import numpy as np
import open3d as o3d

lidar_path=[["/efs/data/Adaptation_dataset_multi_modal/scenario31/unit1/lidar_data/",
"/efs/data/Multi_Modal_Test/scenario31/unit1/lidar_data/"],
["/efs/data/Adaptation_dataset_multi_modal/scenario32/unit1/lidar_data/",
"/efs/data/Multi_Modal/scenario32/unit1/lidar_data/","/efs/data/Multi_Modal_Test/scenario32/unit1/lidar_data/"],
["/efs/data/Adaptation_dataset_multi_modal/scenario33/unit1/lidar_data/","/efs/data/Multi_Modal/scenario33/unit1/lidar_data/",
"/efs/data/Multi_Modal_Test/scenario33/unit1/lidar_data/"],["/efs/data/Multi_Modal/scenario34/unit1/lidar_data/",
"/efs/data/Multi_Modal_Test/scenario34/unit1/lidar_data/"]]

#lidar_path=[["/efs/data/Adaptation_dataset_multi_modal/scenario31/unit1/lidar_data/",
#"/efs/data/Multi_Modal_Test/scenario31/unit1/lidar_data/"]]

preprocessed_lidar_path=[["/efs/data/preprocess_lidar/Adaptation_dataset_multi_modal/scenario31/",
"/efs/data/preprocess_lidar/Multi_Modal_Test/scenario31/"],
["/efs/data/preprocess_lidar/Adaptation_dataset_multi_modal/scenario32/",
"/efs/data/preprocess_lidar/Multi_Modal/scenario32/","/efs/data/preprocess_lidar/Multi_Modal_Test/scenario32/"],
["/efs/data/preprocess_lidar/Adaptation_dataset_multi_modal/scenario33/","/efs/data/preprocess_lidar/Multi_Modal/scenario33/",
"/efs/data/preprocess_lidar/Multi_Modal_Test/scenario33/"],["/efs/data/preprocess_lidar/Multi_Modal/scenario34/",
"/efs/data/preprocess_lidar/Multi_Modal_Test/scenario34/"]]
#preprocessed_lidar_path=[["/efs/data/preprocess_lidar/Adaptation_dataset_multi_modal/scenario31/",
#"/efs/data/preprocess_lidar/Multi_Modal_Test/scenario31/"]]

lidar_background_path="/efs/data/preprocess_lidar/Background/"

scenario_list=["scenario31","scenario32","scenario33","scenario34"]
scenario_min_points=[16400,18000,18000,18600]
#scenario_list=["scenario31"]
#scenario_min_points=[16400]

########################## Background filtering ################################
filter_distance_min=0.3
filter_distance_max=5
lidar_distance_min=40
lidar_distance_cst=30


for scenario_idx in range(len(scenario_list)):

    init_idx=0
    lidar_list=os.listdir(lidar_path[scenario_idx][0])
    background_pcl = o3d.io.read_point_cloud(lidar_path[scenario_idx][0]+lidar_list[0])
    
    while((np.asarray(background_pcl.points).shape[0])<scenario_min_points[scenario_idx]):
        init_idx+=1
        lidar_list=os.listdir(lidar_path[scenario_idx][init_idx])
        background_pcl = o3d.io.read_point_cloud(lidar_path[scenario_idx][0]+lidar_list[0])    
    

    for lidar_path_item in lidar_path[scenario_idx]:
        
        lidar_list=os.listdir(lidar_path_item)


        for lidar_item in lidar_list: 
            lidar=lidar_path_item+lidar_item
            pcd2 = o3d.io.read_point_cloud(lidar)
            if((np.asarray(pcd2.points).shape[0])>=scenario_min_points[scenario_idx]):

                pcd_tree = o3d.geometry.KDTreeFlann(pcd2)
                background_pcl_list=[]

                for point_item in background_pcl.points:

                    [k, idx, _]=pcd_tree.search_knn_vector_3d(point_item, 1)
                
                    dx=point_item[0]-pcd2.points[idx[0]][0]
                    dy=point_item[1]-pcd2.points[idx[0]][1]
                    dz=point_item[2]-pcd2.points[idx[0]][2]
                    #distance=math.sqrt((dx*dx)+(dy*dy)+(dz*dz))
                    distance=math.sqrt((dx*dx)+(dy*dy))

                    px=point_item[0]
                    py=point_item[1]
                    pz=point_item[2]
                    #point_distance=math.sqrt((px*px)+(py*py)+(pz*pz))
                    point_distance=math.sqrt((px*px)+(py*py))

                

                    filter_distance=filter_distance_min+(filter_distance_max-filter_distance_min)*(point_distance/lidar_distance_cst)**4    

                    if (distance<filter_distance):
                        x=(point_item[0]+pcd2.points[idx[0]][0])/2
                        y=(point_item[1]+pcd2.points[idx[0]][1])/2
                        z=(point_item[2]+pcd2.points[idx[0]][2])/2

                        background_pcl_list.append([x,y,z])

                background_pcl.points = o3d.utility.Vector3dVector(np.array(background_pcl_list))
            


    lidar_background_path_item=lidar_background_path+scenario_list[scenario_idx]+"_background.ply"
    o3d.io.write_point_cloud(lidar_background_path_item,background_pcl,write_ascii=True)


########################### Preprocessing original data ##########################################

for scenario_idx in range(len(scenario_list)):

    lidar_background="/efs/data/preprocess_lidar/Background/"+scenario_list[scenario_idx]+"_background.ply"
    background_pcl = o3d.io.read_point_cloud(lidar_background)

    for lidar_path_idx in range(len(lidar_path[scenario_idx])):

        lidar_path_item=lidar_path[scenario_idx][lidar_path_idx]
        preprocessed_lidar_path_item=preprocessed_lidar_path[scenario_idx][lidar_path_idx]
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
            preprocessed_lidar=preprocessed_lidar_path_item+lidar_item  
            o3d.io.write_point_cloud(preprocessed_lidar,pcd,write_ascii=True)


