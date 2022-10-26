import os
import random
import torchvision.transforms.functional as F
from PIL import Image

images_path="/efs/data/Adaptation_dataset_multi_modal/scenario31/unit1/camera_data_raw/"
images_augmentation_path="/efs/data/Adaptation_dataset_multi_modal/scenario31/unit1/camera_data_aug/"

image_list=os.listdir(images_path)

for image_item in image_list: 

    img=images_path+image_item
    img_sample=Image.open(img)

    brightness_factor=random.uniform(0.5,3)   # Min 0.5 Max 3
    img_aug1=F.adjust_brightness(img_sample, brightness_factor)
    img_aug_path1=images_augmentation_path+image_item[:-4]+"_1.jpg"
    img_aug1.save(img_aug_path1, 'JPEG')

    contrast_factor=random.uniform(0.5,4)  # Min 0.5 Max 4
    img_aug2=F.adjust_contrast(img_sample, contrast_factor)
    img_aug_path2=images_augmentation_path+image_item[:-4]+"_2.jpg"
    img_aug2.save(img_aug_path2, 'JPEG')

    gamma_factor=random.uniform(0.5,3)   # Min 0.5 Max 3
    img_aug3=F.adjust_gamma(img_sample, gamma_factor)
    img_aug_path3=images_augmentation_path+image_item[:-4]+"_3.jpg"
    img_aug3.save(img_aug_path3, 'JPEG')

    hue_factor=random.uniform(-0.5,0.5)   # Min -0.5 Max 0.5
    img_aug4=F.adjust_hue(img_sample, hue_factor)
    img_aug_path4=images_augmentation_path+image_item[:-4]+"_4.jpg"
    img_aug4.save(img_aug_path4, 'JPEG')

    saturation_factor=random.uniform(0,4)  # Min 0 Max 4
    img_aug5=F.adjust_saturation(img_sample, saturation_factor)
    img_aug_path5=images_augmentation_path+image_item[:-4]+"_5.jpg"
    img_aug5.save(img_aug_path5, 'JPEG')

    sharpness_factor=random.uniform(0,10)  # Min 0 Max 10
    img_aug6=F.adjust_sharpness(img_sample, sharpness_factor)
    img_aug_path6=images_augmentation_path+image_item[:-4]+"_6.jpg"
    img_aug6.save(img_aug_path6, 'JPEG')

    kernel_size_factor=(9,7)
    sigma_factor=(3, 5)
    img_aug7=F.gaussian_blur(img_sample, kernel_size=kernel_size_factor, sigma=sigma_factor)
    img_aug_path7=images_augmentation_path+image_item[:-4]+"_7.jpg"
    img_aug7.save(img_aug_path7, 'JPEG')

