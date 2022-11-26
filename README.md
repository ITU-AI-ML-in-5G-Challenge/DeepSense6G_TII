# DeepSense6G_TII
The repository contains the code for the solution of Team TII for ITU AI/ML in 5G Grand Challenge 2022: [`ML5G-PS-011: Multi Modal Beam Prediction: Towards Generalization`](https://deepsense6g.net/multi-modal-beam-prediction-challenge/) 
* Full Report: documents/report.pdf
* Presentation: documents/presentation.pdf

## Introduction  
We develop a transformer-based multi-modal deep learning framework for sensing assisted beam prediction. We first preprocess sensor data by enhancing and segmenting images, filtering point-clouds, transforming radar signal and user's GPS location. We then employ ResNet CNN to extract the features from image, point-cloud and radar raw data. The GPT transformer is used after each convolutional block to fuse feature maps of different modalities. We utilize data augmentation, soft targets, focal loss, cosine decay schedular, exponential moving average to train the model. Experimental results shows our model produces effective beam prediction generalized to different scenarios. Our framework can be easily extended to different applications of sensing and communicaitons. 
## Installation
Clone this repository:
```sh
git clone https://github.com/DeepSenseChallengeTeam/DeepSense6G_TII.git
cd DeepSense6G_TII
```
Create the environment:
```sh
conda env create -f environment.yml 
conda activate tfuse 
```
## Data Preprocessing
We enhance and augment the multi-modal sensor data provided by the challenge. The final data necessary to reproduce our experiment can be downloaded directly from this link: [`MultiModalSensorPreprocessedData`](https://drive.google.com/drive/folders/1zvOOJpGodEnjqvAiAeXkzOdjWmz1semF?usp=sharing)

We develop following tools to preprocess the original dataset for training our model: 

* [`Lidar_data_preprocessing.py`](./Lidar_data_preprocessing.py): filter the backgrounds and retain the mobile objects.
* [`Image_data_augmentation.py`](./Image_data_augmentation.py): chagne image brightness, contrast, gamma, hue, saturation, sharpness, bluring. 
* [`Lidar_data_augmentation.py`](./Lidar_data_augmentation.py): change point-cloud by down sampling, adding noise. 
* [`radar_data_augmentation.py`](./radar_data_augmentation.py): change radar signals by adding noise in the spectral domain.

## Training and Evaluation
The framework can be experimented with different approaches and hyperparameters for training and data preprocessing. The configurations and descriptions can be viewed as follows:
```
python3 train2_seq.py --help
```
The core code of our solution can be found in the following scripts:

* [`train2_seq.py`](./train2_seq.py): main engine for multi-modal deep learning.
* [`model2_seq.py`](./model2_seq.py): transformer based sensing aid beam prediction model.
* [`data2_seq.py`](./data2_seq.py): data preparation for training and evaluation.

A minimal example of running the training script to reproduce our best submitted model: 
```sh
python3 train2_seq.py --id test --logdir log --device cuda --epochs 150 --lr 1e-4 --batch_size 12 --add_velocity 1 --add_mask 0 --enhanced 1 --filtered 0 --loss focal --scheduler 1 --load_previous_best 0 --temp_coef 1 --train_adapt_together 1 --finetune 0 --Test 0 --augmentation 1 --angle_norm 1 --custom_FoV_lidar 1 --add_seg 0 --ema 1 --flip 0
```
The best pretrained model can be downloaded from this link: [`best_model.pth`](https://tiiuae-my.sharepoint.com/:u:/g/personal/yu_tian_tii_ae/ESWmKoHeKsxJorYTr6MxgjQBlCXrRQoSrgLDxs7ljxEr_g?e=bPrCgS) 

The following script reproduce our best result, by saving the pretrained model in ./log/test folder:
```sh
python3 train2_seq.py --id test --logdir log --Test 1 --add_velocity 1 --add_mask 0 --enhanced 1 --filtered 0 --angle_norm 1 --custom_FoV_lidar 1 --add_seg 0
```
The best DBA score of each scenario on the test dataset is as follows:
|Scenario 31|Scenario 32|Scenario 33|Scenario 34|Overall|
|:-:|:-:|:-:|:-:|:-:|
|0.5331|0.7173|0.7910|0.8209|0.6671|


## Future work
We are experimenting further approaches to improve the solutions such as semi-supervised learning, batch former, fine tuning. We also plan to extend the framework for wider integrated sensing and communications. 

## Contact

* Qiyang Zhao, Yu Tian, Zine el abidine Kherroubi, Fouzi Boukhalfa
* {qiyang.zhao, yu.tian, zine.kherroubi, fouzi.boukhalfa}@tii.ae
* Team members contributed equally to this work. 

## Reference
* [TransFuser](https://github.com/autonomousvision/transfuser): Imitation with Transformer-Based Sensor Fusion for Autonomous Driving
* [MIRNet](https://github.com/swz30/MIRNet): Learning Enriched Features for Ream Image Resotration and Enhancement
* [PIDNet](https://github.com/XuJiacong/PIDNet): A Real-time Semantic Segmentation Network Inspired form PID Controller
