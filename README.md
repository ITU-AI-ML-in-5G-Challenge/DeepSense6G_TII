# DeepSense6G_TII
The repository contains the code for the solution of team TII in ['Multi Modal Beam Prediction Challenge 2022: Towards Generalization'](https://deepsense6g.net/multi-modal-beam-prediction-challenge/). 
## Contents
1. Setup
2. Dataset 
3. Train and Evaluation
4. Future work
## Setup
Clone the repo and build the conda environment:
```
git clone https://github.com/DeepSenseChallengeTeam/DeepSense6G_TII.git
conda env create -f environment.yml 
conda activate tfuse 
```
## Dataset
As we have generated many preprocessed data inlcuding segmented, masked, enhanced, and augumented, our dataset can be downloaded from [](). After downloading please put it in the folder of './DeepSense6G_TII/'.

## Train and Evaluation
The code for training is provide in (XXXX).
To train the model, please run the following script in the terminal
```
python3 train2_seq.py --id test --logdir log --device cuda --epochs 150 --lr 1e-4 --batch_size 12 --add_velocity 1 --add_mask 0 --enhanced 1 --filtered 0 --loss focal --scheduler 1 --load_previous_best 0 --temp_coef 1 --train_adapt_together 1 --finetune 0 --Test 0 --augmentation 1 --angle_norm 1 --custom_FoV_lidar 1 --add_seg 0 --ema 1 --flip 0
```

In the challenge, we have achieved the best overall DBA score of 0.6671. The scores corresponding to different scenarios are listed in the table below

|Scenario 31|Scenario 32|Scenario 33|Scenario 34|Overall|
|---|---|---|---|---|
|0.5331|0.7173|0.7910|0.8209|0.6671|

The pretrained model of this score can be downloaded from [](). Then save it in the fold './DeepSense6G_TII/log/test'

To evaluate the pretrained model, please run the following script
'''
python3 train2_seq.py --id test --logdir log --Test 1 --add_velocity 1 --add_mask 0 --enhanced 1 --filtered 0 --angle_norm 1 --custom_FoV_lidar 1 --add_seg 0
'''

## Future work
We develop other functions like finetune, fixmatch (semi-supervised learning) which can be used in other applciations. 






