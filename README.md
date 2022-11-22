# DeepSense6G_TII
The repo is the deep-learning-based framework of team TII for ['Multi Modal Beam Prediction Challenge 2022: Towards Generalization'](https://deepsense6g.net/multi-modal-beam-prediction-challenge/). 
## Dataset
### Preprocessed data
1. Camera data: 
    - 1.1 we utilized a black mask to cover the useless background. Controlled by the argument '--add_mask'
    - 1.2 We utilized semantic segmentation to detect the vehicles in the images of daytime scenarios 31 and 32. Controlled by the argument '--add_seg'
    - 1.3 We utilized MIRNet to enhance the lightness of the images of night time scenarios 33 and 34. Controlled by the argument '--enhanced'
2. Lidar data: we compared all the lidar data in one scenario, detected the stable background, and then removed the background from the lidar while just keeping the mobile vechiles and pedestrians. The code is in XXXX(lidar preprocess code). Controlled by the argument '--filtered'
3. Radar data: we utilized the following radar preprocessing code to get the range-velocity feature and range-angle feature, then used minmax function to normalize the oabtained features. If setting the argument '--add_velocity=1', both features will be used. Otherwise, only the range-angle feature is applied.
```
def range_velocity_map(data):
    data = np.fft.fft(data, axis=1) # Range FFT
    data = np.fft.fft(data, 256, axis=2) # Velocity FFT
    data = np.fft.fftshift(data, axes=2)
    data = np.abs(data).sum(axis = 0) # Sum over antennas
    data = np.log(1+data)
    return data

def range_angle_map(data):
    data = np.fft.fft(data, axis = 1) # Range FFT
    data -= np.mean(data, 2, keepdims=True)
    data = np.fft.fft(data, 256, axis = 0) # Angle FFT
    data = np.fft.fftshift(data, axes=0)
    data = np.abs(data).sum(axis = 2) # Sum over velocity
    return data.T
def minmax(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())
```
### Data Augmentation
We utilize two data augmentations here: adaptation and flip
1. Augment the adaptation dataset: randomly changing the brightness, contrast, gamma, hue, saturarion, sharpness, and blurring for the camera data; downsampling and adding noise to the Lidar points cloud; adding noise to the FFT coefficients in the spectral domain of the radar data. If setting the argument '--augmentation=1', data augmentation will be activated. 
2. horizontally flip the camera, ladar, radar, and GPS data: If '--flip=1', the 
## Train
## Test
### Pretrained model
You just need to navigate the folder of 'Solution of TII' And then run the following commands in the terminal. 
The 'beam_pred.csv' file will be in the same folder.

```
conda env create -f environment.yml 
conda activate tfuse 
python3 train2_seq.py --Test 1 --finetune 0
```
