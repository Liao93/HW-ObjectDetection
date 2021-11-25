# HW-ObjectDetection

## Training

### Environment
- python 3.7.11
- h5py 2.8.0
- numpy 1.21.2
- Pillow 8.4.0
- torch 1.10.0
- torchvision 0.11.1
- cudatoolkit 11.3.1
- albumentations 1.1.0

Before running 'train.py', check these files is at the right path.
- All the training images: './train/'
- digitStruct.mat: './train/digitStruct.mat'

This code contains two backbones. You can choose one by commenting or uncommenting the block.

For Training, I used the training scripts from here. (https://github.com/pytorch/vision/tree/main/references/detection)

## Inference

You can run 'inference.ipynb' to create the answer.json.

You have to modify some variables in this code, to the path of your files.
- The path of 'test.zip' in 'unzip' command.
- Your model's path in the 'Set model path' cell.
- The path of json file in the last cell. You can modify it to the path you want.
- This code runs on Colab, so the data list (testing images) path is default to '/content/test'.  You may have to modify it if you want to run it on your own machine.

Pre-trained model can be found here:
- Backbone ResNet50_FPN: './models/model_ResNet50_FPN_379.pkl'
- Backbone Efficientnet_B2: './models/model_Efficientnet_B2_352.pkl'
