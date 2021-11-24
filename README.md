# HW-ObjectDetection
 
## Environment
- python 3.7.11
- h5py 2.8.0
- numpy 1.21.2
- Pillow 8.4.0
- matplotlib 3.4.3
- torch 1.10.0
- torchvision 0.11.1
- cudatoolkit 11.3.1
- scikit-learn 1.0
- albumentations 1.1.0

## Training

Check these files is at the right path.
- All the training images: './train/'
- digitStruct.mat: './train/digitStruct.mat'

For Training, I used the training scripts from here. (https://github.com/pytorch/vision/tree/main/references/detection)

## Inference

You can use inference.ipynb to create the answer.json.

You have to modify some variables in this code, to the path of your files.
- The path of 'test.zip' in 'unzip' command.
- Your model's path in the 'Set model path' cell.
- The path of json file in the last cell. You can modify it to the path you want.
- This code runs on Colab, so the data list (testing images) path is default to '/content/test'.  You may have to modify it if you want to run it on your own machine.
