import h5py
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import albumentations as A
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from visionApi.engine import train_one_epoch, evaluate
from visionApi import utils

file_num = 33402
mat = h5py.File('./train/digitStruct.mat', 'r')


def get_img_name(f, idx=0):
    names = f['digitStruct/name']
    entry = f[names[idx, 0]]
    img_name = "".join(chr(i) for i in entry[:])
    return(img_name)


def get_img_boxes(f, idx=0):
    """
    get the 'height', 'left', 'top', 'width', 'label'
    of bounding boxes of an image
    :return: nummpy array
    """
    bbox_prop = ['height', 'left', 'top', 'width', 'label']
    dic = {key: [] for key in bbox_prop}

    bboxs = f['digitStruct/bbox']
    box = f[bboxs[idx, 0]]
    for key in box.keys():
        # only 1 bbox in this image
        if box[key].shape[0] == 1:
            dic[key].append(int(box[key][0][0]))
        # many bboxs in this image
        else:
            for i in range(box[key].shape[0]):
                dic[key].append(int(f[box[key][i][0]][()].item()))
    arr = np.column_stack((dic['left'], dic['top'],
                           dic['width'], dic['height'], dic['label']))
    return arr


idx_list = [i for i in range(file_num)]


class SVHNDataset(torch.utils.data.Dataset):
    def __init__(self, mat, i_list, training=False):
        self.mat = mat
        self.idx_list = i_list
        self.training = training

    def __getitem__(self, idx):
        # load images
        img_id = self.idx_list[idx]
        filename = get_img_name(self.mat, img_id)
        root = 'train'
        img_path = os.path.join(root, filename)
        img = Image.open(img_path).convert("RGB")

        # get bounding box
        boxes = []
        labels = []
        boxes_arr = get_img_boxes(self.mat, img_id)

        transformed_dict = self.A_transforms(np.asarray(img), boxes_arr)
        img = transformed_dict["image"]
        boxes_arr = np.array(list(
            map(list, transformed_dict["bboxes"]))).astype(float)

        objs_num = boxes_arr.shape[0]
        for i in range(objs_num):
            # digit 0 is labeled as 10
            labels.append(boxes_arr[i][4])

            xmin = boxes_arr[i][0]
            ymin = boxes_arr[i][1]
            xmax = xmin + boxes_arr[i][2]
            ymax = ymin + boxes_arr[i][3]
            boxes.append([xmin, ymin, xmax, ymax])

        img = transforms.ToTensor()(img)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([img_id])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((objs_num,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return img, target

    def __len__(self):
        return len(self.idx_list)

    def A_transforms(self, img_arr, bboxes, h=128, w=128):
        if self.training:
            transform = A.Compose([
                A.Resize(height=h, width=w, always_apply=True),
                A.ChannelShuffle(p=1.0),
                A.RandomBrightnessContrast(p=1.0),
                A.geometric.transforms.ShiftScaleRotate(p=0.5)],
                bbox_params=A.BboxParams(format='coco'))
        else:
            transform = A.Compose([
                A.Resize(height=h, width=w, always_apply=True)],
                bbox_params=A.BboxParams(format='coco'))

        transformed = transform(image=img_arr, bboxes=bboxes)
        return transformed


"""backbone ResNet50_fpn"""
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# 10 classes + background
num_classes = 1 + 10
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
""""""

"""backbone EfficientNet_B2"""
"""
backbone = torchvision.models.efficientnet_b2(pretrained=True).features
backbone.out_channels = 1408

anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))

roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)
num_classes = 1 + 10  # 10 classes + background
model = FasterRCNN(backbone,
                   num_classes=num_classes,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)
"""
""""""

train_data = SVHNDataset(mat, idx_list, training=True)
dataloader = DataLoader(train_data, batch_size=4, shuffle=True,
                        collate_fn=utils.collate_fn)

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(device)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

num_epochs = 10

for epoch in range(num_epochs):
    # train for one epoch
    train_one_epoch(model, optimizer, dataloader, device,
                    epoch, print_freq=500)
    torch.save(model.state_dict(), "model_ep{}.pkl".format(epoch+1))
    # update the learning rate
    lr_scheduler.step()

print("That's it!")
