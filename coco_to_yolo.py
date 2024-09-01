import os
import gc
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import yaml
from shutil import copyfile
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

json_file_path = 'captions_train2014.json'

data = json.load(open(json_file_path, 'r'))
yolo_anno_path = 'yolo_anno'

if not os.path.exists(yolo_anno_path):
    os.makedirs(yolo_anno_path)

# 需要注意下因为我们的annotation lable是不连续的,会导致后面报错,所以这里生成一个map映射
cate_id_map = {}
num = 0
for cate in data['categories']:
    cate_id_map[cate['id']] = num
    num+=1   # cate_id_map -> {87: 0, 1034: 1, 131: 2, 318: 3, 588: 4}


# convert the bounding box from COCO to YOLO format.

def cc2yolo_bbox(img_width, img_height, bbox): # [xmin,ymin,width,height] -> [x,y,width,height](归一化)
    dw = 1. / img_width
    dh = 1. / img_height
    x = bbox[0] + bbox[2] / 2.0
    y = bbox[1] + bbox[3] / 2.0
    w = bbox[2]
    h = bbox[3]

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


f = open('train.csv', 'w') # train.cvs用于传入train_test_split函数来分割训练集和验证集,详见sklearn.model_selection官方文档说明
f.write('id,file_name\n')
for i in tqdm(range(len(data['images']))): #生成txt标注
    filename = data['images'][i]['file_name']
    img_width = data['images'][i]['width']
    img_height = data['images'][i]['height']
    img_id = data['images'][i]['id']
    yolo_txt_name = filename.split('.')[0] + '.txt'  # remove .jpg

    f.write('{},{}\n'.format(img_id, filename))
    yolo_txt_file = open(os.path.join(yolo_anno_path, yolo_txt_name), 'w')

    for anno in data['annotations']:
        if anno['image_id'] == img_id:
            yolo_bbox = cc2yolo_bbox(img_width, img_height, anno['bbox'])  # "bbox": [x,y,width,height]
            yolo_txt_file.write(
                '{} {} {} {} {}\n'.format(cate_id_map[anno['category_id']], yolo_bbox[0], yolo_bbox[1], yolo_bbox[2],
                                          yolo_bbox[3]))
    yolo_txt_file.close()
f.close()

# spli train dataset to train and valid dataset
train = pd.read_csv('train.csv')
train_df, valid_df = train_test_split(train, test_size=0, random_state=233)
train_df.loc[:, 'split'] = 'train'
valid_df.loc[:, 'split'] = 'valid'
df = pd.concat([train_df, valid_df]).reset_index(drop=True)

#创建文件夹
os.makedirs('coco_yolo/images/train', exist_ok=True)
os.makedirs('coco_yolo/images/valid', exist_ok=True)
os.makedirs('coco_yolo/labels/train', exist_ok=True)
os.makedirs('coco_yolo/labels/valid', exist_ok=True)

# move the images and annotations to relevant splited folders

for i in tqdm(range(len(df))):
    row = df.loc[i]
    name = row.file_name.split('.')[0]
    if row.split == 'train':
        copyfile(f'D://Dataset//cowboyoutfits//images//{name}.jpg', f'D://Pycharm//yolov5-5.0//training//cowboy//images//train//{name}.jpg')
        copyfile(f'D://Pycharm//Project_DL//training//yolo_anno//{name}.txt', f'D://Pycharm//yolov5-5.0//training//cowboy//labels//train//{name}.txt')
    else:
        copyfile(f'D://Dataset//cowboyoutfits//images//{name}.jpg', f'D://Pycharm//yolov5-5.0//training//cowboy//images//valid//{name}.jpg')
        copyfile(f'D://Pycharm//Project_DL//training//yolo_anno//{name}.txt', f'D://Pycharm//yolov5-5.0//training//cowboy//labels//valid//{name}.txt')


data_yaml = dict(
    train = '../cowboy/images/train/',
    val = '../cowboy/images/valid',
    nc = 5,
    names = ['belt', 'sunglasses', 'boot', 'cowboy_hat', 'jacket']
)

# we will make the file under the yolov5/data/ directory.
with open('coco_yolo/data.yaml', 'w') as outfile: #这个data.yaml得自己手动创建
    yaml.dump(data_yaml, outfile, default_flow_style=True)