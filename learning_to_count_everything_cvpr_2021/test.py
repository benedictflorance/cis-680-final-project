#!/usr/bin/env python
# coding: utf-8

"""
Test code written by Viresh Ranjan

Last modified by: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
Date: 2021/04/19
"""

import copy
from model import CountRegressor, Resnet50FPN
from utils import MAPS, Scales, Transform, extract_features
from utils import MincountLoss, PerturbationLoss
from PIL import Image
from collections import defaultdict
import os
import torch
import argparse
import json
import numpy as np
from tqdm import tqdm
from os.path import exists
import torch.optim as optim
import matplotlib.pyplot as plt

NUM_ANGLES = 9
AVG_METHOD = "max"


parser = argparse.ArgumentParser(description="Few Shot Counting Evaluation code")
parser.add_argument("-dp", "--data_path", type=str, default='./data/', help="Path to the FSC147 dataset")
parser.add_argument("-ts", "--test_split", type=str, default='test', choices=["val_PartA","val_PartB","test_PartA","test_PartB","test", "val"], help="what data split to evaluate on")
parser.add_argument("-m",  "--model_path", type=str, default="./data/pretrainedModels/FamNet_Save1.pth", help="path to trained model")
parser.add_argument("-a",  "--adapt", action='store_true', help="If specified, perform test time adaptation")
parser.add_argument("-gs", "--gradient_steps", type=int,default=100, help="number of gradient steps for the adaptation")
parser.add_argument("-lr", "--learning_rate", type=float,default=1e-7, help="learning rate for adaptation")
parser.add_argument("-wm", "--weight_mincount", type=float,default=1e-9, help="weight multiplier for Mincount Loss")
parser.add_argument("-wp", "--weight_perturbation", type=float,default=1e-4, help="weight multiplier for Perturbation Loss")
parser.add_argument("-g",  "--gpu-id", type=int, default=0, help="GPU id. Default 0 for the first GPU. Use -1 for CPU.")
args = parser.parse_args()

data_path = args.data_path
anno_file = data_path + 'parts_new_annotations_scaled.json'
data_split_file = data_path + 'dataset_split_angles.json'
im_dir = 'parts_dataset_resized'
gt_dir = 'parts_gt_density_maps'

#data_path = args.data_path
#anno_file = data_path + 'annotation_FSC147_384.json'
#data_split_file = data_path + 'Train_Test_Val_FSC_147.json'
#im_dir = data_path + 'images_384_VarV2'

if not exists(anno_file) or not exists(im_dir):
    print("Make sure you set up the --data-path correctly.")
    print("Current setting is {}, but the image dir and annotation file do not exist.".format(args.data_path))
    print("Aborting the evaluation")
    exit(-1)

if not torch.cuda.is_available() or args.gpu_id < 0:
    use_gpu = False
    print("===> Using CPU mode.")
else:
    use_gpu = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

resnet50_conv = Resnet50FPN()
if use_gpu: resnet50_conv.cuda()
resnet50_conv.eval()

regressor = CountRegressor(6, pool='mean')
regressor.load_state_dict(torch.load(args.model_path))
if use_gpu: regressor.cuda()
regressor.eval()

with open(anno_file) as f:
    annotations = json.load(f)

with open(data_split_file) as f:
    data_split = json.load(f)

def get_batches(ids):
    tups = defaultdict(list)
    nums = [_id[0:_id.find("_")]  for _id in ids]
    for _id, num in zip(ids, nums):
        tups[num].append(_id)

    for k, v in tups.items():
        assert len(v) == 9
    return tups.values()

cnt = 0
SAE = 0  # sum of absolute errors
SSE = 0  # sum of square errors

print("Evaluation on {} data".format(args.test_split))
im_ids = data_split[args.test_split]
all_angles = get_batches(im_ids)
#print(all_angles)

pbar = tqdm(all_angles)
for batch in pbar:
    min_pred, max_pred, avg_pred = 99999, 0, 0
    gt_cnt = None
    for im_id in batch:
        print(im_id)
        if "215" in im_id: print(im_id)
        if not im_id == "215_angle8_img.png" and not im_id == "325_angle1_img.png" and not im_id == "134_angle4_img.png":
            continue

        
        print(im_id)
        anno = annotations[im_id]
        bboxes = anno['box_examples_coordinates']
        dots = np.array(anno['points'])

        rects = list()
        for bbox in bboxes:
            x1, y1 = bbox[0][0], bbox[0][1]
            x2, y2 = bbox[2][0], bbox[2][1]
            rects.append([y1, x1, y2, x2])

        image = Image.open('{}/{}'.format(im_dir, im_id))
        image.load()
        sample = {'image': image, 'lines_boxes': rects}
        sample = Transform(sample)
        image, boxes = sample['image'], sample['boxes']

        if use_gpu:
            image = image.cuda()
            boxes = boxes.cuda()

        with torch.no_grad(): features = extract_features(resnet50_conv, image.unsqueeze(0), boxes.unsqueeze(0), MAPS, Scales)

        if not args.adapt:
            with torch.no_grad(): output = regressor(features)
        else:
            features.required_grad = True
            adapted_regressor = copy.deepcopy(regressor)
            adapted_regressor.train()
            optimizer = optim.Adam(adapted_regressor.parameters(), lr=args.learning_rate)
            for step in range(0, args.gradient_steps):
                optimizer.zero_grad()
                output = adapted_regressor(features)
                lCount = args.weight_mincount * MincountLoss(output, boxes)
                lPerturbation = args.weight_perturbation * PerturbationLoss(output, boxes, sigma=8)
                Loss = lCount + lPerturbation
                # loss can become zero in some cases, where loss is a 0 valued scalar and not a tensor
                # So Perform gradient descent only for non zero cases
                if torch.is_tensor(Loss):
                    Loss.backward()
                    optimizer.step()
            features.required_grad = False
            output = adapted_regressor(features)

        #if im_id == "215_angle8_img.png" or im_id == "325_angle1_img.png" or im_id == "134_angle4_img.png":
        plt.imsave("pred_image_newloss{}.png".format(im_id[0:im_id.find("_")]), np.array(output.to("cpu")).squeeze().squeeze())



        gt_cnt = dots.shape[0]
        pred_cnt = output.sum().item()

        avg_pred += pred_cnt
        
        cnt = cnt + 1

        print(pred_cnt)
        if pred_cnt < min_pred: 
            min_pred = pred_cnt

        if pred_cnt > max_pred: 
            max_pred = pred_cnt
            

        err = abs(gt_cnt - pred_cnt)
        SAE += err
        SSE += err**2

    '''
    avg_pred /= NUM_ANGLES

    if AVG_METHOD == "min":
        pred_cnt = min_pred 
    elif AVG_METHOD == "max":
        pred_cnt = max_pred 
    elif AVG_METHOD == "avg":
        pred_cnt = round(avg_pred, 0)
    else:
        print("ERR")
    '''

    #print(avg_pred, max_pred, min_pred, gt_cnt)
    #sys.exit(1)
        
    #pbar.set_description('{:<8}: actual-predicted: {:6d}, {:6.1f}, error: {:6.1f}. Current MAE: {:5.2f}, RMSE: {:5.2f}'.\
    #                     format(im_id, gt_cnt, pred_cnt, abs(pred_cnt - gt_cnt), SAE/cnt, (SSE/cnt)**0.5))
    print("")

#cnt = cnt / 9
print('On {} data, MAE: {:6.2f}, RMSE: {:6.2f}'.format(args.test_split, SAE/cnt, (SSE/cnt)**0.5))
