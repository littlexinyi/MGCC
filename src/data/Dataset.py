# -*- encoding: utf-8 -*-
'''
@File    :   Dataset.py
@Time    :   2023/12/18 11:39:13
@Author  :   Xinyi Wu 
@Version :   1.0
@Contact :   wuxinyi17@nudt.edu.cn
'''
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import os
from utils.read_write_data import read_dict,read_json
import cv2
import torchvision.transforms.functional as F
import random


def fliplr(img, dim):
    """
    flip horizontal
    :param img:
    :return:
    """
    inv_idx = torch.arange(img.size(dim) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(dim, inv_idx)
    return img_flip

def change_path(dataset_name, pathss):
    if(dataset_name == 'ICFG-PEDES'):
        if("train" in pathss):
            occlusion_path = pathss.replace('train','train_occlusion_new')
        elif("test" in pathss):
            occlusion_path = pathss.replace('test','test_occlusion_new')
        else:
            occlusion_path = pathss
    elif(dataset_name == 'CUHK-PEDES'):
        if("cam_a" in pathss):
            occlusion_path = pathss.replace('cam_a','cam_a_occlusion_new')
        elif("cam_b" in pathss):
            occlusion_path = pathss.replace('cam_b','cam_b_occlusion_new')
        elif("CUHK01" in pathss):
            occlusion_path = pathss.replace('CUHK01','CUHK01_occlusion_new')
        elif("CUHK03" in pathss):
            occlusion_path = pathss.replace('CUHK03','CUHK03_occlusion_new')
        elif("Market" in pathss):
            occlusion_path = pathss.replace('Market','Market_occlusion_new')
        elif("test_query" in pathss):
            occlusion_path = pathss.replace('test_query','test_query_occlusion_new')
        elif("train_query" in pathss):
            occlusion_path = pathss.replace('train_query','train_query_occlusion_new')            
        else:
            occlusion_path = pathss
    elif(dataset_name == 'RSTPReid'):
        if("imgs" in pathss):
            occlusion_path = pathss.replace('imgs','imgs_occlusion_new')
        else:
            occlusion_path = pathss
    return occlusion_path

class TrainDataset(data.Dataset):
    def __init__(self, opt, tran):

        self.opt = opt
        self.flip_flag = (self.opt.mode == 'train')
        self.dataset_name = opt.dataset
        # data_save = read_dict(os.path.join(opt.dataroot, 'processed_data', opt.mode + '_save.pkl'))
        data_save = read_json(os.path.join(opt.dataroot, 'processed_data', opt.mode + '_save.json'))

        if(opt.train_dt_type == 0):
            print("Train Dataset: Occluded Dataset")
            self.img_path = [change_path(self.dataset_name, os.path.join(opt.dataroot, img_path)) for img_path in data_save['img_path']] #train with occluded datasets
        elif(opt.train_dt_type == 1):
            print("Train Dataset: Holistic Dataset")
            self.img_path = [os.path.join(opt.dataroot, img_path) for img_path in data_save['img_path']]  #train with holistic datasets
        
        self.label = data_save['img_id']

        self.same_id_index = data_save['same_id_index']

        self.transform = tran

        self.max_same_id = 2

        self.num_data = len(self.img_path)
        self.caption = data_save['captions']
        
    def __getitem__(self, index):
        """
        :param index:
        :return: image and its label
        """
        image = Image.open(self.img_path[index])
        image = self.transform(image)
        label = torch.from_numpy(np.array([self.label[index]])).long()

        captions = self.caption[index]
        
        same_id_captions = []
        if(len(self.same_id_index[index]) >= 2):
            same_id_index = random.sample(self.same_id_index[index], 2)         #随机选两个弱匹配文本
            for w in same_id_index:
                same_id_captions.append(self.caption[w])
        else:
            for w in self.same_id_index[index]:
                same_id_captions.append(self.caption[w])
            while(len(same_id_captions) < self.max_same_id):
                same_id_captions.append('0')
        return image, label, captions, same_id_captions

    def get_data(self, index, img=True):
        if img:
            image = Image.open(self.img_path[index])
            image = self.transform(image)
        else:
            image = 0

        label = torch.from_numpy(np.array([self.label[index]])).long()
        
        captions = self.caption[index]
        # caption_code, caption_length = self.caption_mask(self.caption_code[index])

        return image, label, captions

    def __len__(self):
        return self.num_data


class Gallery_img_dateset(data.Dataset):
    def __init__(self, opt, tran, mode):     #mode = test or val

        self.opt = opt

        self.dataset_name = opt.dataset

        
        data_save = read_json(os.path.join(opt.dataroot, 'processed_data', mode + '_save.json'))

        if(opt.test_dt_type == 0):
            print("Test Dataset: Occluded Dataset")
            self.img_path = [change_path(self.dataset_name, os.path.join(opt.dataroot, img_path)) for img_path in data_save['img_path']]

        elif(opt.test_dt_type == 1):
            print("Test Dataset: Holistic Dataset")
            self.img_path = [os.path.join(opt.dataroot, img_path) for img_path in data_save['img_path']]  #test with holistic datasets

        self.label = data_save['img_id']

        self.transform = tran

        self.num_data = len(self.img_path)


    def __getitem__(self, index):
        """
        :param index:
        :return: image and its label
        """

        image = Image.open(self.img_path[index])
        image = self.transform(image)

        label = torch.from_numpy(np.array([self.label[index]])).long()

        return image, label

    def __len__(self):
        return self.num_data


class Query_txt_dateset(data.Dataset):
    def __init__(self, opt, mode):

        self.opt = opt
        # data_save = read_dict(os.path.join(opt.dataroot, 'processed_data', mode + '_save.pkl'))

        data_save = read_json(os.path.join(opt.dataroot, 'processed_data', mode + '_save.json'))

        self.caption = data_save['captions']

        self.label = data_save['caption_id']

        self.num_data = len(self.caption)

    def __getitem__(self, index):

        label = torch.from_numpy(np.array([self.label[index]])).long()

        captions = self.caption[index]

        return captions, label
    def __len__(self):
        return self.num_data





