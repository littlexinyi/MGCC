# -*- encoding: utf-8 -*-
'''
@File    :   options.py
@Time    :   2023/12/18 11:43:29
@Author  :   Xinyi Wu 
@Version :   1.0
@Contact :   wuxinyi17@nudt.edu.cn
'''
import argparse
import torch
import logging
import os
from utils.read_write_data import makedir

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class options():
    def __init__(self):
        self._par = argparse.ArgumentParser(description='options for Deep Cross Modal')
        #base parameters
        self._par.add_argument('--model_name', type=str, default='experiment', help='experiment name')
        self._par.add_argument('--mode', type=str, default='train', help='choose mode [train or test]')
        self._par.add_argument('--GPU_id', type=str, default='0', help='choose GPU ID [0,1,2,3,4,5,6,7]')
        self._par.add_argument('--device', type=str, default='', help='cuda device')
        self._par.add_argument('--dataset', type=str, default='CUHK-PEDES', help='choose the dataset [CUHK-PEDES, ICFG-PEDES, RSTPReid]')
        self._par.add_argument('--dataroot', type=str,  help='root path of the Dataset')
        self._par.add_argument('--save_path', type=str, default='./checkpoints/', help='save the result during training')
        self._par.add_argument('--train_dt_type', type= int, default = 0, help='train dataset: 0 or 1, 0 means occluded, 1 means holistic')
        self._par.add_argument('--test_dt_type', type= int, default = 0, help='test dataset: 0 or 1, 0 means occluded, 1 means holistic')

        #training parameters
        self._par.add_argument('--batch_size', type=int, default=64, help='batch size')
        self._par.add_argument('--epoch', type=int, default=60, help='train epoch')
        self._par.add_argument('--epoch_decay', type=list, default=[20, 40], help='decay epoch') 

        self._par.add_argument('--adam_alpha', type=float, default=0.9, help='momentum term of adam')
        self._par.add_argument('--adam_beta', type=float, default=0.999, help='momentum term of adam')
        self._par.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        self._par.add_argument('--feature_length', type=int, default=768, help='the length of feature')
        self._par.add_argument('--class_num', type=int, default=11000,
                               help='num of classes of datasets for ID Classification')      #CUHK 11000   ICFG 3102   RSTP 3701

        self._par.add_argument('--loss_type',  type = str, default='InfoNCE', help='use loss type,choice [Rank, InfoNCE]') 
        self._par.add_argument('--cr_loss', action="store_true", help='use cr loss or not')          
        self._par.add_argument('--ID_LOSS', action="store_true", help='use ID loss or not')       
        self._par.add_argument('--epoch_begin', type=int, default=5, help='when calculate the auto margin')
        self._par.add_argument('--margin', type=float, default=0.2, help='ranking loss margin')
        self._par.add_argument('--cr_beta', type=float, default=0.1, help='ranking loss margin')


        #Pretrain model
        self._par.add_argument('--pretrain_path', type=str, default = './pretrain/', help='data root of the Data')
        self._par.add_argument('--img_model', type = str, default='CLIP', help='img backbone choice [ResNet50, CLIP]')
        self._par.add_argument('--txt_model', type = str, default='CLIP', help='txt backbone choice [Bert, CLIP]')
        self._par.add_argument('--image_fintune', action="store_true", help='fintune img backbone or not')   #不输入默认为False
        self._par.add_argument('--text_fintune', action="store_true", help='fintune text backbone or not')    #不输入默认为False

        #MGCC Module
        self._par.add_argument('--img_text_logits', action="store_true")    
        self._par.add_argument('--img_word_logits', action="store_true")    
        self._par.add_argument('--patch_text_logits', action="store_true")   
        self._par.add_argument('--patch_word_logits', action="store_true")    
        self._par.add_argument('--Aggregation_Type', type = str, default='Attention', help='Aggregation type choice [Attention, Max_Max, Max_Mean, Mean_Mean, Mean_Max]')
        self._par.add_argument('--softmax_t', type=float, default=1e-2, help='the temperature of softmax')
        
        #Token Selection Module
        self._par.add_argument('--Topk_Selection', action="store_true", help='use the top_k tokens or not')          
        self._par.add_argument('--max_token_length', type=int, default=76, help='the length of max word tokens, 76 for CLIP-Text, 100 for Bert')
        self._par.add_argument('--Rt', type=float, default=0.4, help='the radio of top_k')
        self._par.add_argument('--Rv', type=float, default=0.2, help='the radio of top_k')        
        self._par.add_argument('--max_patch', type=int, default=49, help='the length of max patch') #vit-32: 49;vit-16:196
        self._par.add_argument('--max_words', type=int, default=25, help='the length of max words') #we unify the length of texts to 25


        self.opt = self._par.parse_args()

        self.opt.device = torch.device('cuda:{}'.format(self.opt.GPU_id[0]))


def config(opt):

    log_config(opt)
    model_root = os.path.join(opt.save_path, 'breakpoint')
    if os.path.exists(model_root) is False:
        makedir(model_root)


def log_config(opt):
    logroot = os.path.join(opt.save_path, 'log')
    if os.path.exists(logroot) is False:
        makedir(logroot)
    filename = os.path.join(logroot, opt.mode + '.log')
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(handler)
    if opt.mode != 'test':
        logger.info(opt)



