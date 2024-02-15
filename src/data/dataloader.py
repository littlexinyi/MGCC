# -*- encoding: utf-8 -*-
'''
@File    :   dataloader.py
@Time    :   2023/12/18 11:39:20
@Author  :   Xinyi Wu 
@Version :   1.0
@Contact :   wuxinyi17@nudt.edu.cn
'''
from torchvision import transforms
from PIL import Image
import torch
from data.Dataset import TrainDataset, Gallery_img_dateset, Query_txt_dateset

__factory = {
    'ICFG-PEDES': [TrainDataset, Gallery_img_dateset, Query_txt_dateset],
    'CUHK-PEDES': [TrainDataset, Gallery_img_dateset, Query_txt_dateset],
    'RSTPReid': [TrainDataset, Gallery_img_dateset, Query_txt_dateset]    
}

def get_dataloader(opt,mode):
    """
    tranforms the image, downloads the image with the id by data.DataLoader
    """

    if mode == 'train':
        transform_list = [
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224), Image.BICUBIC),   # interpolation
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))]
        tran = transforms.Compose(transform_list)

        train_dataset = __factory[opt.dataset][0](opt, tran)

        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size,
                                                 shuffle=True, drop_last=False, num_workers=4)
        # img_num, txt_num = dataset.__len__()
        print('{}-{} has {} <img,text> pairs'.format(opt.dataset, mode, len(train_dataset)))

        return dataloader

    else:       #train / val 
        tran = transforms.Compose([
            transforms.Resize((224, 224), Image.BICUBIC),  # interpolation
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))]
        )

        # img_dataset = CUHKPEDE_img_dateset(opt, tran, mode)
        img_dataset = __factory[opt.dataset][1](opt, tran, mode)

        img_dataloader = torch.utils.data.DataLoader(img_dataset, batch_size=opt.batch_size,
                                                 shuffle=False, drop_last=False, num_workers=4)

        txt_dataset = __factory[opt.dataset][2](opt, mode)

        txt_dataloader = torch.utils.data.DataLoader(txt_dataset, batch_size=opt.batch_size,
                                                 shuffle=False, drop_last=False, num_workers=4)

        print('{}-{} has {} photos, {} texts'.format(opt.dataset, mode, len(img_dataset), len(txt_dataset)))

        return img_dataloader, txt_dataloader
