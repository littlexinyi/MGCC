# -*- encoding: utf-8 -*-
'''
@File    :   test.py
@Time    :   2023/12/18 11:44:15
@Author  :   Xinyi Wu 
@Version :   1.0
@Contact :   wuxinyi17@nudt.edu.cn
'''
from option.options import options
from data.dataloader import get_dataloader
import torch
from model.model import TextImgPersonReidNet
import os
from utils.read_write_data import write_txt, write_one_row, load_checkpoint, read_json
from metrics.metric import R1_mAP_eval
import time
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import random
import re

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

def pre_caption(caption_list):
    caption_list2 = []
    for caption in caption_list:
        caption = re.sub(
            r"([,.'!?\"()*#:;~])",
            ' ',
            caption.lower(),
        ).replace('-',' ').replace('/',' ').replace('<person>', 'person')

        caption = re.sub(
            r"\s{2,}",
            ' ',
            caption,
        )
        caption = caption.rstrip('\n')
        caption = caption.strip(' ')

        caption_list2.append(caption)

    return caption_list2


def feature_gene(opt, network, gallery_loader, query_loader):
    image_feature_global_list = []
    patch_feature_part_list = []
    image_selected_indices_list = []
    img_labels = []

    for times, [image, label] in enumerate(gallery_loader):
        image = image.to(opt.device)
        label = label.to(opt.device)

        with torch.no_grad():
            if(opt.mode == 'train'):
                img_global_i, patch_part_i  = network.img_embedding(image) 
                selected_indices = torch.tensor([])
            elif(opt.mode == 'test'):
                img_global_i, patch_part_i, selected_indices = network.img_embedding(image)                
                                                #[bs, Rv*49]
        image_feature_global_list.append(img_global_i)
        patch_feature_part_list.append(patch_part_i)
        image_selected_indices_list.append(selected_indices)
        img_labels.append(label.view(-1))

    img_labels = torch.cat(img_labels, 0)
    image_selected_indices = torch.cat(image_selected_indices_list, 0)   

    text_feature_global_list = []
    word_feature_part_list = []
    txt_labels = []
    word_tokens_list = []
    select_tokens_list = []
    attention_weight_list = []
    for times, [text, label] in enumerate(query_loader):
        # text = text.to(opt.device)
        label = label.to(opt.device)

        with torch.no_grad():
            text = list(text)
            text = pre_caption(text)
            if(opt.mode == 'train'):
                text_global_i, word_part_i = network.txt_embedding(text)
                attention_cls_part = torch.tensor([])
                word_tokens = []
                select_token_indices = []
            elif(opt.mode == 'test'):   #TODO 
                text_global_i, word_part_i, attention_cls_part, word_tokens, select_token_indices = network.txt_embedding(text)

        text_feature_global_list.append(text_global_i)
        word_feature_part_list.append(word_part_i)
        attention_weight_list.append(attention_cls_part)
        txt_labels.append(label.view(-1))
        word_tokens_list += word_tokens
        select_tokens_list += select_token_indices

    txt_labels = torch.cat(txt_labels, 0)
    attention_weights = torch.cat(attention_weight_list, 0)
    
    img_labels = img_labels.cpu()
    txt_labels = txt_labels.cpu()

    return text_feature_global_list, word_feature_part_list, image_feature_global_list, patch_feature_part_list, txt_labels, img_labels, image_selected_indices, attention_weights, word_tokens_list, select_tokens_list


def Inference(opt, epoch, network, gallery_loader, query_loader):
    evaluator = R1_mAP_eval(max_rank=20)
    infer_start_t = time.time()
    text_feature_global_list, word_feature_part_list, image_feature_global_list, patch_feature_part_list, txt_labels, img_labels, image_selected_indices,attention_weights,  word_tokens_list, select_tokens_list =  feature_gene(opt, network, gallery_loader, query_loader)
    query_length = len(txt_labels)
    cmc, mAP, mINP = evaluator.compute(network, text_feature_global_list, word_feature_part_list, image_feature_global_list, patch_feature_part_list, txt_labels, img_labels)
    all_infer_time = time.time() - infer_start_t 
    infer_time_per_query = all_infer_time / query_length


    return cmc, mAP, mINP, all_infer_time, infer_time_per_query

t_resize = transforms.Compose([
transforms.Resize((224, 224), Image.BICUBIC),
])

def Image_Token_Selection(background_img_path, attention_mask_indices, save_path):
    image = Image.open(background_img_path)
    image = t_resize(image)
    image = np.asarray(image) 
    #VIT-32
    image_tokens = image.reshape(7, 32, 7, 32, 3).swapaxes(1, 2).reshape(49, 32, 32, 3)
    for i in range(49):
        if i not in attention_mask_indices:    
          image_tokens[i] = 0.2 * image_tokens[i] + 0.8 * 255

    image_t = image_tokens.reshape(7, 7, 32, 32, 3).swapaxes(1, 2).reshape(224, 224, 3)
    img_show = np.concatenate([image] + [image_t], axis=1)
    plt.figure(figsize=(10, 5))
    plt.imshow(img_show)
    plt.savefig(save_path)
    plt.axis('off')    

    return 0

def Text_Token_Visualize(word_labels, attention_weights, select_token_indices, save_path):
    # 绘制热力图
    fig, ax = plt.subplots()
    im = ax.imshow(np.array([attention_weights]), cmap="YlGnBu")            #YlGnBu

    # 添加标签
    fig.set_figheight(3)
    fig.set_figwidth(10)
    ax.tick_params(axis='x', labelsize= 14) # 调整x轴标签字体大小
    ax.set_xticks(np.arange(len(attention_weights)))
    ax.set_yticks([0])
    ax.set_xticklabels(word_labels, rotation=90, fontsize=12)

    for w in select_token_indices:
        ax.get_xticklabels()[w].set_bbox(dict(facecolor='red', edgecolor='red', alpha = 0.3, linewidth=1))
    ax.set_yticklabels([''])
    ax.set_title('Attention Weights of Each Word')

    # 添加热力分布bar
    # cbar = ax.figure.colorbar(im, ax=ax)
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.7)
    cbar.ax.set_ylabel("Heat distribution", rotation=-90, va="bottom")
    #保存图像
    plt.savefig(save_path)
    # 显示图像
    # plt.show()

def rank_token_visualization(opt, network, gallery_loader, query_loader):
    evaluator = R1_mAP_eval(max_rank=20)
    text_feature_global_list, word_feature_part_list, image_feature_global_list, patch_feature_part_list, txt_labels, img_labels, image_selected_indices,word_attention_weights, word_tokens_list, select_tokens_list =  feature_gene(opt, network, gallery_loader, query_loader)
    
    #visualize data generation
    data_save = read_json(os.path.join(opt.dataroot, 'processed_data/test_save.json'))
    img_path_list = [change_path(opt.dataset, os.path.join(opt.dataroot, img_path)) for img_path in data_save['img_path']]
    caption_list = data_save['captions']

    ##Step1: Rank_visualization
    distmat, q_pids, g_pids = evaluator.similarity_compute(network, text_feature_global_list, word_feature_part_list, image_feature_global_list, patch_feature_part_list, txt_labels, img_labels)
    num_q, num_g = distmat.shape
    gallery_indices = np.argsort(-distmat, axis=1)  #[num_q, num_g]  返回相似度从大到小排序后的索引
    gallery_indices_top10 = gallery_indices[:, :10]
    visualization_log = os.path.join(opt.save_path, opt.dataset + '_visualize_log.txt')

    for q_idx in range(num_q):  #遍历每个query 
        write_txt(f"query index: {q_idx}, caption_label: {q_pids[q_idx]}, caption: {caption_list[q_idx]} \n ", visualization_log)
        g_idx_top10 = gallery_indices_top10[q_idx]        #(10, )

        for g_idx in range(10):
            gg = g_idx_top10[g_idx]
            write_txt(f"matched gallery index: {gg}, img_label:{g_pids[gg]}, gallery_img_path: {img_path_list[gg]} \n", visualization_log)
    print("visualization_log successfully generated!")

    if(opt.Topk_Selection):
        ##Step 2: Image Token Selection Visualization
        for i in range(len(img_path_list)):
            print(i)
            print(img_path_list[i])
            #image_selected_indices     #被选中的patch索引
            ww = img_path_list[i].split("/")
            # print(ww)
            img_name = ww[-1]
            if("bmp" in img_name):
                img_name = img_name.replace("bmp", "jpg")
            save_path = os.path.join(opt.save_path, 'image_visualization/')
            os.makedirs(save_path, exist_ok=True)
            save_path = save_path + img_name
            print("save_path:", save_path)
            Image_Token_Selection(img_path_list[i], image_selected_indices[i], save_path)

        # #Step 3: Word Token Selection Visualization

        word_attention_weights = word_attention_weights.detach().cpu()      #[num_texts, 25]

        text_name_list = []
        for q in range(len(img_path_list)):
            ww = img_path_list[q].split("/")[-1]
            text_name = ww.split(".")[0]
            text_name_list.append(text_name+"_1.png")
            text_name_list.append(text_name+"_2.png")

        for j in range(len(word_tokens_list)):
        
            word_labels = word_tokens_list[j]
            word_labels2 = []
            for word_labels_i in word_labels:
                word_labels_i = word_labels_i[:-4]               #去掉</w>
                word_labels2.append(word_labels_i)
            
            if(len(word_labels2) > opt.max_words):
                word_labels2 = word_labels2[:opt.max_words]
            elif(len(word_labels2) < opt.max_words):
                yhi = opt.max_words - len(word_labels2)
                for w in range(yhi):
                    word_labels2.append('<pad>')

            # print("word_labels:", word_labels2)
            attention_weights = word_attention_weights[j].tolist()         #list  len = 25
            # print("attention_weights:",attention_weights)
            select_token_indices = select_tokens_list[j]                # len = 25*Rt
            save_path = os.path.join(opt.save_path, 'text_visualization/')
            os.makedirs(save_path, exist_ok=True)

            save_path = save_path + text_name_list[j]

            Text_Token_Visualize(word_labels2, attention_weights, select_token_indices, save_path)
    
    return 0

def main(opt):
    opt.device = torch.device('cuda:{}'.format(opt.GPU_id))

    opt.save_path = './checkpoints/{}/'.format(opt.dataset) + opt.model_name

    gallery_loader, query_loader = get_dataloader(opt, "test")

    network = TextImgPersonReidNet(opt).to(opt.device)

    print("test_model:", opt.model_name)
    print("test dataset:",opt.dataset)

    checkpoint_feature_path = os.path.join(opt.save_path, 'breakpoint/best_model') 
    checkpoint_epoch_path = os.path.join(opt.save_path, 'breakpoint/current_epoch')
    if os.path.exists(checkpoint_feature_path):
        model_state = load_checkpoint(checkpoint_feature_path)
        # print("model_state:", model_state)
        network.load_state_dict(model_state)
        checkpoint_epoch = torch.load(checkpoint_epoch_path)
        print('Testing: Best Model Checkpoint successfully loaded!')
        print("Best Model Epoch:{}".format(checkpoint_epoch))
        #model test
        network.eval()
        #model_test
        cmc, mAP, mINP, all_infer_time, infer_time_per_query = Inference(opt, 1, network, gallery_loader, query_loader)
        str = "Testing: t2i: @R1: {:.4}, @R5: {:.4}, @R10: {:.4}, mAP: {:.4}, mINP: {:.4}".format(cmc[0], cmc[4], cmc[9], mAP, mINP)
        print(str)
        #rank list visualize & token visualize
        
        rank_token_visualization(opt, network, gallery_loader, query_loader)

        torch.cuda.empty_cache()  
    else:
        print("no saved model!")

if __name__ == '__main__':
    opt = options().opt
    main(opt)





