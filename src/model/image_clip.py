# -*- encoding: utf-8 -*-
'''
@File    :   image_clip.py
@Time    :   2023/12/18 11:41:19
@Author  :   Xinyi Wu 
@Version :   1.0
@Contact :   wuxinyi17@nudt.edu.cn
'''
import torch
import torch.nn as nn
import numpy as np
from transformers import CLIPConfig, CLIPModel, CLIPVisionModel
from transformers import logging
import os
import torch.nn.functional as F

class Image_Clip(nn.Module):
    def __init__(self, opt):
        super(Image_Clip, self).__init__()
        print("=> using pre-trained model for Image '{}'".format('Clip'))
        self.clip_path = opt.pretrain_path
        if(opt.max_patch == 49):
            self.clip_model = CLIPModel.from_pretrained(os.path.join(self.clip_path,'clip-vit-base-patch32'))
        elif(opt.max_patch == 196):
            self.clip_model = CLIPModel.from_pretrained(os.path.join(self.clip_path,'clip-vit-base-patch16'))   
        self.cnn = self.clip_model.vision_model

        self.opt = opt
        self.finetune = opt.image_fintune

        if(not self.finetune):
            for param in self.cnn.parameters():
                param.requires_grad = False

    def forward(self, images):
        """Extract image feature vectors."""
        features = self.cnn(images, output_attentions = True)
        visual_tokens = features.last_hidden_state        
        all_part_tokens = visual_tokens[:,1:,:] 

        pooler_output = features.pooler_output
        pooler_output = F.normalize(pooler_output, dim=-1)

        selected_indices = []

        if(self.opt.Topk_Selection):
            attention_map = features.attentions[11]     #[bs, Rv*49]
            select_part_tokens, selected_indices = self.top_k_selection(all_part_tokens, attention_map)
        else:                   
            select_part_tokens = all_part_tokens
            selected_indices = [[i for i in range(49)] for j in range(self.opt.batch_size)]  
            selected_indices = torch.tensor(selected_indices)

        part_tokens =  F.normalize(select_part_tokens, dim=-1)
        
        if(self.opt.mode == 'test'):
            return pooler_output, part_tokens, selected_indices
        elif(self.opt.mode == 'train'):
            return pooler_output, part_tokens
        # return pooler_output

    def top_k_selection(self, all_patch_embeddings, attention_map):

        attention_map = attention_map.mean(axis=1)

        attention_cls_part = attention_map[:, 0, 1:]       
        attention_cls_part = attention_cls_part.squeeze() 

        sorted, indices = torch.sort(attention_cls_part, descending=True)
        Rv = self.opt.Rv
        select_token = round(Rv * self.opt.max_patch)    
        selected_indices = indices[:,0:select_token]
        selected_patch_embedding = []
        for i in range(selected_indices.size(0)):   #bs
          all_patch_embeddings_i = all_patch_embeddings[i, :,:].squeeze()
          top_k_embedding = torch.index_select(all_patch_embeddings_i, 0, selected_indices[i])
          top_k_embedding = top_k_embedding.unsqueeze(0)
          selected_patch_embedding.append(top_k_embedding)
        selected_patch_embedding = torch.cat(selected_patch_embedding, 0)

        return selected_patch_embedding, selected_indices

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)
