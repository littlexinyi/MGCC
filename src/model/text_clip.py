# -*- encoding: utf-8 -*-
'''
@File    :   text_clip.py
@Time    :   2023/12/18 11:42:24
@Author  :   Xinyi Wu 
@Version :   1.0
@Contact :   wuxinyi17@nudt.edu.cn
'''
import torch
import torch.nn as nn
import numpy as np
from transformers import CLIPConfig, CLIPModel, CLIPTokenizer
from transformers import logging
import os
import torch.nn.functional as F

def account_len(np_a):
    lens = 0
    for w in range(np_a.shape[0]):
        if(not np_a[w] == 49407):
            lens+=1
        elif(np_a[w] == 49407):
            break
    return lens

class Text_Clip(nn.Module):
    def __init__(self, opt):
        super(Text_Clip, self).__init__()
        print("=> using pre-trained model for Text '{}'".format('Clip'))

        self.clip_path = opt.pretrain_path
        if(opt.max_patch == 49):
            clip_model = CLIPModel.from_pretrained(os.path.join(self.clip_path,'clip-vit-base-patch32'))
            self.tokenizer = CLIPTokenizer.from_pretrained(os.path.join(self.clip_path,'clip-vit-base-patch32'))
        elif(opt.max_patch == 196):
            clip_model = CLIPModel.from_pretrained(os.path.join(self.clip_path,'clip-vit-base-patch16'))
            self.tokenizer = CLIPTokenizer.from_pretrained(os.path.join(self.clip_path,'clip-vit-base-patch16'))   
                     
        self.textExtractor = clip_model.text_model
        # layer norm，最后一层加了归一化
        self.fc = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 768, bias=False))
        self.opt = opt
        self.finetune = opt.text_fintune
        self.device = opt.device
        if(not self.finetune):
            for param in self.textExtractor.parameters():
                param.requires_grad = False

    def forward(self, texts):       #每个batch的数据
        texts = list(texts)
        encoded_input = self.tokenizer(texts, padding = 'max_length', truncation = True, max_length = 77, return_tensors = "pt")

        tokens = encoded_input["input_ids"]
        token_len_list = []
        token_len_max = 0
        np_tokens = np.array(tokens)
        for w in range(np_tokens.shape[0]):
            token_lens = account_len(np_tokens[w])
            token_len_list.append(token_lens)

        tokens = tokens.to(self.device)
        attention_mask = encoded_input["attention_mask"]            
        attention_mask = attention_mask.to(self.device)

        features = self.textExtractor(tokens, output_attentions = True)  

        text_tokens = features.last_hidden_state    #[bz,77,512]
        # cls_token = text_tokens[:,0,:]     
        part_tokens = text_tokens[:,1:,:]  

        pooler_output = features.pooler_output      #[bz, 512]         
        pooler_output = self.fc(pooler_output)
        text_embeddings = F.normalize(pooler_output, dim=-1)
        all_part_tokens = self.fc(part_tokens) 
        if(self.opt.Topk_Selection):
            attention_map = features.attentions[11]

            select_part_tokens, attention_cls_part, selected_indices = self.top_k_selection(all_part_tokens, attention_map, token_len_list)
            
            word_tokens_list = []
            selected_indices_list = selected_indices.tolist()

            for i in range(len(texts)):
                word_tokens_i = self.tokenizer.tokenize(texts[i])
                word_tokens_list.append(word_tokens_i)
        else:
            select_part_tokens = all_part_tokens

        word_embeddings =  F.normalize(select_part_tokens, dim=-1) 

        if(self.opt.mode == 'test'):
            if(self.opt.Topk_Selection):
                return text_embeddings, word_embeddings, attention_cls_part, word_tokens_list, selected_indices_list
            else:
                return text_embeddings, word_embeddings
        elif(self.opt.mode == 'train'):
            return text_embeddings, word_embeddings

    def top_k_selection(self, all_word_embeddings, attention_map, token_len_list):
        attention_map = attention_map.mean(axis=1)    
        bs = attention_map.size(0)
        selected_word_embedding = []
        attention_weights_list = []
        selected_indices_list = []
        for i in range(bs):
            attention_cls_part = attention_map[i, token_len_list[i]]          
            attention_cls_part = attention_cls_part[1:token_len_list[i]]
            len_attention_cls_part = attention_cls_part.shape[0]          

            all_word_embeddings_i = all_word_embeddings[i, :,:].squeeze()      
            all_word_embeddings_i = all_word_embeddings_i[1:token_len_list[i],:]  

            sort1, indices = torch.sort(attention_cls_part, descending=True)
            
            if(len_attention_cls_part > self.opt.max_words):
                selected_indices = indices[: self.opt.max_words]
                attention_cls_part = torch.index_select(attention_cls_part, 0, selected_indices)
                all_word_embeddings_i = torch.index_select(all_word_embeddings_i, 0, selected_indices)

            elif(len_attention_cls_part < self.opt.max_words):
                yhi = self.opt.max_words - len_attention_cls_part
                add_yhi = torch.zeros(yhi).to(self.device)
                attention_cls_part = torch.cat((attention_cls_part, add_yhi), dim = 0)

        
            sort2, indices2 = torch.sort(attention_cls_part, descending=True)    
            Rt = self.opt.Rt
            select_token = round(Rt * self.opt.max_words) 
            selected_indices2 = indices2[:select_token]

            attention_weights_list.append(attention_cls_part.unsqueeze(0))
            selected_indices_list.append(selected_indices2.unsqueeze(0))    

            top_k_embedding = torch.index_select(all_word_embeddings_i, 0, selected_indices2)
            top_k_embedding = top_k_embedding.unsqueeze(0)
            selected_word_embedding.append(top_k_embedding)

        attention_weights_last = torch.cat(attention_weights_list, 0)   
        selected_indices_last = torch.cat(selected_indices_list, 0)          
        selected_word_embedding = torch.cat(selected_word_embedding, 0)   

        return selected_word_embedding, attention_weights_last, selected_indices_last

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

