# -*- encoding: utf-8 -*-
'''
@File    :   bert_text.py
@Time    :   2023/12/18 11:40:12
@Author  :   Xinyi Wu 
@Version :   1.0
@Contact :   wuxinyi17@nudt.edu.cn
'''
import torch
import torch.nn as nn
import os
from transformers import BertModel, BertConfig, BertTokenizer
import torch.nn.functional as F

def l2norm(X, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

#——————构造模型——————
class Text_Bert_Net(nn.Module):
    def __init__(self, opt):            
        super(Text_Bert_Net, self).__init__()
        print("=> using pre-trained model for Text '{}'".format('Bert'))
        self.opt = opt
        self.bert_path = opt.pretrain_path
        modelConfig = BertConfig.from_pretrained(os.path.join(self.bert_path,'bert-base-uncased-config.json'))
        self.textExtractor = BertModel.from_pretrained(os.path.join(self.bert_path, 'bert-base-uncased-pytorch_model.bin'), config= modelConfig)
        embedding_dim = self.textExtractor.config.hidden_size    
             #768
        if(not opt.text_fintune):
          for param in self.textExtractor.parameters():
              param.requires_grad = False    

        self.fc = nn.Sequential(nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, 768, bias=False))
        self.device = opt.device

    def forward(self, texts):
        
        device = self.opt.device
        tokens_tensor, segments_tensors, input_masks_tensors = self.get_tokens(texts)
        tokens = tokens_tensor.to(device)
        segments = segments_tensors.to(device)
        attention_mask = input_masks_tensors.to(device)

        output = self.textExtractor(tokens, token_type_ids=segments,
                                 		attention_mask=attention_mask, output_attentions=True)
        last_hidden_state = output[0]

        text_embeddings = torch.mean(last_hidden_state, dim = 1)
        text_embeddings = self.fc(text_embeddings)   

        text_embeddings = F.normalize(text_embeddings, dim=-1)
        all_word_embeddings = last_hidden_state[:,1:,:]      
        if(self.opt.Topk_Selection):
          attention_map = output.attentions[11]     
          selected_word_embedding = self.top_k_selection(all_word_embeddings, attention_map)
        else:
          selected_word_embedding = all_word_embeddings

        word_embeddings = F.normalize(selected_word_embedding, dim=-1)

        return text_embeddings, word_embeddings
        # return text_embeddings
 
    def top_k_selection(self, all_word_embeddings, attention_map):

        attention_map = attention_map.mean(axis=1)  

        attention_cls_part = attention_map[:, 0, 1:]    
        attention_cls_part = attention_cls_part.squeeze()   

        sorted, indices = torch.sort(attention_cls_part, descending=True)
        Rt = self.opt.Rt
        select_token = round(Rt * self.opt.max_token_length)    
        indices = indices[:,0:select_token]
        selected_word_embedding = []
        for i in range(indices.size(0)):
          all_word_embeddings_i = all_word_embeddings[i, :,:].squeeze()
          top_k_embedding = torch.index_select(all_word_embeddings_i, 0, indices[i])
          top_k_embedding = top_k_embedding.unsqueeze(0)
          selected_word_embedding.append(top_k_embedding)
        selected_word_embedding = torch.cat(selected_word_embedding, 0)  

        return selected_word_embedding


    def get_tokens(self, texts):      
          tokenizer = BertTokenizer.from_pretrained(os.path.join(self.bert_path, 'bert-base-uncased-vocab.txt'))
          tokens, segments, input_masks = [], [], []
          for text in texts:
            bert_input = tokenizer(text,
                       padding = True,
                       truncation=True,
                       return_tensors="pt",
                       max_length=100)

            indexed_tokens = list(bert_input.input_ids[0])
            tokens.append(indexed_tokens) 
            segments.append([0] * len(indexed_tokens))            
            input_masks.append([1] * len(indexed_tokens))

          max_length = 101
          for j in range(len(tokens)): 
              padding = [0] * (max_length - len(tokens[j]))
              tokens[j] += padding
              segments[j] += padding
              input_masks[j] += padding

          tokens_tensor = torch.tensor(tokens)        
          segments_tensors = torch.tensor(segments)
          input_masks_tensors = torch.tensor(input_masks)
          return tokens_tensor, segments_tensors, input_masks_tensors
