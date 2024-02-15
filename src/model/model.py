# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2023/12/18 11:41:55
@Author  :   Xinyi Wu 
@Version :   1.0
@Contact :   wuxinyi17@nudt.edu.cn
'''
from torch import nn
from torchvision import models
import torch
from torch.nn import init
from torch.nn import functional as F
from model.bert_text import Text_Bert_Net
from model.image_clip import Image_Clip
from model.text_clip import Text_Clip

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1)
        init.constant_(m.bias.data, 0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        init.constant(m.bias.data, 0.0)

class conv(nn.Module):

    def __init__(self, input_dim, output_dim, relu=False, BN=False):
        super(conv, self).__init__()

        block = []
        block += [nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False)]

        if BN:
            block += [nn.BatchNorm2d(output_dim)]
        if relu:
            block += [nn.LeakyReLU(0.25, inplace=True)]

        self.block = nn.Sequential(*block)
        self.block.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.block(x)
        x = x.squeeze(3).squeeze(2)
        return x

class TextImgPersonReidNet(nn.Module):

    def __init__(self, opt):
        super(TextImgPersonReidNet, self).__init__()

        self.opt = opt
        if(opt.img_model == 'ResNet50'):
            print("=> using pre-trained model '{}' as img backbone".format('ResNet50'))
            resnet50 = models.resnet50(pretrained=True)
            self.ImageExtract = nn.Sequential(*(list(resnet50.children())[:-2]))
            self.global_avgpool = nn.AdaptiveMaxPool2d((1, 1))
            self.conv_global = conv(2048, opt.feature_length)           #feature_length 768

        elif(opt.img_model == 'CLIP'):
            self.ImageExtract = Image_Clip(opt)

        if(opt.txt_model == 'Bert'):
            self.TextExtract = Text_Bert_Net(opt)      #Bert
        elif(opt.txt_model == 'CLIP'):
            self.TextExtract = Text_Clip(opt) 

        #coarse_grained weights
        if(self.opt.img_text_logits):
            self.global_mat_weight = nn.parameter.Parameter(torch.eye(opt.feature_length), requires_grad=True)

        bs = self.opt.batch_size
        if(self.opt.Topk_Selection):
            max_length_batch =  round(self.opt.Rt * self.opt.max_words)
            num_patch = round(self.opt.Rv * self.opt.max_patch)
        else:
            max_length_batch = self.opt.max_token_length
            num_patch = self.opt.max_patch

        #cross_grained weights
        if(self.opt.img_word_logits):
            self.word_logit_weight = nn.parameter.Parameter(torch.eye(max_length_batch), requires_grad=True)
        if(self.opt.patch_text_logits):
            self.patch_logit_weight = nn.parameter.Parameter(torch.eye(num_patch), requires_grad=True)
        #fine_grained weights
        if((self.opt.patch_word_logits == True) and (self.opt.Aggregation_Type == 'Attention')):
            self.local_mat_weight = nn.parameter.Parameter(torch.eye(opt.feature_length), requires_grad=True)
            self.patch_mat_weight =  nn.parameter.Parameter(torch.eye(num_patch), requires_grad=True)
            self.word_mat_weight = nn.parameter.Parameter(torch.eye(max_length_batch), requires_grad = True)
            self.patch_mat_weight2 = nn.parameter.Parameter(torch.eye(num_patch), requires_grad=True)
            self.word_mat_weight2 = nn.parameter.Parameter(torch.eye(max_length_batch), requires_grad=True)

    def forward(self, image, captions):

        img_global, patch_part  = self.img_embedding(image)

        txt_global, word_part = self.txt_embedding(captions)          

        return img_global, patch_part, txt_global, word_part

    def img_embedding(self, image):
        if(self.opt.mode == 'train'):
            if(self.opt.img_model == 'ResNet50'):
                image_global = self.ImageExtract(image)
                image_global = self.global_avgpool(image_global)
                image_global = self.conv_global(image_global)
                patch_part = 0
            else:
                image_global, patch_part = self.ImageExtract(image)
            return image_global, patch_part
        
        elif(self.opt.mode == 'test'):
            image_global, patch_part, selected_indices = self.ImageExtract(image) 
            return image_global, patch_part, selected_indices

    def txt_embedding(self, captions):   
        if(self.opt.mode == 'train'):        
            text_global, word_part = self.TextExtract(captions)        #[bz, embedding]
            return text_global, word_part
        elif(self.opt.mode == 'test'):
            if(self.opt.Topk_Selection):
                text_global, word_part, attention_cls_part, word_tokens, selected_indices = self.TextExtract(captions)        #[bz, embedding]
                return text_global, word_part, attention_cls_part, word_tokens, selected_indices      
            else:
                text_global, word_part = self.TextExtract(captions)        #[bz, embedding]
                return text_global, word_part               


    def get_similarity(self, img_global, patch_part, txt_global, word_part):
    
        if(self.opt.img_model == 'CLIP'):
            logit_scale = self.ImageExtract.clip_model.logit_scale.exp()        #np.log(1 / 0.07).exp()
        else:
            logit_scale = 1

        total_logits = []
        #single_grained_logits
        if(self.opt.img_text_logits):
            img_text_logits = logit_scale * torch.matmul(torch.matmul(img_global, self.global_mat_weight), torch.t(txt_global))       #[bz,bz]
            total_logits.append(img_text_logits)
        softmax_t = self.opt.softmax_t
        #coarse_grained_logits
        if(self.opt.img_word_logits):
            img_word_logits = logit_scale * torch.sum(torch.matmul(img_global, word_part.permute(0,2,1)) \
                * torch.matmul(torch.softmax(torch.matmul(img_global, word_part.permute(0,2,1)) / softmax_t, dim=-1), self.word_logit_weight), dim = -1).t()
            total_logits.append(img_word_logits)
        
        if(self.opt.patch_text_logits):
            patch_text_logits = logit_scale * torch.sum(torch.matmul(patch_part, txt_global.t()) \
                * torch.matmul(torch.softmax(torch.matmul(patch_part, txt_global.t()) / softmax_t, dim=1).permute(0,2,1), self.patch_logit_weight).permute(0,2,1), dim = 1)
            total_logits.append(patch_text_logits)
        # #fine_grained logits
        if(self.opt.patch_word_logits):
            patch_word_logits = logit_scale * self.aggregation_fine_grained_similarity(patch_part, word_part)
            total_logits.append(patch_word_logits)


        sim_i_2_t = sum(total_logits) / len(total_logits)

        return sim_i_2_t

    def aggregation_fine_grained_similarity(self, patch_part, word_part):
        bs_img, num_patch, dim = patch_part.shape
        bs_text, max_length_batch, dim = word_part.shape
        fine_grained_sim_scores = torch.matmul(patch_part.view(-1, dim), word_part.view(-1, dim).t()).view(bs_img, num_patch, bs_text, max_length_batch) 
        if(self.opt.Aggregation_Type == 'Attention'):
            softmax_t = self.opt.softmax_t
            fine_grained_sim_scores = torch.matmul(torch.matmul(patch_part.view(-1, dim), self.local_mat_weight), word_part.view(-1, dim).t()).view(bs_img, num_patch, bs_text, max_length_batch)    #[bs_img, num_patch, bs_text, max_length_batch]        
            word_level_logit = torch.sum(torch.matmul(torch.softmax(fine_grained_sim_scores/softmax_t, dim=1).permute(0,2,3,1), self.patch_mat_weight).permute(0,3,1,2) * fine_grained_sim_scores, dim = 1) #[bs_img, bs_text, max_length_batch]
            patch_level_logit = torch.sum(torch.matmul(torch.softmax(fine_grained_sim_scores/softmax_t, dim = -1), self.word_mat_weight) * fine_grained_sim_scores, dim = -1)      #[bs_img, num_patch, bs_text]

            word_level_logit2 = torch.sum(torch.matmul(torch.softmax(word_level_logit/softmax_t, dim=-1), self.word_mat_weight2) * word_level_logit, dim=-1)                                         #[bs_img, bs_text]
            patch_level_logit2 = torch.sum(torch.matmul(torch.softmax(patch_level_logit/softmax_t, dim=1).permute(0,2,1), self.patch_mat_weight2).permute(0,2,1) * patch_level_logit, dim=1)     #[bs_img, bs_text]

        if(self.opt.Aggregation_Type == "Max_Mean"):
            word_level_logit, index1 = torch.max(fine_grained_sim_scores, dim = 1)    
            patch_level_logit, index2 = torch.max(fine_grained_sim_scores, dim = -1)
            word_level_logit2 = torch.mean(word_level_logit, dim = -1) 
            patch_level_logit2 = torch.mean(patch_level_logit, dim = 1)  
        if(self.opt.Aggregation_Type == "Max_Max"):
            word_level_logit, index1 = torch.max(fine_grained_sim_scores, dim = 1)
            patch_level_logit, index2 = torch.max(fine_grained_sim_scores, dim = -1)
            word_level_logit2, index3 = torch.max(word_level_logit, dim = -1)
            patch_level_logit2, index4 = torch.max(patch_level_logit, dim = 1)
        if(self.opt.Aggregation_Type == "Mean_Mean"):
            word_level_logit = torch.mean(fine_grained_sim_scores, dim = 1)
            patch_level_logit = torch.mean(fine_grained_sim_scores, dim = -1)
            word_level_logit2 = torch.mean(word_level_logit, dim = -1)
            patch_level_logit2 = torch.mean(patch_level_logit, dim = 1)

        if(self.opt.Aggregation_Type == "Mean_Max"):
            word_level_logit = torch.mean(fine_grained_sim_scores, dim = 1)
            patch_level_logit = torch.mean(fine_grained_sim_scores, dim = -1)
            word_level_logit2, index1 = torch.max(word_level_logit, dim = -1)
            patch_level_logit2, index2 = torch.max(patch_level_logit, dim = 1)

        return (word_level_logit2 + patch_level_logit2) / 2
