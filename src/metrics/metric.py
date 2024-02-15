# -*- encoding: utf-8 -*-
'''
@File    :   metric.py
@Time    :   2023/12/18 11:39:53
@Author  :   Xinyi Wu 
@Version :   1.0
@Contact :   wuxinyi17@nudt.edu.cn
'''
import torch
import numpy as np
import os
from utils.read_write_data import write_txt

def cosine_similarity(qf, gf):

    epsilon = 0.00001
    # dist_mat = qf.mm(gf.t())   #torch.mm(a,b)  矩阵矢量相乘，行*列相加和，等价于torch.matmul   与torch.mul(a,b)不同   矩阵对应元素相乘，维度必须相等
    # #求指定维度上的2范数
    # qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    # gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    # qg_normdot = qf_norm.mm(gf_norm.t())
    # #.mul 按位乘  .mm 矩阵乘
    # dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    #等效代码
    dist_mat = torch.matmul(qf, torch.t(gf)) 
    dist_mat = dist_mat.cpu().numpy()

    #按照最大最小值裁剪数组
    # dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    # dist_mat = np.arccos(dist_mat)    #对数组元素求其反余弦值   cos相似度越逼近1，向量夹角越小，二者越相似，距离应该越小。
    return dist_mat

def eval_func(distmat, q_pids, g_pids, max_rank=20):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    ## max_rank，统计gallery可能性最大的前多少个
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(-distmat, axis=1)  #[num_q, num_g]
    #相似度从大到小排序后的索引值的数组.对于每个query，排序其相似gallery。
    #  3 1 2 0
    #  0 3 2 1
    # 将每行排序后的预测标签gallery和真实标签query比较，一致则为1，不一致则为0   matches 每行对应与该query相似度排序的gallery样本是否为同一ID
    # matches 维度 [num_q, num_g] 为T/F的数组
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []  #原始的方法，存储所有query的[000111...]数组
    all_AP = []   # 存储所有query的AP
    all_INP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):  #遍历每个query 

        orig_cmc = matches[q_idx]       #1*num_gallery
        if not np.any(orig_cmc):    #orig_cmc中没有满足条件的
            # this condition is true when query identity does not appear in gallery，则跳过该query，称为无效query
            continue

        cmc = orig_cmc.cumsum()     #当前列之前的和加到当前列上
        pos_idx = np.where(orig_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx] / (pos_max_idx + 1.0)
        all_INP.append(inp)
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.   #有效query个数

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        # tmp_cmc = [x /(i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    #assert 用于判断一个表达式，在该表达式为false时触发异常，即num_vaild_q = 0时，没有有效的query

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q      #all_cmc列表里所有元素对应位置相加 再除以总Query个数
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)

    
    return all_cmc, mAP, mINP


class R1_mAP_eval():
    
    def __init__(self,max_rank=20):
        super(R1_mAP_eval, self).__init__()
        self.max_rank = max_rank

    def similarity_compute(self, model, query_features_list, query_part_features_list, gallery_features_list, gallery_part_features_list, q_pids, g_pids):
        q_pids = np.asarray(q_pids.cpu())      #  一维数组
        g_pids = np.asarray(g_pids.cpu())      
 
        query_len = len(query_features_list)
        gallery_len = len(gallery_features_list)      
        sim_list = []
        for i in range(query_len):
            query_global = query_features_list[i] 
            query_part = query_part_features_list[i]
            each_row = []
            for j in range(gallery_len):
                gallery_global = gallery_features_list[j]   #[bs_gallery, 768]
                gallery_part = gallery_part_features_list[j]    #[bs_gallery, m, 768]

                # sim_logits = get_similarity(query_global, query_part, gallery_global, gallery_part)   #query(text) to gallery(img)
                sim_logits2 = model.get_similarity(gallery_global, gallery_part, query_global, query_part).t()
                # sim_logits = sim_logits.cpu().detach().numpy()
                sim_logits2 = sim_logits2.cpu().detach().numpy()

                each_row.append(sim_logits2)
            each_row = np.concatenate(tuple(each_row), axis = -1)        #列拼接        
            sim_list.append(each_row)

        distmat2 = np.concatenate(tuple(sim_list), axis = 0)        

        return distmat2, q_pids, g_pids

    def compute(self, model, query_features_list, query_part_features_list, gallery_features_list, gallery_part_features_list, q_pids, g_pids):  # called after each epoch

        q_pids = np.asarray(q_pids.cpu())      #  一维数组
        g_pids = np.asarray(g_pids.cpu())      
 
        query_len = len(query_features_list)
        gallery_len = len(gallery_features_list)      
        sim_list = []
        for i in range(query_len):
            query_global = query_features_list[i] 
            query_part = query_part_features_list[i]
            each_row = []
            for j in range(gallery_len):
                gallery_global = gallery_features_list[j]   #[bs_gallery, 768]
                gallery_part = gallery_part_features_list[j]    #[bs_gallery, m, 768]

                # sim_logits = get_similarity(query_global, query_part, gallery_global, gallery_part)   #query(text) to gallery(img)
                sim_logits2 = model.get_similarity(gallery_global, gallery_part, query_global, query_part).t()
                # sim_logits = sim_logits.cpu().detach().numpy()
                sim_logits2 = sim_logits2.cpu().detach().numpy()

                each_row.append(sim_logits2)
            each_row = np.concatenate(tuple(each_row), axis = -1)        #列拼接        
            sim_list.append(each_row)

        distmat2 = np.concatenate(tuple(sim_list), axis = 0)  

        cmc, mAP, mINP = eval_func(distmat2, q_pids, g_pids)

        return cmc, mAP, mINP


