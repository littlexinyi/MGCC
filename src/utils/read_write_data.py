# -*- encoding: utf-8 -*-
'''
@File    :   read_write_data.py
@Time    :   2023/12/18 11:43:14
@Author  :   Xinyi Wu 
@Version :   1.0
@Contact :   wuxinyi17@nudt.edu.cn
'''
import os
import json
import pickle
import csv
import torch

def save_checkpoint(state, checkpoint_folder, save_id_loss = False):
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    torch.save(state['model'], os.path.join(checkpoint_folder, 'best_model'))
    torch.save(state['epoch'], os.path.join(checkpoint_folder, 'current_epoch'))
    torch.save(state['test_best'], os.path.join(checkpoint_folder, 'test_best'))
    if(save_id_loss):
        torch.save(state['id_loss'], os.path.join(checkpoint_folder, 'id_loss'))

def load_checkpoint(path):

    model_state = torch.load(path)
    return model_state

def makedir(root):
    if not os.path.exists(root):
        os.makedirs(root)


def write_json(data, root):
    with open(root, 'w') as f:
        json.dump(data, f)


def read_json(root):
    with open(root, 'r') as f:
        data = json.load(f)

    return data


def read_dict(root):
    with open(root, 'rb') as f:
        data = pickle.load(f)

    return data


def save_dict(data, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def write_txt(data, name):
    with open(name, 'a') as f:
        f.write(data)
        f.write('\n')

def write_one_row(one_raw, save_path, save_file_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(save_path, save_file_name) + '.csv', 'a', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(one_raw)




