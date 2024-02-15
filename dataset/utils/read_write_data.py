# -*- coding: utf-8 -*-

import os
import json
import pickle
import os.path as osp
from PIL import Image
def makedir(root):
    if not os.path.exists(root):
        os.makedirs(root)

def write_json(data, root):
    with open(root+'.json', 'w') as f:
        json.dump(data, f, indent=4, separators=(", ", ": "))

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

def read_image(path):
    """Reads image from path using ``PIL.Image``.
    Args:
        path (str): path to an image.

    Returns:
        PIL image to tensor
    """
    got_img = False

    if not osp.exists(path):
        raise IOError('"{}" does not exist'.format(path))
    while not got_img:
        try:
            img = Image.open(path).convert('RGB')
 
            got_img = True
        except IOError:
            print('IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'.format(path))
    return img



