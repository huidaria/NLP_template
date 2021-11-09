#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：emhui 
@File    ：util.py.py
@Author  ：wanghuifen
@Date    ：2021/11/2 21:31 
@url     ：https://pypi.tuna.tsinghua.edu.cn/simple
'''

import json
import torch
# import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import errno
import os
import os.path as op
import yaml
import random
import numpy as np
from omegaconf import OmegaConf

def mkdir(path):
    # if it is the current folder, skip.
    if path == '':
        return
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def load_config_file(file_path):
    with open(file_path, 'r') as fp:
        return OmegaConf.load(fp)

def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def load_from_yaml_file(yaml_file):
    with open(yaml_file, 'r') as fp:
        return yaml.load(fp, Loader=yaml.FullLoader)


def find_file_path_in_yaml(fname, root):
    if fname is not None:
        if op.isfile(fname):
            return fname
        elif op.isfile(op.join(root, fname)):
            return op.join(root, fname)
        else:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), op.join(root, fname)
            )


def ensure_dir(dirname):
    '''

    :param dirname:
    :return: #创建文件夹
    '''
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)



def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


'''
数据量如果比较大的话 需要一行一行来读
'''
def read_json_large(fname):
    content = []

    with open(fname, 'r', encoding='utf-8') as f:
        for line in f:
            content.append(json.loads(line))

    return content




def write_json_large(content, fname):
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(content, f, ensure_ascii=False, indent=4)



def get_dir(path):

    '''
    :param path:
    :return: 返回路径下的文件夹
    '''

    dirpath = []
    for file in os.listdir(path):
        if os.path.isdir(file):
            if file == '__pycache__':
                continue
            dirpath.append(file)

        elif os.path.isfile(file):
            continue
        else:
            continue
    print("该路径下的目录文件有：")
    print(dirpath)

    return dirpath
