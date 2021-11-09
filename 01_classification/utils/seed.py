#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：emhui 
@File    ：seed.py.py
@Author  ：wanghuifen
@Date    ：2021/11/2 21:35 
@url     ：https://pypi.tuna.tsinghua.edu.cn/simple
'''
import random
import os
import numpy as np
import torch

def set_seed(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

