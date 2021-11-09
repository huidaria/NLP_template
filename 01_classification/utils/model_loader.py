#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：emhui 
@File    ：model_loader.py.py
@Author  ：wanghuifen
@Date    ：2021/11/2 21:26 
@url     ：https://pypi.tuna.tsinghua.edu.cn/simple
'''
import torch

def load_model(model, model_path):
    """
    Load model from saved weights.
    """
    if hasattr(model, "module"):
        model.module.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    else:
        model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    return model