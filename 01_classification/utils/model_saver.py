#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：emhui 
@File    ：model_saver.py.py
@Author  ：wanghuifen
@Date    ：2021/11/2 21:27 
@url     ：https://pypi.tuna.tsinghua.edu.cn/simple
'''
import torch


def save_model(model, model_path):
    """
    Save model weights to file.
    """
    if hasattr(model, "module"):
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)
