#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：emhui 
@File    ：some_metrics.py
@Author  ：wanghuifen
@Date    ：2021/11/3 21:27 
@url     ：https://pypi.tuna.tsinghua.edu.cn/simple
'''
import os
from sklearn.metrics import precision_score, recall_score,f1_score,roc_auc_score, accuracy_score
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

project_dir = os.path.dirname(os.path.abspath("__file__"))
bert_model_path = os.path.join(project_dir, "bert_base_chinese")

def calcul_metrics(y_true, y_pred):
    y_pred = [1 if y >= 0.5 else 0 for y in y_pred]
    acc = accuracy_score(y_true, y_pred)
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return acc, p, r, f1