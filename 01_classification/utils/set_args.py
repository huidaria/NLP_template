#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：emhui 
@File    ：set_args.py
@Author  ：wanghuifen
@Date    ：2021/11/3 21:36 
@url     ：https://pypi.tuna.tsinghua.edu.cn/simple
'''

import os
import argparse
max_length = 512
def init_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='em_model', type = str)
    parser.add_argument('--batch_size', default=512, type = int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--learning_rate', default=2e-5, type=float)
    parser.add_argument('--dropout_rate', default=0.1, type=float)
    parser.add_argument('--max_length', default=max_length, type=int)
    parser.add_argument('--hidden_size', default=128, type=int)
    parser.add_argument('--train_paths', default=train_path_dir+'./train.tfcord', type=str)
    parser.add_argument('--eval_paths', default=train_path_dir + './val.tfcord', type=str)
    parser.add_argument('--test_paths', default=None, type=int)
    parser.add_argument('--predict_paths', default=None, type=int)
    parser.add_argument('--model_path', default=model_path, type=str)
    parser.add_argument('--buffer_size', default=40000, type=int)
    parser.add_argument('--tensordboard_dir', default=None, type=str)
    parser.add_argument('--checkpoint_dir', default=None, type=str)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    init_args()

