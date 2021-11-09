#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：emhui 
@File    ：dataloader.py.py
@Author  ：wanghuifen
@Date    ：2021/11/2 20:14 
@url     ：https://pypi.tuna.tsinghua.edu.cn/simple
'''
import sys
sys.path.append('../')
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch



def read_dataset(args, path):
    '''
    :param args:
    :param path:
    :return: 数据处理成bert输入的三种id的格式以及对应的label
    '''
    model_path = '../pretrained_models/bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    dataset, columns = [], {}

    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            #csv，tsv文件读取出来的m每一行都是一个列表
            line = line[:-1].split("\t")
            tgt = int(line[columns["label"]])
            if "features" not in columns:  # Sentence classification.
                text_a = line[columns["sentences"]]
                res = tokenizer(text_a, padding="max_length",max_length=512)

            else:  # Sentence-pair classification.
                text_a, text_b = line[columns["sentences"]], line[columns["features"]]
                res = tokenizer(text_a, padding="max_length", max_length=512)

            input_ids = res["input_ids"]
            token_type_ids = res["token_type_ids"]
            attention_mask = res["attention_mask"]

            dataset.append((input_ids, token_type_ids, attention_mask, tgt))
    return dataset

class BertDataset():

    def __init__(self, dataset):

        self.dataset = dataset

    def __getitem__(self, item):
        res = self.dataset[item]
        input_ids = res[0]
        token_type_ids = res[1]
        attention_mask = res[2]
        label = res[3]
        return torch.tensor(input_ids,dtype = torch.long), \
               torch.tensor(token_type_ids, dtype=torch.long), \
               torch.tensor(attention_mask, dtype=torch.long),\
               torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.dataset)


