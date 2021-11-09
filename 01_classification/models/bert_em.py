#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：emhui 
@File    ：bert_em.py.py
@Author  ：wanghuifen
@Date    ：2021/11/2 20:56 
@url     ：https://pypi.tuna.tsinghua.edu.cn/simple
'''

from transformers import BertTokenizer, BertConfig, BertModel
import torch.nn as nn
import torch
import numpy as np 

class BertEM(nn.Module):
    def __init__(self, args, config):
        super(BertEM, self).__init__()
        # Binary classification problem (num_labels = 2)
        self.num_labels = config.num_labels
        # Pre-trained BERT model
        self.bert = BertModel.from_pretrained(args.pretrained_model)
        # Dropout to avoid overfitting
        #self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # A single layer classifier added on top of BERT to fine tune for binary classification
        #self.output_layer_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_layer_2 = nn.Linear(config.hidden_size, config.num_labels)
        # Weight initialization
        torch.nn.init.xavier_normal_(self.output_layer_2.weight)

    def forward(self, input_ids, token_type_ids, attention_mask,label):
        # Forward pass through pre-trained BERT
        outputs = self.bert(input_ids = input_ids,  token_type_ids=token_type_ids,attention_mask=attention_mask)
        target = label
        '''
        bert如果没有设置hidden_size = true的话，最后的output输出的结果为两种，
        output[0]也就是last_hidden_state shape = batch_size*length*vocab_size
        output[1]也是output[-1]是cls的输出向量，也叫作pooler_output, shape = batch_size*vocab_size
        
        也就是说
        # 返回每个位置编码、CLS位置编码、所有中间隐层、所有中间attention分布等。
        # sequence_output, pooled_output, (hidden_states), (attentions)。
        # (hidden_states), (attentions)需要config中设置相关配置，否则默认不保存
        
        '''
        # Last layer output (Total 12 layers)
        pooled_output = outputs[-1]

        #pooled_output = self.dropout(pooled_output)
        #pooled_output = torch.tanh(self.output_layer_1(pooled_output))
        logits = self.output_layer_2(pooled_output)
        # criterion = nn.CrossEntropyLoss(weight=torch.tensor([1,1]).cuda(),size_average=True)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, target)
        return logits, loss