#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：emhui_v2 
@File    ：run_classifier.py
@Author  ：wanghuifen
@Date    ：2021/11/7 10:24 
@url     ：https://pypi.tuna.tsinghua.edu.cn/simple
'''
import sys
sys.path.append('../')
sys.path.append('../../')
from data_loader.dataloader import *
from torch.utils.data import Dataset, DataLoader
from bert_em import *
import argparse
from utils import *
from omegaconf import OmegaConf
import torch.nn as nn
from tensorboardX import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import numpy as np
from tqdm import tqdm
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'
#忽略警告信息
import warnings
warnings.filterwarnings("ignore")


'''
def load_or_initialize_parameters(args, model):
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)
'''

def build_optimizer(args,config, model):
    str2optimizer = {"adamw": AdamW, "adafactor": Adafactor}

    str2scheduler = {"linear": get_linear_schedule_with_warmup, "cosine": get_cosine_schedule_with_warmup,
                     "cosine_with_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
                     "polynomial": get_polynomial_decay_schedule_with_warmup,
                     "constant": get_constant_schedule, "constant_with_warmup": get_constant_schedule_with_warmup}

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if config.optimizer.type in ["adamw"]:
        optimizer = str2optimizer[config.optimizer.type](optimizer_grouped_parameters, lr=config.optimizer.params.lr, correct_bias=False)
    else:
        optimizer = str2optimizer[config.optimizer.type](optimizer_grouped_parameters, lr=config.optimizer.params.lr,
                                                  scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup, args.train_steps)
    return optimizer, scheduler

def train(args, config, model):
    dataset = read_dataset(args, args.train_path)
    random.shuffle(dataset)
    dataset = BertDataset(dataset)

    dataloder = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    optimizer, scheduler = build_optimizer(args,config, model)
    model.train()
    model.zero_grad()
    base_dev_loss = float('inf')
    base_dev_f1 = 0.0
    writer = SummaryWriter('../../train_log')
    global_train_loss = 0.0
    global_step = 0
    print('-----------------------------------开始训练-----------------------------------------')
    for epoch in range(config.train_epochs):
        logger.info("第{}个epoch开始训练".format(epoch))
        global_train_loss = 0.0
        all_target = []
        all_pred = []
        for index,(input_ids, token_type_ids, attention_mask, label) in enumerate(dataloder):
            # 避免影响 在每个batch中开启model.train()
            model.train()
            global_step += 1
            input_ids = input_ids.to(args.device)
            token_type_ids = token_type_ids.to(args.device)
            attention_mask = attention_mask.to(args.device)
            label = label.to(args.device)
            logits, loss = model(input_ids, token_type_ids, attention_mask, label)
            #考虑为多gpu的情况
            if torch.cuda.device_count() > 1:
                loss = torch.mean(loss)
            global_train_loss += loss.item()
            # 返回索引值
            pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
            all_pred.extend(pred.cpu().numpy().tolist())
            all_target.extend(label.cpu().numpy().tolist())
            #计算loss之后进行梯度的更新和学习率的更新
            loss.backward()
            optimizer.step()
            scheduler.step()
            #将模型的梯度参数设置为0
            model.zero_grad()

            #三个batch验证依次
            if (index + 1) % 3 == 0:
                print("--------------------------------------启动验证----------------------------------------")
                print("epoch = {}, batch = {}".format(epoch+1, index+1))
                
                p1, r1, f11 = precision_score(all_target, all_pred, pos_label=1), recall_score(all_target, all_pred,
                                                                                               pos_label=1), \
                              f1_score(all_target, all_pred, pos_label=1)

                logger.info("训练中label = 1的数据中p1 = {}, r1 = {}, f1 = {}".format(p1, r1, f11))

                p0, r0, f10 = precision_score(all_target, all_pred, pos_label=0), recall_score(all_target, all_pred,
                                                                                               pos_label=0), \
                              f1_score(all_target, all_pred, pos_label=0)
                logger.info("训练集中label = 0的数据中p1 = {}, r1 = {}, f1 = {}".format(p0, r0, f10))

                auc = accuracy_score(all_target, all_pred)
                logger.info("训练集中的数据中auc = {}".format(auc))

                area = roc_auc_score(all_target, all_pred)
                logger.info("训练集中的数据中roc_auc_score = {}".format(area))

                
                global_train_loss/=3
                
                logger.info("训练集中的数据中loss = {}".format(global_train_loss))

                dev_loss, dev_f1 = evaluate(args,config,model)
                writer.add_scalar('Train/Loss', global_train_loss, global_step)
                writer.add_scalar('dev/Loss', dev_loss, global_step)
                if dev_f1 > base_dev_f1:
                    logger.info("保存模型  epoch = {}, index = {}".format(epoch, index))
                    
                    save_model(model, config.save_model_path + 'best_model'+str(epoch)+'.bin')
                    base_dev_f1 = dev_f1
                elif dev_f1 == base_dev_f1 and dev_loss < base_dev_loss:
                    logger.info("保存模型  epoch = {}, index = {}".format(epoch, index))
                    save_model(model, config.save_model_path + 'best_model' + str(epoch) + '.bin')

                if dev_loss < base_dev_loss:
                    base_dev_loss = dev_loss
                global_train_loss = 0.0
                all_pred = []
                all_target = []

def evaluate(args, config, model):
    dataset = read_dataset(args, args.dev_path)
    len_dev = len(dataset)
    random.shuffle(dataset)
    dataset = BertDataset(dataset)
    
    dataloder = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    #开启验证模式
    all_target = []
    all_pred = []
    model.eval()
    global_dev_loss = 0.0
    for index, (input_ids, token_type_ids, attention_mask, label) in enumerate(dataloder):
        input_ids = input_ids.to(args.device)
        token_type_ids = token_type_ids.to(args.device)
        attention_mask = attention_mask.to(args.device)
        label = label.to(args.device)

        #不需要进行梯度更新
        with torch.no_grad():
            logits, loss = model(input_ids, token_type_ids, attention_mask, label)
            # 考虑为多gpu的情况
            if torch.cuda.device_count() > 1:
                loss = torch.mean(loss)
            global_dev_loss += loss.item()

        #返回索引值
        pred = torch.argmax(nn.Softmax(dim=1)(logits), dim = 1)
        all_pred.extend(pred.cpu().numpy().tolist())
        all_target.extend(label.cpu().numpy().tolist())

    #依次得到评价的精确度、召回率，f1值，
    p1, r1, f11 = precision_score(all_target, all_pred, pos_label=1), recall_score(all_target, all_pred, pos_label=1),\
                  f1_score(all_target, all_pred, pos_label=1)

    logger.info("验证集中label = 1的数据中p1 = {}, r1 = {}, f1 = {}".format(p1,r1,f11))

    p0, r0, f10 = precision_score(all_target, all_pred, pos_label=0), recall_score(all_target, all_pred, pos_label=0),\
                  f1_score(all_target, all_pred, pos_label=0)
    logger.info("验证集中label = 0的数据中p1 = {}, r1 = {}, f1 = {}".format(p0, r0, f10))

    auc = accuracy_score(all_target, all_pred)
    logger.info("验证集中的数据中auc = {}".format(auc))


    area = roc_auc_score(all_target, all_pred)
    logger.info("验证集中的数据中roc_auc_score = {}".format(area))

    logger.info("验证集中的数据中loss = {}".format(global_dev_loss/len_dev))

    return loss, f11

def test_predict(args, config, model):
    print("--------------------------------------开始测试----------------------------------------")
    dataset = BertDataset(read_dataset(args,args.test_path))
    dataloder = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    
    #开启验证模式
    all_target = []
    all_pred = []
    model.eval()
    for index, (input_ids, token_type_ids, attention_mask, label) in tqdm(enumerate(dataloder)):
        print("index=", index)
        input_ids = input_ids.to(args.device)
        token_type_ids = token_type_ids.to(args.device)
        attention_mask = attention_mask.to(args.device)
        label = label.to(args.device)
        print("aaa")

        #不需要进行梯度更新
        with torch.no_grad():
            print("bbb")
            logits, _ = model(input_ids, token_type_ids, attention_mask, label)
            print("ccc")

        #返回索引值
        print("ddd")
        pred = torch.argmax(nn.Softmax(dim=1)(logits), dim = 1)
        all_pred.extend(pred.cpu().numpy().tolist())
        all_target.extend(label.cpu().numpy().tolist())
        print("index over")

    #依次得到评价的精确度、召回率，f1值，
    p1, r1, f11 = precision_score(all_target, all_pred, pos_label=1), recall_score(all_target, all_pred, pos_label=1),\
                  f1_score(all_target, all_pred, pos_label=1)

    logger.info("测试集中label = 1的数据中p1 = {}, r1 = {}, f1 = {}".format(p1,r1,f11))

    p0, r0, f10 = precision_score(all_target, all_pred, pos_label=0), recall_score(all_target, all_pred, pos_label=0),\
                  f1_score(all_target, all_pred, pos_label=0)
    logger.info("测试集中label = 0的数据中p1 = {}, r1 = {}, f1 = {}".format(p0, r0, f10))

    auc = accuracy_score(all_target, all_pred)
    logger.info("测试集中的数据中auc = {}".format(auc))


    area = roc_auc_score(all_target, all_pred)
    logger.info("测试集中的数据中roc_auc_score = {}".format(area))

    return f11


if __name__ == '__main__':
    set_seed(7)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("使用的设备为device = {}".format(device))

    global logger
    logger = setup_logger("em", '../log', 0, filename='model_train.txt')

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type = str, default ='../../corpora/preprocessed_data/EntityMatcher/amazon_google/train.tsv')
    parser.add_argument('--dev_path', type = str, default = '../../corpora/preprocessed_data/EntityMatcher/amazon_google/valid.tsv')
    parser.add_argument('--test_path', type = str, default = '../../corpora/preprocessed_data/EntityMatcher/amazon_google/test.tsv')
    parser.add_argument('--pretrained_model', type = str, default='../../pretrained_models/bert-base-uncased')
    parser.add_argument('--device', type=str, default=device)
    parser.add_argument('--scheduler', type = str, default='constant_with_warmup')
    parser.add_argument("--warmup", type=float, default=0.1,help="Warm up value.")
    args = parser.parse_args()

    TRAIN_CONFIG_PATH = './train_config.yaml'
    MODEL_CONFIG_PATH = './model_config.yaml'
    train_config = load_config_file(TRAIN_CONFIG_PATH)
    model_config = load_config_file(MODEL_CONFIG_PATH)

    config = OmegaConf.merge(train_config, model_config)

    model = BertEM(args, config)
    # Load or initialize parameters.
    #load_or_initialize_parameters(args, model)

    #将model放在gpu上运行
    model = model.to(args.device)
    #考虑是否是多gpu的情况 将模型分发
    if torch.cuda.device_count() > 1:
        print("GPU数量为{}".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    else:
        print("数量为1")
    trainset = read_dataset(args, args.train_path)
    instances_num = len(trainset)
    batch_size = config.batch_size
    args.train_steps = int(instances_num * config.train_epochs / batch_size) + 1
    train(args, config, model)

    '''
    model1 = BertEM(args, config)

    model1 = load_model(model1,'../save_modelbest_model2.bin')
    model1 = model1.to(args.device)
    #考虑是否是多gpu的情况 将模型分发
    if torch.cuda.device_count() > 1:
        print("GPU数量为{}".format(torch.cuda.device_count()))
        model1 = torch.nn.DataParallel(model1)
    else:
        print("数量为1")
    test_predict(args, config, model1)
    '''






