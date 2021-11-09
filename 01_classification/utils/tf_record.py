#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：emhui 
@File    ：tf_record
@Author  ：wanghuifen
@Date    ：2021/11/3 21:58 
@url     ：https://pypi.tuna.tsinghua.edu.cn/simple
'''

import os
import argparse
import shutil
from datetime import datetime, timedelta
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
import tensorflow as tf
from transformers import TFBertModel
from mlx_studio.context import JOB_CONTEXT
from mlx_studio.modelhub import register_model
from mlx_studio.modelhub.modelhub import ModelType

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
project_dir = os.path.dirname(os.path.abspath('__file__'))
# bert_mode_path='/data00/home/pengsong/Data/bert_model/bert_base_chinese'
bert_mode_path = os.path.join(project_dir, 'bert_base_chinese')
model_path = JOB_CONTEXT.model_path
train_path_dir = JOB_CONTEXT.train_path
max_length = 16
regularizers_lambda = 0.01


class QueryQualityModel(tf.keras.Model):
    def __init__(self, args):
        super(QueryQualityModel, self).__init__()
        # self.args = args
        self.bert_encoder = TFBertModel.from_pretrained(bert_mode_path)
        self.dropout = tf.keras.layers.Dropout(args.dropout_rate)
        self.query_dense = tf.keras.layers.Dense(
            args.hidden_size,
            kernel_initializer='glorot_normal',
            bias_initializer=tf.keras.initializers.constant(0.1),
            kernel_regularizer=tf.keras.regularizers.l2(regularizers_lambda),
            bias_regularizer=tf.keras.regularizers.l2(regularizers_lambda),
            name='query_emb'
        )
        self.intent_output = tf.keras.layers.Dense(
            1,
            activation='sigmoid',
            kernel_initializer='glorot_normal',
            bias_initializer=tf.keras.initializers.constant(0.1),
            kernel_regularizer=tf.keras.regularizers.l2(regularizers_lambda),
            bias_regularizer=tf.keras.regularizers.l2(regularizers_lambda),
            name='quality_score'
        )

    def call(self, inputs, training=True):
        hidden_output = self.bert_encoder(inputs['input_ids'], token_type_ids=inputs['token_type_ids'],
                                          attention_mask=inputs['attention_mask'])[-1]
        if training:
            hidden_output = self.dropout(hidden_output)
        dense_output = self.query_dense(hidden_output)
        output = self.intent_output(dense_output)
        return output


def init_args(continuous_training=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='query_quality_model', type=str)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--learning_rate', default=2e-5, type=float)
    parser.add_argument('--dropout_rate', default=0.1, type=float)
    parser.add_argument('--max_length', default=max_length, type=int)
    parser.add_argument('--label_size', default=2, type=int)
    parser.add_argument('--hidden_size', default=128, type=int)
    parser.add_argument('--train_paths', default=train_path_dir + '/train.tfrecord', type=str)
    parser.add_argument('--eval_paths', default=train_path_dir + '/val.tfrecord', type=str)
    parser.add_argument('--test_paths', default=None, type=str)
    parser.add_argument('--predict_paths', default=None, type=str)
    parser.add_argument('--model_path', default=model_path, type=str)
    parser.add_argument('--buffer_size', default=40000, type=int)
    parser.add_argument('--tensorboard_dir', default=None, type=str)
    parser.add_argument('--checkpoint_dir', default=None, type=str)
    args = parser.parse_args()

    return args


def parse_tfrecord(x, args):
    features = {
        'input_ids': tf.io.FixedLenFeature([args.max_length], tf.int64),
        'token_type_ids': tf.io.FixedLenFeature([args.max_length], tf.int64),
        'attention_mask': tf.io.FixedLenFeature([args.max_length], tf.int64),
        'label': tf.io.FixedLenFeature([1], tf.int64)
    }
    parsed = tf.io.parse_single_example(x, features)
    input_ids = parsed['input_ids']
    token_type_ids = parsed['token_type_ids']
    attention_mask = parsed['attention_mask']
    # label=parsed['label']
    label = tf.cast(parsed['label'], tf.float32)
    return {
               'input_ids': input_ids,
               'token_type_ids': token_type_ids,
               'attention_mask': attention_mask
           }, label


def calcul_metrics(y_true, y_pred):
    y_pred = [1 if y >= 0.5 else 0 for y in y_pred]
    acc = accuracy_score(y_true, y_pred)
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return acc, p, r, f1


def train(args, is_continuous=False):
    train_dataset = tf.data.TFRecordDataset(args.train_paths).map(lambda x: parse_tfrecord(x, args),
                                                                  num_parallel_calls=16).shuffle(
        args.buffer_size).batch(args.batch_size)
    val_dataset = tf.data.TFRecordDataset(args.eval_paths).map(lambda x: parse_tfrecord(x, args),
                                                               num_parallel_calls=16).batch(args.batch_size)
    model = QueryQualityModel(args)
    if is_continuous:
        latest = tf.train.latest_checkpoint(args.checkpoint_dir)
        model.load_weights(latest)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    loss_func = tf.keras.losses.BinaryCrossentropy()
    start_time = datetime.today()
    best_auc = 0
    best_epock = 0
    for k in range(args.epochs):
        i = 0
        for features, y_true in train_dataset:
            i += 1
            with tf.GradientTape() as t:
                # y_pred = model(features['input_ids'],features['token_type_ids'],features['attention_mask'])
                y_pred = model(features)
                loss = loss_func(y_true, y_pred)
                loss = tf.math.reduce_mean(loss)
            grads = t.gradient(loss, model.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 0.5)  # 梯度截断，避免梯度爆炸，一般设置为1，还没有测试设置0.1和1的差别
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if i % 100 == 0:
                end_time = datetime.today()
                time_diff = ':'.join(str(end_time - start_time).split(':')[:2])
                y_true = tf.squeeze(y_true).numpy()
                y_pred = tf.squeeze(y_pred).numpy()
                auc = roc_auc_score(y_true, y_pred)
                acc, p, r, f = calcul_metrics(y_true, y_pred)
                print(
                    f'epoch/batch:{k}/{i},acc={round(acc, 6)},auc={round(auc, 6)},precision_score={round(p, 6)},recall_score={round(r, 6)},f1_score={round(f, 6)}, time spent:{time_diff}')

                # val
        i = 0
        y_pred_list = []
        y_true_list = []
        total_val_loss = []
        for features, y_true in val_dataset:
            # y_pred = model(features['input_ids'],features['token_type_ids'],features['attention_mask'],training=False)
            y_pred = model(features, training=False)
            y_true = tf.squeeze(y_true).numpy()
            y_pred = tf.squeeze(y_pred).numpy()
            y_pred_list.extend(y_pred)
            y_true_list.extend(y_true)
            i += 1
        auc = roc_auc_score(y_true_list, y_pred_list)
        acc, p, r, f = calcul_metrics(y_true_list, y_pred_list)
        cur_time = datetime.today().strftime('%Y-%m-%d %H:%M')
        print(
            f'val metrics,acc={round(acc, 6)},auc={round(auc, 6)},precision_score={round(p, 6)},recall_score={round(r, 6)},f1_score={round(f, 6)}, cur_time:{cur_time} {"*" if auc > best_auc else ""}')
        if best_auc < auc:
            # print(f'best_auc:{best_auc}, cur_auc:{auc}')
            best_auc = auc
            best_epock = k
            model.save(args.model_path, save_format='tf')
            register_model(os.path.join(args.model_path, 'saved_model.pb'), model_type=ModelType.Tensorflow,
                           model_name=args.model_name)
        if k - best_epock > 3:
            break


if __name__ == "__main__":
    args = init_args(True)
    train(args)
    # predict_result(args)
    # test(args)
    # predict_result(args)
    # eval_result(args)
