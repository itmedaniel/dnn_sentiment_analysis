#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/13 23:53
# @Author  : WenMin
# @Email    : < wenmin593734264@gmial.com >
# @File    : emotion_CNN.py
# @Software: PyCharm Community Edition

import numpy as np
import pandas as pd
import jieba
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.datasets import imdb
np.random.seed(1337)  # for reproducibility

# set parameters:  设定参数
# max_features = 5000  # 最大特征数（词汇表大小）
maxlen = 50         # 序列最大长度
batch_size = 128      # 每批数据量大小
embedding_dims = 256  # 词嵌入维度
nb_filter = 64      # 1维卷积核个数
filter_length = 3    # 卷积核长度
hidden_dims = 250    # 隐藏层维度
nb_epoch = 30        # 迭代次数
train_num = 150000
min_count = 3  # 序列的最小长度

def judge(s):
    if float(s) > 0:
        return 1
    if float(s) < 0:
        return 0
all_ = pd.read_csv('result_score_new.csv', header=None)
all_['words'] = all_[0]
all_['label'] = all_[1].apply(lambda s: judge(s))
content = []
for i in all_['words']:
    content.extend(i)

abc = pd.Series(content).value_counts()
abc = abc[abc >= min_count]
abc[:] = range(1, len(abc)+1)
abc[''] = 0
word_set = set(abc.index)


def doc2num(s, maxlen):
    s = [i for i in s if i in word_set]
    s = s[:maxlen] + ['']*max(0, maxlen-len(s))
    return list(abc[s])

all_['doc2num'] = all_['words'].apply(lambda s: doc2num(s, maxlen))

#手动打乱数据
idx = range(len(all_))
np.random.shuffle(idx)
all_ = all_.loc[idx]

#按keras的输入要求来生成数据
x = np.array(list(all_['doc2num']))
y = np.array(list(all_['label']))
y = y.reshape((-1, 1))


model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
# 先从一个高效的嵌入层开始，它将词汇的索引值映射为 embedding_dims 维度的词向量
model.add(Embedding(len(abc),
                    embedding_dims,
                    input_length=maxlen,
                    dropout=0.5))

# we add a Convolution1D, which will learn nb_filter
# word group filters of size filter_length:
# 添加一个 1D 卷积层，它将学习 nb_filter 个 filter_length 大小的词组卷积核
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
# we use max pooling:
# 使用最大池化
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
# 添加一个原始隐藏层
model.add(Dense(hidden_dims))
model.add(Activation('relu'))
model.add(Dropout(0.5))


# We project onto a single unit output layer, and squash it with a sigmoid:
# 投影到一个单神经元的输出层，并且使用 sigmoid 压缩它
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()  # 模型概述

# 定义损失函数，优化器，评估矩阵
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 训练，迭代 nb_epoch 次
model.fit(x[:train_num], y[:train_num],
          batch_size=batch_size,
          epochs=nb_epoch,
          validation_data=(x[:train_num], y[:train_num]))
score = model.evaluate(x[train_num:], y[train_num:], batch_size=batch_size)[1]
print (score)

model.save('emotion_CNN.h5')