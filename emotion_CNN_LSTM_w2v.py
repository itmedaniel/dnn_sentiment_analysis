#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/14 17:41
# @Author  : WenMin
# @Email    : < wenmin593734264@gmial.com >
# @File    : emotion_CNN_LSTM_w2v.py
# @Software: PyCharm Community Edition
from gensim.models import word2vec
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Convolution1D, MaxPooling1D
np.random.seed(1337)  # for reproducibility


# Embedding  词嵌入
maxlen = 100           # 序列最大长度
embedding_size = 128   # 词向量维度

# Convolution  卷积
filter_length = 5    # 滤波器长度
nb_filter = 64       # 滤波器个数
pool_length = 4      # 池化长度

# LSTM
lstm_output_size = 64   # LSTM 层输出尺寸

# Training   训练参数
batch_size = 128   # 批数据量大小
nb_epoch = 30     # 迭代次数


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


EMBEDDING_FILE = 'new_med100.model.bin'
word_model = word2vec.Word2Vec.load(EMBEDDING_FILE)
weights = word_model.wv.syn0
vocab = dict([(k, v.index) for k, v in word_model.wv.vocab.items()])


def to_ids(words):
    def word_to_id(word):
        id = vocab.get(word.decode('utf-8'))
        if id is None:
            id = 0
        return id
    words = words.strip().split()
    x = list(map(word_to_id, words))
    return x


def doc2num(s):
    l =[]
    for i in s.strip().split():
        if i.decode('utf-8') in word_model:
            l.append(word_model[i.decode('utf-8')])
        else:
            continue
    l = np.array(l)
    result = np.mean(l, axis=0, keepdims=True)
    return result

all_['doc2num'] = all_['words'].apply(lambda s: to_ids(s))

#手动打乱数据
idx = range(len(all_))
np.random.shuffle(idx)
all_ = all_.loc[idx]

#按keras的输入要求来生成数据
x = sequence.pad_sequences(all_['doc2num'], maxlen=maxlen)
y = np.array(list(all_['label']))
y = y.reshape((-1, 1)) #调整标签形状


embedding_layer = Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1], weights=[weights],
                            trainable=False, input_length=maxlen)


# 构建模型
model = Sequential()
model.add(embedding_layer)
# model.add(Embedding(len(abc), embedding_size, input_length=maxlen))  # 词嵌入层
model.add(Dropout(0.25))       # Dropout层

# 1D 卷积层，对词嵌入层输出做卷积操作
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
# 池化层
model.add(MaxPooling1D(pool_length=pool_length))
# LSTM 循环层
model.add(LSTM(lstm_output_size))
# 全连接层，只有一个神经元，输入是否为正面情感值
model.add(Dense(1))
model.add(Activation('sigmoid'))  # sigmoid判断情感

model.summary()   # 模型概述

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# 训练
print('Train...')
model.fit(x[:train_num], y[:train_num],
          batch_size=batch_size,
          epochs=nb_epoch,
          validation_data=(x[:train_num], y[:train_num]))

# 测试
score, acc = model.evaluate(x[train_num:], y[train_num:], batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
model.save('emotion_CNN_LSTM.h5')