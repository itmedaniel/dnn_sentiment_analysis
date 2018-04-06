#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/9/13 11:42
# @Author  : WenMin
# @Email    : < wenmin593734264@gmial.com >
# @File    : load_cnn_lstm_emotion_model.py
# @Software: PyCharm Community Edition


import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Convolution1D, MaxPooling1D
import jieba

np.random.seed(1337)  # for reproducibility

maxlen = 100           # 序列最大长度
batch_size = 128   # 批数据量大小
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


model = load_model('emotion_CNN_LSTM.h5')
s = '这些人反对小崔，利益使然，这些人的良心都让狗吃了'

s = np.array(doc2num(list(jieba.cut(s)), maxlen))
s = s.reshape((1, s.shape[0]))
score = model.predict(s, batch_size=batch_size)
print ('score:', 2*score[0][0]-1)


