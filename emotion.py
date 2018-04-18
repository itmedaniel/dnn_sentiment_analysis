#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/10 10:31
# @Author  : WenMin
# @Email    : < wenmin593734264@gmial.com >
# @File    : Sentiment analysis.py
# @Software: PyCharm Community Edition

import numpy as np
import pandas as pd
import random
import jieba
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding
from keras.layers import LSTM
random.seed(1024)

# pos = pd.read_excel('pos.xls', header=None)
# pos['label'] = 1
# neg = pd.read_excel('neg.xls', header=None)
# neg['label'] = 0
# all_ = pos.append(neg, ignore_index=True)
# all_['words'] = all_[0].apply(lambda s: list(jieba.cut(s))) #调用结巴分词


def judge(s):
    if float(s) > 0:
        return 1
    if float(s) < 0:
        return 0
all_ = pd.read_csv('result_score_new_1.csv', header=None)
all_['words'] = all_[0]
all_['label'] = all_[1].apply(lambda s: judge(s))

maxlen = 50
min_count = 3

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

idx = range(len(all_))
np.random.shuffle(idx)
all_ = all_.loc[idx]

x = np.array(list(all_['doc2num']))
all_len = len(x)
y = np.array(list(all_['label']))
y = y.reshape((-1, 1))

train_size = int(all_len*0.8)
test_size = all_len - train_size

model = Sequential()
model.add(Embedding(len(abc), 256, input_length=maxlen))
model.add(LSTM(128, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 512

model.fit(x[:train_size], y[:train_size], batch_size=batch_size, nb_epoch=5, validation_split=0.2)
# s = []
score = model.evaluate(x[train_size:], y[train_size:], batch_size=batch_size)
print (score)

model.save_weights('emotion_lstm.h5')