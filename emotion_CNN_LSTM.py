#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/14 0:36
# @Author  : WenMin
# @Email    : < wenmin593734264@gmial.com >
# @File    : emotion_CNN_LSTM.py
# @Software: PyCharm Community Edition

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
maxlen = 20           # 序列最大长度
embedding_size = 64   # 词向量维度

# Convolution  卷积
filter_length = 5    # 滤波器长度
nb_filter = 64       # 滤波器个数
pool_length = 4      # 池化长度

lstm_output_size = 64   # LSTM 层输出尺寸

min_count = 3  # 序列的最小长度


def judge(s):
    if float(s) >= 0:
        return 1
    if float(s) < 0:
        return -1
all_ = pd.read_csv('result_score_new_1.csv', header=None)
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
x = np.array(list(all_['doc2num']))
y = np.array(list(all_['label']))
y = y.reshape((-1, 1))

all_len = len(x)
train_size = int(all_len*0.7)
test_size = all_len - train_size
valida_size = int(test_size * 0.5)
batch_size = 64
nb_epoch = 10

model = Sequential()
model.add(Embedding(len(abc), embedding_size, input_length=maxlen))
model.add(Dropout(0.1))
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=pool_length))
model.add(LSTM(lstm_output_size))
model.add(Dense(1, activation='tanh'))
model.summary()
model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x[:test_size], y[:test_size],
          batch_size=batch_size,
          epochs=nb_epoch,
          validation_data=(x[test_size:test_size+valida_size], y[test_size:test_size+valida_size]))

score, acc = model.evaluate(x[test_size+valida_size:], y[test_size+valida_size:], batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
model.save('emotion_CNN_LSTM.h5')