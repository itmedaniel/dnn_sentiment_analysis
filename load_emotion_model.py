#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/17 21:41
# @Author  : WenMin
# @Email    : < wenmin593734264@gmial.com >
# @File    : load_model.py
# @Software: PyCharm Community Edition

import numpy as np
from keras.models import Sequential,load_model
from keras.layers import Dense
import jieba
import numpy as np
import pandas as pd
import jieba

def judge(s):
    if float(s) > 0:
        return 1
    if float(s) < 0:
        return 0
all_ = pd.read_csv('result_score.csv', header=None)
all_['words'] = all_[0]
all_['label'] = all_[1].apply(lambda s: judge(s))

maxlen = 100 #截断词数
min_count = 3 #出现次数少于该值的词扔掉。这是最简单的降维方法

content = []
for i in all_['words']:
    content.extend(i)

abc = pd.Series(content).value_counts()
abc = abc[abc >= min_count]
abc[:] = range(1, len(abc)+1)
abc[''] = 0 #添加空字符串用来补全
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
y = y.reshape((-1,1)) #调整标签形状
batch_size = 128
train_num = 25000

model = load_model('emotion.h5')
# score = model.evaluate(x[train_num:], y[train_num:], batch_size = batch_size)
# score = model.predict(x[train_num:], batch_size=batch_size)
# for i in score:
#     print (i)
s = '这些人反对小崔，利益使然，这些人的良心都让狗吃了'

s = np.array(doc2num(list(jieba.cut(s)), maxlen))
s = s.reshape((1, s.shape[0]))
score = model.predict(s, batch_size=batch_size)
print ('score:', 2*score[0][0]-1)

