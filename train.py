from keras.layers import *
import json
import re
import random
from bert4keras.backend import keras, set_gelu, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Dropout, Dense
import pandas as pd
import numpy as np
import re
import os
import utils


set_gelu('tanh')  # 切换gelu版本
maxlen = 128
batch_size = 32

p = 'D:/lyh/model/chinese_roberta_wwm_ext_L-12_H-768_A-12/'
config_path = p +'bert_config.json'
checkpoint_path = p + 'bert_model.ckpt'
dict_path = p +'vocab.txt'

tokenizer = Tokenizer(dict_path, do_lower_case=True)

def read_data():
    res = []
    data_path = 'D:/lyh/span_ner/new/train.json'
    train_data = utils.read_json(data_path)
    for i in train_data:
        for d in train_data[i]:
            i = i[:maxlen]
            if d[0] in i:
                res.append([d[1],d[0],i])
    return res

def clean_data(text):
    temp = ['\uf043','\uf020','\uf076','\uf046','\uf075','\uf06c','\uf09f','\uf0d8','\uf072','．','…','„']
    List = re.findall(r'[(]cid.*?[)]',text)
    List = List + temp
    for l in List:
        text = text.replace(l,'')
    return text    

class data_generator(DataGenerator):
    """
        数据生成器
    """

    def __iter__(self, random=False):
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for i in idxs:
            data = self.data[i]
            text2 = data[2]
            text1 = data[1]
            label = data[0]
            token_ids, segment_ids = tokenizer.encode(text1, text2,maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        # print(model.predict(x_true))
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


def predict(data):
    for x_true, y_true in data:
        # print(model.predict(x_true))
        y_pred = model.predict(x_true).argmax(axis=1)
        print(y_pred)



class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('D:/lyh/span_ner/new/best_model_add_new.weights')
        test_acc = evaluate(test_generator)
        print(u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n'
              % (val_acc, self.best_val_acc, test_acc))



# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,

) 

output = Lambda(lambda x: x[:, 0],
                    name='CLS-token')(bert.model.output)
output = Dense(units=100,
                activation='relu',
                kernel_initializer=bert.initializer)(output)

output = Dense(units=100,
                activation='relu',
                kernel_initializer=bert.initializer)(output)

output = Dense(units=27,
                activation='softmax')(output)                

model = keras.models.Model(bert.model.input, output)
model.summary()

model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(2e-5),  # 用足够小的学习率
        # optimizer=PiecewiseLinearLearningRate(Adam(5e-5), {10000: 1, 30000: 0.1}),
        metrics=['accuracy'],
    )

all_data = read_data()
random_order = range(len(all_data))
np.random.shuffle(list(random_order))
train_data = [all_data[j] for i, j in enumerate(random_order) if i % 8 != 2 and i % 8 != 1]
valid_data = [all_data[j] for i, j in enumerate(random_order) if i % 8 == 1]
test_data = [all_data[j] for i, j in enumerate(random_order) if i % 8 == 2]
    
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)

evaluator = Evaluator()
model.fit_generator(train_generator.forfit(),
                    steps_per_epoch=len(train_generator),
                    epochs=10,
                    callbacks=[evaluator])

