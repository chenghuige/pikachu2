# coding: utf-8

import pandas as pd
import numpy as np
import random
import jieba 
import jieba.analyse
from gensim.models import word2vec 
import json
import warnings
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras import layers
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint

from keras.models import Model,load_model

#from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score
from keras import backend as K

from keras.utils import multi_gpu_model
import gc
import re
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=UserWarning,module = 'gensim' )
from keras.callbacks import Callback
maxlen = 100
#评价指标函数，可以不用管，我用的20分类
def acc(true_age,true_gender,true_y,predict):
    dic_y = {"0":(1,1),"1":(1,2),"2":(2,1),"3":(2,2),"4":(3,1),"5":(3,2),"6":(4,1),"7":(4,2),"8":(5,1),"9":(5,2),"10":(6,1),"11":(6,2),"12":(7,1),"13":(7,2),"14":(8,1),"15":(8,2),"16":(9,1),"17":(9,2),"18":(10,1),"19":(10,2)}
    count = 0
    count_age = 0
    count_gender = 0
    for i in range(len(predict)):
        if predict[i]==true_y.values[i]:
            count+=1
        if dic_y[str(predict[i])][0]==true_age.values[i]:
            count_age+=1
        if dic_y[str(predict[i])][1]==true_gender.values[i]:
            count_gender+=1
    return count_age/true_y.shape[0],count_gender/true_y.shape[0],(count_age+count_gender)/true_y.shape[0],count/true_y.shape[0]
#模型里面的回调函数，监控指标的。
class Metrics(Callback):

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.best_f1 = 0
        self.best_acc = 0

    def on_epoch_end(self, epoch, logs={}):
        val_predict = np.argmax(np.asarray(self.model.predict(self.validation_data[0])), axis=1)
        val_targ = np.argmax(self.validation_data[1], axis=1)
        a,b,c,d = acc(valid["age"],valid["gender"],valid["y"],val_predict)
        print('****************验证集年龄准确率为{},性别准确率为{},总的准确率为{},都对准确率{}*************"'.format(a,b,c,d))
        if d > self.best_acc:
            self.best_acc = d
            #self.model.save_weights('best_model.weights')
            # self.model.save('best_model_521.h5')
        return
#train_embedding_id.csv是按照用户时间把id排序好的数据
print('start read data...')
train = pd.read_csv("./train_embedding_id.csv")
#这块是我自己组合的20分类不用管
map_y = {"1_1":0,"1_2":1,"2_1":2,"2_2":3,"3_1":4,"3_2":5,"4_1":6,"4_2":7,"5_1":8,"5_2":9,"6_1":10,"6_2":11,"7_1":12,"7_2":13,"8_1":14,"8_2":15,"9_1":16,"9_2":17,"10_1":18,"10_2":19}
y = []
for i in range(train.shape[0]):
    y.append(map_y[str(train.age[i])+"_"+str(train.gender[i])])
train["y"] = y
#打乱数据
data = train.sample(frac = 1.0,random_state=2020)
#train = data.iloc[:850000,:]
train = data.iloc[:800000,:]
valid = data.iloc[800000:,:]
#这个是w2v出来的矩阵向量，还有词对应的索引，你的意思应该就是在w2v里面就把这个拼接好了，增加向量维度
embedding_word2vec = np.array(json.load(open('wv_creative_521_d300.json'))['wv'])
word_index = json.load(open('creative_word_index_521_d300.json'))

#words = [train,valid]

#分词填充函数，利用词索引字典word_index将id转化为数字并填充至统一长度。
def pad(text):
    out = []
    for i in text:
        if i in word_index:
            out.append(word_index[i])
        else:
            out.append(0)
    if len(out)>maxlen:
        out = out[:maxlen]
    else:
        out = out+[0]*(maxlen-len(out))
    return out

def generate_data(data):
    result = []
    for i in range(data.shape[0]):
        result.append(pad(eval(data.creative_idembedding.values[i])))
    return np.array(result)

train_data = generate_data(train)
valid_data = generate_data(valid)

#模型定义函数
def bi_gru_model(maxseq = None, embedding_dim = None,embedding_word2vec = None,gru_unit = None,num_classes = None ):
    content = Input(shape=(maxseq,), dtype='int32')
    #下面这个就是嵌入层，把我们训练好的矩阵直接放进去（weights），可以不用训练了
    embedding = Embedding(embedding_word2vec.shape[0], 
                          embedding_dim, 
                          #weights=[embedding_word2vec],
                          #trainable=False,
                          trainable=True,
                          input_length=maxseq)

    x = SpatialDropout1D(0.2)(embedding(content))
    x0 = GRU(gru_unit, return_sequences=True)(x)
    
    max_pool = GlobalMaxPooling1D()(x0)

 
    output = Dense(num_classes, activation="softmax")(max_pool)

    model = Model(inputs=content, outputs=output)
    return model

def train_gru(train_x = train_data, val_x = valid_data):

    F1_score = 0
 
    gc.collect()
    K.clear_session()
    #model = bi_gru_model(maxseq = maxlen,embedding_word2vec = embedding_word2vec,embedding_dim = 300,gru_unit = 200,num_classes = 20)
    model = bi_gru_model(maxseq = maxlen,embedding_word2vec = embedding_word2vec,embedding_dim = 128,gru_unit = 200,num_classes = 20)
    model = multi_gpu_model(model,2)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
    train_y = train['y']
    val_y = valid['y']

    y_train_onehot = to_categorical(train_y)
    y_val_onehot = to_categorical(val_y)

    history = model.fit([train_x], 
                  [y_train_onehot],
                  epochs=20,
                  batch_size=256, 
                  validation_data=(val_x, y_val_onehot),
                  callbacks = [ Metrics(),
                      EarlyStopping(monitor='val_acc', patience=2, verbose=0, mode='max'),
                           ReduceLROnPlateau(monitor="val_acc", verbose=0, mode='max', factor=0.5, patience=1)])
                  #ModelCheckpoint('bi_gru'+ '.hdf5', monitor='val_acc', verbose=0,save_best_only=True,mode='max',save_weights_only=True)])
    
    #model = load_model('bi_gru.h5',custom_objects={'AttentionWeightedAverage': AttentionWeightedAverage})
    # model = load_model('best_model_521.h5')
    #model.load_weights('best_model.weights')
    print(model.summary())
    return 

print('Start train...')
#训练即可
out = train_gru(train_x=train_data, val_x = valid_data)


print('***************************finished****************************')
