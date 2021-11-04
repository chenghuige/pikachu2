import json
from gensim.models import word2vec
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import jieba
import pandas as pd
maxwords = 2000000
maxseq = 1000


def lis(x):
    out = []
    for i in x:
        out.append(str(i))
    return out

train_ad = pd.read_csv("./train/ad.csv")
train_click_log = pd.read_csv("./train/click_log.csv")
train_user = pd.read_csv("./train/user.csv")
test_ad = pd.read_csv("./test/ad.csv")
test_click_log = pd.read_csv("./test/click_log.csv")

train_data = pd.merge(train_click_log,train_ad,on = "creative_id",how = "left")
train_data = pd.merge(train_data,train_user,on = "user_id",how = "left")
train_data.shape

test_data = pd.merge(test_click_log,test_ad,on = "creative_id",how = "left")


train_id = train_data[["user_id","time","creative_id"]].sort_values(by = ["user_id","time"],ascending = True)
train_embedding_id = train_id.groupby(["user_id"])["creative_id"].apply(lis).reset_index().rename(columns = {"creative_id":"embedding_id"})
#print(train_embedding_id.head())
test_id = test_data[["user_id","time","creative_id"]].sort_values(by = ["user_id","time"],ascending = True)
test_embedding_id = test_id.groupby(["user_id"])["creative_id"].apply(lis).reset_index().rename(columns = {"creative_id":"embedding_id"})
#print(test_embedding_id.head())
"""
train_embedding_id["age"] = train_user["age"]
train_embedding_id["gender"] = train_user["gender"]
train_embedding_id["y"] = [str(train_user["age"][i])+"_"+str(train_user["gender"][i]) for i in range(900000)]
train_embedding_id["y"] = train_embedding_id["y"].map({"1_1":1,"1_2":2,"2_1":3,"2_2":4,"3_1":5,"3_2":6,"4_1":7,"4_2":8,"5_1":9,"5_2":10,"6_1":11,"6_2":12,"7_1":13,"7_2":14,"8_1":15,"8_2":16,"9_1":17,"9_2":18,"10_1":19,"10_2":20})
print(train_embedding_id.head())
print(train_embedding_id["y"].unique())
train_embedding_id.to_csv("./train_embedding_id.csv",index = False)
test_embedding_id.to_csv("./test_embedding_id.csv",index = False)

"""
#all_data = pd.read_csv("./all_data.csv",encoding = "utf-8")
texts = []
print("*************************start process data**********************")

for id_ in train_embedding_id.embedding_id:
    texts.append(id_)
for id_ in test_embedding_id.embedding_id:
    texts.append(id_)
print(len(texts))
print(texts[0])
print(type(texts[0]))
#统计词频
word_index = {}
word_count = {}
for text in texts:
    for i in text:
        if i not in word_count:
            word_count[i] = 1
        else:
            word_count[i]+=1
sort_word = sorted(word_count.items(),key = lambda x:x[1],reverse = True)
print(sort_word[:10])
print(len(sort_word))
for i in range(len(sort_word)):
    if sort_word[i][0] not in word_index and sort_word[i][1] >1:
        word_index[sort_word[i][0]] = i+1
print(len(word_index))        
"""      
tokenizer = Tokenizer(num_words = maxwords)
tokenizer.fit_on_texts(texts)
#data_seq = tokenizer.texts_to_sequences(texts)
#data_pad = pad_sequences(data_seq, maxlen=maxseq)
"""
print("*******************start train************************")
count = 0
#word_index = tokenizer.word_index

#将词索引写入json
with open('creative_word_index_521_d300.json','w') as f:
    json.dump(word_index, f, ensure_ascii=False)

model_dm = word2vec.Word2Vec(texts,
                             size = 300,
                             sg= 1,
                             min_count = 1,
                             window = 128,
                             workers = 24,
                             seed = 2020)

word_len = len(word_index)
embedding_word2vec = np.zeros((word_len + 1, 300))
for word, i in word_index.items():
    embedding_vector = model_dm[word] if word in model_dm else None
    if embedding_vector is not None:
        count += 1
        embedding_word2vec[i] = embedding_vector
    else:
        unk_vec = np.random.random(300) * 0.5
        unk_vec = unk_vec - unk_vec.mean()
        embedding_word2vec[i] = unk_vec
#将词矩阵索引写入json

wv = {}
wv['wv'] = embedding_word2vec.tolist()
with open('wv_creative_521_d300.json','w',encoding = 'utf-8') as f:
    json.dump(wv,f,ensure_ascii=False)
f.close()

