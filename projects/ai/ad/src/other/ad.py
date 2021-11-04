import pandas as pd
import numpy as np
#import lightgbm as lgb
import json
from gensim.models import word2vec
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import jieba
import pandas as pd
#%matplotlib inline
train_ad = pd.read_csv("./train/ad.csv")
train_click_log = pd.read_csv("./train/click_log.csv")
train_user = pd.read_csv("./train/user.csv")
test_ad = pd.read_csv("./test/ad.csv")
test_click_log = pd.read_csv("./test/click_log.csv")
    
train_data = pd.merge(train_click_log,train_ad,on = "creative_id",how = "left")
train_data = pd.merge(train_data,train_user,on = "user_id",how = "left")
train_data.shape

test_data = pd.merge(test_click_log,test_ad,on = "creative_id",how = "left")
def lis(x):
    out = []
    for i in x:
        out.append(str(i))
    return out

def data_process(data):
    result = pd.DataFrame()
    columns = ["creative_id","ad_id","product_id","product_category","advertiser_id","industry"]
    count = 0
    for column in columns:
        data_id = data[["user_id","time",column]].sort_values(by = ["user_id","time"],ascending = True)
        data_embedding_id = data_id.groupby(["user_id"])[column].apply(lis).reset_index().rename(columns = {column:column+"embedding"})
        if count == 0:
            result = data_embedding_id
        else:
            result = pd.merge(result,data_embedding_id,on = "user_id",how = "left")
        count+=1
    return result

train_embedding_id = data_process(train_data)
test_embedding_id = data_process(test_data)

train_embedding_id["age"] = train_user["age"]
train_embedding_id["gender"] = train_user["gender"]
train_embedding_id["y"] = [str(train_user["age"][i])+"_"+str(train_user["gender"][i]) for i in range(900000)]
train_embedding_id["y"] = train_embedding_id["y"].map({"1_1":1,"1_2":2,"2_1":3,"2_2":4,"3_1":5,"3_2":6,"4_1":7,"4_2":8,"5_1":9,"5_2":10,"6_1":11,"6_2":12,"7_1":13,"7_2":14,"8_1":15,"8_2":16,"9_1":17,"9_2":18,"10_1":19,"10_2":20})
print(train_embedding_id.head())
print(train_embedding_id["y"].unique())
train_embedding_id.to_csv("./train_embedding_id.csv",index = False)
test_embedding_id.to_csv("./test_embedding_id.csv",index = False)


"""
train_data_no_y = train_data[["time","creative_id","click_times","ad_id","product_id","product_category","advertiser_id","industry"]]
test_data_no_y = test_data[["time","creative_id","click_times","ad_id","product_id","product_category","advertiser_id","industry"]]
all_data = pd.concat([train_data_no_y,test_data_no_y],axis = 0)
for column in all_data.columns:
    all_data[column] = [column+"_"+str(i) for i in all_data[column]]
print(all_data.head())
all_data.to_csv("./all_data.csv",index = False)
"""
