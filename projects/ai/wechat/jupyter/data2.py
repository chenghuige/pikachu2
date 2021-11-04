#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from collections import OrderedDict
import glob
import sys 
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')
import gezi
from gezi import tqdm


# In[3]:


glob.glob('../input/*')


# In[4]:


d = pd.read_csv('../input/feed_info.csv')


# In[5]:


d.head()


# In[7]:


d_valid = pd.read_csv('../input/user_action2.csv')


# In[8]:


d_valid.head()


# In[9]:


d_test = pd.read_csv('../input/test_a.csv')


# In[10]:


d_test.head()


# In[12]:


d_valid = d_valid.join(d.set_index('feedid'), on='feedid')


# In[24]:


d_valid = d_valid[d_valid.date_ == 14]


# In[13]:


d_test = d_test.join(d.set_index('feedid'), on='feedid')


# In[25]:


len(d_valid)


# In[26]:


d_valid.bgm_song_id.isna().sum() / len(d_valid)


# In[29]:


d_valid.bgm_singer_id.isna().sum() / len(d_valid)


# In[23]:


d_test.bgm_song_id.isna().sum() / len(d_test)


# In[30]:


d_test.bgm_singer_id.isna().sum() / len(d_test)


# In[31]:


d_valid.manual_tag_list.isna().sum() / len(d_valid)


# In[32]:


d_valid.machine_tag_list.isna().sum() / len(d_valid)


# In[33]:


d_valid.manual_keyword_list.isna().sum() / len(d_valid)


# In[34]:


d_valid.machine_keyword_list.isna().sum() / len(d_valid)


# In[35]:


d_valid.description.isna().sum() / len(d_valid)


# In[36]:


d_test.manual_keyword_list.isna().sum() / len(d_test)


# In[27]:


len(d_valid)


# In[28]:


len(d_test)


# In[5]:


d.describe()


# In[39]:


MAX_TAGS = 14
MAX_KEYS = 18
DESC_LEN = 128
DESC_CHAR_LEN = 256
WORD_LEN = 256
START_ID = 3
UNK_ID = 1
EMPTY_ID = 2
NAN_ID = -1


# In[40]:


d = d.fillna(NAN_ID)


# In[41]:


d.head()


# In[42]:


d.feedid = d.feedid.astype(int)
d.authorid = d.authorid.astype(int)
d.bgm_singer_id = d.bgm_singer_id.astype(int)
d.bgm_song_id = d.bgm_song_id.astype(int)
with open('../input/word_corpus.txt', 'w') as f:
  for i, desc in enumerate(d.description.values):
    if str(desc) == str(NAN_ID):
      continue
    print(desc, file=f)
  for i, ocr in enumerate(d.ocr.values):
    if str(ocr) == str(NAN_ID):
      continue
    print(ocr, file=f)
  for i, asr in enumerate(d.asr.values):
    if str(asr) == str(NAN_ID):
      continue
    print(asr, file=f)
    
with open('../input/char_corpus.txt', 'w') as f:
  for i, desc in enumerate(d.description_char.values):
    if str(desc) == str(NAN_ID):
      continue
    print(desc, file=f)
  for i, ocr in enumerate(d.ocr_char.values):
    if str(ocr) == str(NAN_ID):
      continue
    print(ocr, file=f)
  for i, asr in enumerate(d.asr_char.values):
    if str(asr) == str(NAN_ID):
      continue
    print(asr, file=f)


# In[12]:


with open('../input/author_corpus.txt', 'w') as f:
  for i, author in enumerate(d.authorid.values):
    if author == NAN_ID:
      continue
    print(author, file=f)
with open('../input/singer_corpus.txt', 'w') as f:
  for i, singer in enumerate(d.bgm_singer_id.values):
    if singer == NAN_ID:
      continue
    print(singer, file=f)
with open('../input/song_corpus.txt', 'w') as f:
  for i, song in enumerate(d.bgm_song_id.values):
    if song == NAN_ID:
      continue
    print(song, file=f)


# In[13]:


with open('../input/tag_corpus.txt', 'w') as f:
  for i, tag in enumerate(d.manual_tag_list):
    if str(tag) == str(NAN_ID):
      continue
    else:
      print(str(tag).replace(';', ' '), file=f)
  for i, tag in enumerate(d.machine_tag_list):
    if str(tag) == str(NAN_ID):
      continue
    else:
      print(' '.join([x.split()[0] for x in tag.split(';')]), file=f)
    
with open('../input/key_corpus.txt', 'w') as f:
  for i, kw in enumerate(d.manual_keyword_list):
    if str(kw) == str(NAN_ID):
      continue
    else:
      print(str(kw).replace(';', ' '), file=f)
  for kw in d.machine_keyword_list:
    if str(kw) == str(NAN_ID):
      continue
    else:
      print(str(kw).replace(';', ' '), file=f)


# In[25]:


# generate feed_info.npy

# MIN_COUNT = None
MIN_COUNT = 5
    
single_keys = ['author', 'song', 'singer']
multi_keys =  ['manual_tags', 'machine_tags', 'machine_tag_probs', 'machine_tags2', 'manual_keys', 'machine_keys', 'desc', 'desc_char', 'ocr', 'asr']
info_keys = single_keys + multi_keys
info_lens = [1] * len(single_keys) + [MAX_TAGS, MAX_TAGS, MAX_TAGS, MAX_TAGS, MAX_KEYS, MAX_KEYS, DESC_LEN, DESC_CHAR_LEN, WORD_LEN, WORD_LEN]

vocab_names = [
                'user', 'doc',
                'author', 'singer', 'song',
                'key', 'tag', 'word', 'char'
              ]
vocabs = {}

for vocab_name in vocab_names:
  vocab_file =  f'../input/{vocab_name}_vocab.txt'
  # if not doc then mask as UNK for rare words, tags, keys..
  min_count = None if vocab_name == 'doc' else MIN_COUNT
#   min_count = MIN_COUNT if vocab_name in ['word', 'char'] else None
  vocab = gezi.Vocab(vocab_file, min_count=min_count)
  vocabs[vocab_name] = vocab
    
d = pd.read_csv('../input/feed_info.csv')
d = d.fillna(NAN_ID)
d.feedid = d.feedid.astype(int)
d.authorid = d.authorid.astype(int)
d.bgm_singer_id = d.bgm_singer_id.astype(int)
d.bgm_song_id = d.bgm_song_id.astype(int)

def gen_unk():
  l = [0] * sum(info_lens)
  x = 0
  for len_ in info_lens:
    l[x] = UNK_ID
    x += len_
  return l


def gen_empty():
  l = [0] * sum(info_lens)
  x = 0
  for len_ in info_lens:
    l[x] = EMPTY_ID
    x += len_
  return l

embs = [
    gen_empty()
] * 120000
embs[0] = [0] * sum(info_lens)
embs[1] = gen_unk()

def set_feature(feature, feed, key, feedid):
  key_ = key.replace('description', 'desc')
  if str(feed[key]) != str(NAN_ID):
    feature[key_] = list(map(int, str(feed[key]).split())) 
  else:
    feature[key_] = []

#   MAX_LEN = 512 
  MAX_LEN = 256
  if key_ == 'desc':
    MAX_LEN = 128
  elif key_ == 'desc_char':
    MAX_LEN = 256 
    
  vocab_name = 'char' if 'char' in key else 'word'
  vocab = vocabs[vocab_name]
  feature[key_] = [vocab.id(x) for x in feature[key_]]
  feature[key_] = gezi.pad(feature[key_], MAX_LEN, 0) 

def is_neg(row, play=None):
  for action in ACTIONS:
    if row[action] > 0:
      return False

  return True

def is_finish(row):
  return row['finish_rate'] > 0.99

def is_dislike(row):
  return row['finish_rate'] < 0.01

GOOD_TAG_PROB = 0.2

for _, row in tqdm(d.iterrows(), total=len(d), ascii=True, desc='feed_info'):
  author = row['authorid']
  feedid = row['feedid']
  song = row['bgm_song_id']
  singer = row['bgm_singer_id']
  l = [
      vocabs['author'].id(author),
      vocabs['song'].id(song),
      vocabs['singer'].id(singer)
  ]
    
  manual_tags = str(row['manual_tag_list']).split(';')
  manual_tags_ = []
  for tag in manual_tags:
    if tag == 'nan' or tag == str(NAN_ID):
      continue
    else:
      manual_tags_.append(vocabs['tag'].id(int(tag)))
  manual_tags_ = gezi.pad(manual_tags_, MAX_TAGS, 0)
  l += manual_tags_
  
  machine_tags = str(row['machine_tag_list']).split(';')
  machine_tags_ = []
  machine_tag_probs_ = []
  machine_tags2_ = []
  for item in machine_tags:
    if item == 'nan' or item == str(NAN_ID):
      continue
    else:
      x = item.split()
      tag, prob = int(x[0]), float(x[1])
      machine_tags_.append(vocabs['tag'].id(tag))
      machine_tag_probs_.append(int(prob * 1e7))
      if prob > GOOD_TAG_PROB:
        machine_tags2_.append(vocabs['tag'].id(tag))
  machine_tags_ = gezi.pad(machine_tags_, MAX_TAGS, 0)
  l += machine_tags_
  machine_tag_probs_ = gezi.pad(machine_tag_probs_, MAX_TAGS, 0)
  l += machine_tag_probs_
  machine_tags2_ = gezi.pad(machine_tags_, MAX_TAGS, 0)
  l += machine_tags2_
  

  manual_keys = str(row['manual_keyword_list']).split(';')
  manual_keys_ = []
  for key in manual_keys:
    if key == 'nan' or key == str(NAN_ID):
      continue
    else:
      manual_keys_.append(vocabs['key'].id(int(key)))
  manual_keys_ = gezi.pad(manual_keys_, MAX_KEYS, 0)
  l += manual_keys_
    
  machine_keys = str(row['machine_keyword_list']).split(';')
  machine_keys_ = []
  for key in machine_keys:
    if key == 'nan' or key == str(NAN_ID):
      continue
    else:
      machine_keys_.append(vocabs['key'].id(int(key)))
  machine_keys_ = gezi.pad(machine_keys_, MAX_KEYS, 0)
  l += machine_keys_
    
  feature = OrderedDict()
#   for key in ['description', 'description_char', 'ocr', 'ocr_char', 'asr', 'asr_char']:
  for key in ['description', 'description_char', 'ocr', 'asr']:
    set_feature(feature, row, key, feedid)
    
  for key in feature:
    l += feature[key]

  embs[vocabs['doc'].id(int(row['feedid']))] = l

embs = np.asarray(embs)
print(embs.shape, sum(info_lens))

if MIN_COUNT:
  np.save('../input/doc_lookup.npy', embs)
else:
  np.save('../input/doc_ori_lookup.npy', embs)


# In[26]:


np.sum(embs == 1)


# In[27]:


np.sum(embs == 1)


# In[28]:


embs


# In[29]:


vocabs['author'].key(4141)


# In[35]:


embs[1246]


# In[30]:


embs[vocabs['doc'].id(77432)]


# In[31]:


embs[vocabs['doc'].id(43549)]


# In[32]:


embs[vocabs['doc'].id(12921)]


# In[229]:


707 - 451


# In[329]:


d.head()


# In[321]:


d.manual_tag_list


# In[320]:


d.machine_tag_list


# In[341]:


d.bgm_singer_id


# In[66]:


d.machine_tag_list


# In[67]:


d.machine_tag_list.apply(lambda x: len(str(x).split(';'))).describe()


# In[68]:


d.manual_tag_list.apply(lambda x: len(str(x).split(';'))).describe()


# In[69]:


d.machine_tag_list[0]


# In[70]:


tags = set()
for i in range(len(d)):
  for item in str(d.machine_tag_list[i]).split(';'):
    if item == 'nan':
      continue
    if len(item.split()) == 1:
      print(d.machine_tag_list[i])
      continue
    tags.add(int(item.split()[0]))


# In[71]:


min(tags)


# In[72]:


max(tags) # 400


# In[73]:


tags = set()
for i in range(len(d)):
  for item in str(d.manual_tag_list[i]).split(';'):
    if item == 'nan':
      continue
    tags.add(int(item.split()[0]))
  break


# In[74]:


min(tags)


# In[75]:


max(tags)


# In[76]:


keywords = set()
for i in range(len(d)):
  for item in str(d.manual_keyword_list[i]).split(';'):
    if item == 'nan':
      continue
    keywords.add(int(item.split()[0]))
  break


# In[77]:


min(keywords)


# In[78]:


max(keywords)


# In[79]:


keywords = set()
for i in range(len(d)):
  for item in str(d.machine_keyword_list[i]).split(';'):
    if item == 'nan':
      continue
    keywords.add(int(item.split()[0]))
  break


# In[80]:


min(keywords)


# In[81]:


max(keywords) # 30000


# In[82]:


words = set()
for i in range(len(d)):
  for item in str(d.description[i]).split():
    if item == 'nan':
      continue
    words.add(int(item.split()[0]))
  break


# In[83]:


min(words)


# In[84]:


max(words)


# In[85]:


words = set()
for i in range(len(d)):
  for item in str(d.ocr[i]).split():
    if item == 'nan':
      continue
    words.add(int(item.split()[0]))
  break


# In[86]:


min(words)


# In[87]:


max(words)


# In[88]:


words = set()
for i in range(len(d)):
  for item in str(d.asr[i]).split():
    if item == 'nan':
      continue
    words.add(int(item.split()[0]))
  break
print(min(words), max(words))


# In[89]:


chars = set()
for i in range(len(d)):
  for item in str(d.asr_char[i]).split():
    if item == 'nan':
      continue
    chars.add(int(item.split()[0]))
  break
print(min(chars), max(chars))


# In[90]:


chars = set()
for i in range(len(d)):
  for item in str(d.ocr_char[i]).split():
    if item == 'nan':
      continue
    chars.add(int(item.split()[0]))
  break
print(min(chars), max(chars))


# In[91]:


chars = set()
for i in range(len(d)):
  for item in str(d.description_char[i]).split():
    if item == 'nan':
      continue
    chars.add(int(item.split()[0]))
  break
print(min(chars), max(chars))


# In[92]:


d.manual_tag_list.apply(lambda x: len(str(x).split(';'))).describe()


# In[93]:


d.machine_keyword_list.apply(lambda x: len(str(x).split(';'))).describe()


# In[94]:


d.manual_keyword_list.apply(lambda x: len(str(x).split(';'))).describe()


# In[95]:


d.description_char.apply(lambda x: len(str(x).split())).describe([0.25,0.5,0.75,0.9,0.99, 0.999])


# In[96]:


d.description.apply(lambda x: len(str(x).split())).describe([0.25,0.5,0.75,0.9,0.99, 0.999])


# In[97]:


d.ocr_char.apply(lambda x: len(str(x).split())).describe([0.25,0.5,0.75,0.9,0.99, 0.999])


# In[98]:


d.asr_char.apply(lambda x: len(str(x).split())).describe([0.25,0.5,0.75,0.9,0.99, 0.999])


# In[99]:


d.ocr.apply(lambda x: len(str(x).split())).describe([0.25,0.5,0.75,0.9,0.99, 0.999])


# In[100]:


d.asr.apply(lambda x: len(str(x).split())).describe([0.25,0.5,0.75,0.9,0.99, 0.999])


# In[101]:


len(d)


# In[102]:


d.videoplayseconds.describe([0.25,0.5,0.75,0.9,0.99,0.999])


# In[103]:


d[d.videoplayseconds>60].videoplayseconds


# In[104]:


len(set(d.feedid))


# In[105]:


len(set(d.bgm_song_id.astype('Int32')))


# In[106]:


d.bgm_song_id.astype('Int32').max()


# In[107]:


d.bgm_song_id.astype('Int32').min()


# In[108]:


d.bgm_singer_id.astype('Int32').max()


# In[109]:


d.feedid.max()


# In[110]:


d.feedid.min()


# In[111]:


d.authorid.max()


# In[112]:


len(set(d.authorid))


# In[113]:


feeds = {}
def cache_feed():
  df = pd.read_csv('../input/feed_info.csv')
  df = df.fillna(-1)
  for _, row in tqdm(df.iterrows(), total=len(df), ascii=True, desc='feed_info'):
    feeds[row['feedid']] = row
cache_feed()


# In[114]:


d2 = pd.read_csv('../input/user_action.csv')


# In[115]:


d2.read_comment.mean()


# In[116]:


d2.comment.mean()


# In[117]:


d2.like.mean()


# In[118]:


d2.click_avatar.mean()


# In[119]:


d2.forward.mean()


# In[121]:


# def get_finish_rate(feedid, play):
#     return play / 1000 / feeds[feedid].videoplayseconds

# d2['finish_rate'] = list(map(get_finish_rate,d2['feedid'],d2['play']))
finish_rates = []
for _, row in tqdm(d2.iterrows(), total=len(d2), ascii=True, desc='user_action'):
  finish_rates.append(row['play'] / 1000 / feeds[row['feedid']]['videoplayseconds'])
d2['finish_rate'] = finish_rates


# In[123]:


d2.head()


# In[183]:


d[d.feedid == 13265]


# In[182]:


d2[d2.finish_rate < 0.01].sort_values(['play'], ascending=False)


# In[297]:


# get history
# TODO add dislike ? no action and finish rate < 0.1 and stay i
# play time
# staty time as history, play time >  60s play time < 5s like this
# TODO make scripts and parallel base on user
# 历史收益比较大 尝试更多的可能历史

# 引入用户最近观看历史 50 个 ？    每个配合 各种action emb
# latest history  综合表示用户近期历史 ？ TODO

# finish rate 正负 参数如何最好
# stay rate  
# 另外全局角度 actition rate, finish rate, stay rate

ACTIONS = [
  'read_comment',
  'comment',
  'like',
  'click_avatar',
  'forward',
  'follow',
  'favorite'
]

history = {
  'action': {},
  'finish': {},
  'neg': {},
  'dislike': {},
  'latest': {}
}

for action in ACTIONS:
  history[action] = {}

userids = [int(x.strip().split()[0]) for x in open('../input/user_vocab.txt').readlines()]
for userid in userids:
  history['action'][userid] = []
  history['finish'][userid] = []
  history['neg'][userid] = []
  history['dislike'][userid] = []
  history['latest'][userid] = []
  for action in ACTIONS:
    history[action][userid] = []

def is_neg(row, play=None):
  for action in ACTIONS:
    if row[action] > 0:
      return False

  return True


visit_days = {}

d2 = d2.sort_values(['date_'], ascending=False)
for _, row in tqdm(d2.iterrows(), total=len(d2), ascii=True, desc='user_action'):
  feedid = int(row['feedid'])
  userid = int(row['userid'])
  visit_days[f'{userid}_{feedid}'] = int(row['date_'])
  history['latest'][userid].append(feedid)
  if row['finish_rate'] > 0.99:
    history['finish'][userid].append(feedid)
  if row['finish_rate'] < 0.01:
    history['dislike'][userid].append(feedid)
  is_neg_row = is_neg(row)
  if not is_neg_row:
    history['action'][userid].append(feedid)
  else:
    history['neg'][userid].append(feedid)
  for action in ACTIONS:
    if row[action] > 0:
      history[action][userid].append(feedid)


# In[298]:


gezi.save_pickle(visit_days, '../input/visit_days.pkl')


# In[299]:


gezi.save_pickle(history, '../input/history.pkl')


# In[287]:


# for key in history:
#   for key_ in history[key]:
#     history[key][key_] = list(map(int, history[key][key_]))


# In[303]:


m = {
  'userid': [],
  'action': [],
  'action_len': [],
  'finish': [],
  'finish_len': [],
  'neg': [],
  'neg_len': [],
  'dislike': [],
  'dislike_len': [],  
  'latest': [],
  'latest_len': [],
}
OTHER_ACTIONS = ['action', 'finish', 'neg', 'dislike', 'latest']
for action in ACTIONS:
  m[action] = []
  m[f'{action}_len'] = []

for userid in tqdm(userids):
  m['userid'].append(userid)
  for action in ACTIONS + OTHER_ACTIONS:
    m[action].append(' '.join(map(str, history[action][userid])))
    m[f'{action}_len'].append(len(history[action][userid]))


# In[304]:


his_df = pd.DataFrame(m)


# In[305]:


his_df


# In[307]:


his_df.describe(percentiles=[.25,.5,.75,.9,.95,.99, .999])


# In[217]:


his_df[his_df.action_len == 0]


# In[201]:


his_df.to_csv('../input/history.csv', index=False)


# In[163]:


gezi.save_pickle(history, '../input/history.pkl')


# In[212]:


visit_days = {}

for _, row in tqdm(d2.iterrows(), total=len(d2), ascii=True, desc='user_action'):
  feedid = int(row['feedid'])
  userid = int(row['userid'])
  visit_days[f'{userid}_{feedid}'] = int(row['date_'])


# In[216]:


gezi.save_pickle(visit_days, '../input/visit_days.pkl')


# In[214]:


visit_days['55462_72135']


# In[215]:


visit_days['8_51791']


# In[207]:


history['read_comment']


# In[145]:


d2[d2.date_ == 14].to_csv('../input/valid.csv', index=False)


# In[146]:


d2.to_csv('../input/user_action2.csv', index=False)


# In[125]:


d2.finish_rate.describe()


# In[126]:


dneg = d2.loc[(d2.read_comment==0) & (d2.comment == 0) & (d2.like == 0) & (d2.click_avatar == 0) & (d2.forward == 0) & (d2.follow == 0) & (d2.favorite ==0)]


# In[127]:


dpos = d2.loc[(d2.read_comment>0) | (d2.comment > 0) | (d2.like > 0) | (d2.click_avatar > 0) | (d2.forward > 0) | (d2.follow > 0) | (d2.favorite >0)]


# In[128]:


dneg.play.mean() / 1000


# In[129]:


dpos.play.mean() / 1000


# In[130]:


dneg.stay.mean() / 1000


# In[131]:


dpos.stay.mean() / 1000


# In[134]:


dneg.finish_rate.describe()


# In[135]:


dpos.finish_rate.describe()


# In[136]:


len(dneg[dneg.finish_rate > 0.5]) / len(dneg)


# In[138]:


len(dpos[dpos.finish_rate > 0.5]) / len(dpos)


# In[151]:


len(dneg[dneg.finish_rate > 0.99]) / len(dneg)


# In[152]:


len(dpos[dpos.finish_rate > 0.99]) / len(dpos)


# In[266]:


len(dpos[dpos.finish_rate >= 1]) / len(dpos)


# In[268]:


len(dneg[dneg.finish_rate >= 1]) / len(dneg)


# In[292]:


len(dneg) / (len(dneg) + len(dpos))


# In[ ]:





# In[167]:


len(dpos[dpos.finish_rate < 0.1]) / len(dpos)


# In[171]:


len(dpos[dpos.finish_rate < 0.01]) / len(dpos)


# In[184]:


len(dpos[dpos.play > 30000]) / len(dpos)


# In[187]:


len(dpos[dpos.play > 60000]) / len(dpos)


# In[192]:


len(dpos[dpos.play < 10000]) / len(dpos)


# In[193]:


len(dpos[dpos.stay / dpos.play > 1.1]) / len(dpos)


# In[194]:


len(dpos[dpos.stay / dpos.play > 2]) / len(dpos)


# In[196]:


len(dpos[dpos.stay / dpos.play > 1.05]) / len(dpos)


# In[197]:


len(dpos[dpos.stay / dpos.play > 1.01]) / len(dpos)


# In[198]:


len(dneg[dneg.stay / dneg.play > 2]) / len(dneg)


# In[199]:


len(dneg[dneg.stay / dneg.play > 1.05]) / len(dneg)


# In[200]:


len(dneg[dneg.stay / dneg.play > 1.01]) / len(dneg)


# In[189]:


len(dneg[dneg.play > 30000]) / len(dneg)


# In[190]:


len(dneg[dneg.play > 60000]) / len(dneg)


# In[191]:


len(dneg[dneg.play < 10000]) / len(dneg)


# In[170]:


len(dneg[dneg.finish_rate < 0.1]) / len(dneg)


# In[185]:


len(dneg[dneg.finish_rate < 0.1]) / len(dneg)


# In[186]:


len(dneg[dneg.finish_rate < 0.01]) / len(dneg)


# In[142]:


len(d2[d2.finish_rate > 0.9]) / len(d2)


# In[300]:


d2.head()


# In[301]:


dneg[dneg.finish_rate > 0.9].head()


# In[132]:


len(dneg) / len(d2)


# In[133]:


npos = len(d2) - len(dneg) 


# In[ ]:


len(d2)


# In[ ]:


npos + 0.2 * len(dneg)


# In[ ]:


npos / (npos + 0.2 * len(dneg))


# In[ ]:


npos / (npos + 0.25 * len(dneg))


# In[ ]:


d2.head(100)


# In[ ]:


d2.userid.max()


# In[ ]:


d2.userid.min()


# In[ ]:


len(d2)


# In[ ]:


len(set(d2.userid))


# In[ ]:


len(set(d2.feedid))


# In[ ]:


len(set(d2.date_))


# In[ ]:


set(d2.date_)


# In[ ]:


d3 = pd.read_csv('../input/feed_embeddings.csv')


# In[ ]:


d3.head()


# In[ ]:


len(d3)


# In[ ]:


d4 = pd.read_csv('../input/test_a.csv')


# In[ ]:


len(d4)


# In[ ]:


d4.head()


# In[ ]:


d4.feedid.max()


# In[ ]:


set(d4.device)


# In[ ]:


pd.read_csv('../baseline/data/feature/feedid_feature.csv')


# In[ ]:


pd.read_csv('../baseline/data/feature/userid_feature.csv')


# In[ ]:





# In[ ]:


d = pd.read_csv('../input/feed_embeddings.csv')


# In[ ]:


len(d.feed_embedding[0].split())


# In[148]:


d2['userid_hash'] = d2.userid.apply(lambda x: gezi.hash(str(x)))


# In[150]:


d2.sort_values(['userid_hash'], ascending=True)


# In[ ]:


get_ipython().run_line_magic('pinfo', 'gezi.pad')


# In[ ]:


gezi.dict_renamee

