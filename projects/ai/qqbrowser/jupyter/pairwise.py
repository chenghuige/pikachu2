# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import sys, os 
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')
import glob
import numpy as np
import pandas as pd
import copy
from collections import Counter
import gezi
from gezi import tqdm
import plotly.express as px


# %%
# label = pd.read_csv('../input/pairwise/label.csv')


# %%
labels = pd.read_csv('../input/pairwise/label.tsv', sep='\t', header=None, names=['query', 'candidate', 'relevance'])
labels['id'] = range(len(labels))
labels['pid'] = labels.apply(lambda x: gezi.hash_str(str(x.query) + str(x.candidate)).upper(), axis=1)
labels['query'] = labels['query'].astype(int)
labels['candidate'] = labels['candidate'].astype(int)
labels['similarity'] = -1.
label = labels[['id', 'pid', 'query', 'candidate', 'relevance', 'similarity']]
label_ori = label


# %%
label


# %%
label.relevance.describe()


# %%
set(label.relevance)


# %%
len(set(label.relevance))


# %%
label['relevance2'] = label.relevance.apply(lambda x: int(x * 20))


# %%
label


# %%
m = {}
for row in label.itertuples():
  m[f'{row.query}\t{row.candidate}'] = row.relevance


# %%
def get_relevance(query, candidate):
  score = m.get(f'{query}\t{candidate}', None)
  if score is None:
    return m.get(f'{candidate}\t{query}', None)
  else:
    return score


# %%
relevance_counts = label.groupby(['relevance2']).size().reset_index(name='counts')


# %%
relevance_counts


# %%
total = sum(relevance_counts.counts)
total


# %%
ratios = [x/total for x in relevance_counts.counts]
ratios


# %%
len(ratios)


# %%
labels = [0] * (len(ratios) + 1)
for i in range(len(ratios)):
  labels[i + 1] = labels[i] + ratios[i]
labels


# %%
for i in range(len(labels) - 1):
  labels[i] = (labels[i] + labels[i + 1]) / 2.


# %%
labels


# %%
px.pie(relevance_counts, names='relevance2', values='counts')


# %%
uset = gezi.UnionSet()
uset2 = gezi.UnionSet()


# %%
for row in tqdm(label.itertuples(), total=len(label)):
  uset.join(row.query, row.candidate)
  if row.relevance == 1:
    uset2.join(row.query, row.candidate)


# %%
uset.parent


# %%
uset.find(2345203561710400875)


# %%
uset.find(2336192438652390830)


# %%
len(uset.clusters())


# %%
len(uset2.clusters())


# %%
clusters = uset.clusters()


# %%
l = [(key, len(clusters[key])) for key in clusters]


# %%
l


# %%
l = sorted(l, key=lambda x: -x[1])


# %%
l


# %%
clusters2 = uset2.clusters()
l2 = [(key, len(clusters2[key])) for key in clusters2]
l2 = sorted(l2, key=lambda x: -x[1])
l2


# %%
label[label['query'] == 345615612387104174]


# %%
label[label['candidate'] == 345615612387104174]


# %%
clusters2[345615612387104174]


# %%
label[label['query'] == 8064786459832520110]


# %%
label[label['candidate'] == 8064786459832520110]


# %%
label[label['query'] == 2462308248765570478]


# %%
label[label['candidate'] == 2462308248765570478]


# %%
uset2.find(8064786459832520110)


# %%
uset2.find(345615612387104174)


# %%
rows = []
for row in tqdm(label.itertuples(), total=len(label)):
  row = row._asdict()
  del row['Index']
  row['ori'] = 1
  rows.append(copy.copy(row))
  query, candidate = row['query'], row['candidate']
  for x in clusters2.get(uset2.find(query), []):
    if x != query and x != candidate:
      score = get_relevance(x, candidate)
      if score is None:
        row2 = copy.copy(row)
        row2['query'] = x
        m[f'{x}\t{candidate}'] = row['relevance']
        row2['ori'] = 0
        rows.append(row2)
  for x in clusters2.get(uset2.find(candidate), []):
    if x != candidate and x != query:
      score = get_relevance(x, candidate)
      if score is None:
        row2 = copy.copy(row)
        row2['candidate'] = x
        m[f'{query}\t{x}'] = row['relevance']
        row2['ori'] = 0
        rows.append(row2)
  
ndf = pd.DataFrame(rows)


# %%
ndf


# %%
uset2.find(5128432776742593966)


# %%
uset2.find(2345203561710400875)


# %%
len(clusters2[2678482180875490734])


# %%
6668664740488695214 in clusters2[2678482180875490734]


# %%
ndf.to_csv('../input/pairwise/label_new.csv', index=False)


# %%
ndf[ndf.ori == 1]


# %%
np.random.seed(1024)


# %%
l2 = l[1:]


# %%
np.random.shuffle(l2)


# %%
l2


# %%
FOLDS = 3


# %%
count = -(-len(label) // FOLDS)
count


# %%
total = sum([x[1] for x in l])
total


# %%
count = -(-(total - l[0][1]) // (FOLDS - 1))
count


# %%
groups = {
  0: [l[0][0]]
}
groups


# %%
group_counts = {
  0: l[0][1]
}
group_counts


# %%
group_map = {
  l[0][0]: 0
}
group_map


# %%
group = set()
group_index = 1
for i in range(len(l2)):
  group.add(l2[i][0]) 
  if group_index not in group_counts:
    group_counts[group_index] = l2[i][1]
  else:
    group_counts[group_index] += l2[i][1]
  group_map[l2[i][0]] = group_index

  if group_counts[group_index] > count:
    groups[group_index] = list(group)
    group_index += 1
    group = set()
groups[group_index] = list(group)


# %%
groups


# %%
len(groups)


# %%
group_counts


# %%
group_map


# %%
groups_ = []
for row in tqdm(label.itertuples(), total=len(label)):
  root = uset.find(row.query)
  groups_.append(group_map[root])
label['group'] = groups_


# %%
label


# %%
for i in range(FOLDS):
  print(i, len(label[label.group==i]))


# %%
len(label) / 5


# %%
counter = Counter()
for row in tqdm(label.itertuples(), total=len(label)):
  counter[row.query] += 1
  counter[row.candidate] += 1


# %%
counter


# %%
counter.most_common(100)


# %%
label[label['candidate']==8389040351619993006]


# %%
label[label['candidate']==4993326534942504366]


# %%
counterq = Counter()
for row in tqdm(label.itertuples(), total=len(label)):
  counterq[row.query] += 1


# %%
counterq.most_common(100)


# %%
counterq


# %%
label


# %%
FOLDS = 20


# %%
# label = label_ori
label = ndf[ndf.ori == 1]
# label_valid = label[label.id < -(-len(label) // 5)]
label['hash_group'] = label.pid.apply(lambda x: gezi.hash(x) % FOLDS)
label_valid = label[label.hash_group == 0]


# %%
label_valid


# %%
len(label) / FOLDS


# %%
valid_set = set(label_valid['query']) | set(label_valid['candidate'])


# %%
len(valid_set)


# %%
len(set(label['query']) | set(label['candidate']))


# %%
# label_train = label[label.id >= -(-len(label) // 5)]
label_train = label[label.hash_group != 0]
# label_train = label[(label.hash_group != 0) | (label.ori == 0)]
# label_train = pd.concat([label_train, ndf[ndf.ori == 0]])


# %%
label_train


# %%
label_train = label_train[~label_train['query'].isin(valid_set) & ~label_train['candidate'].isin(valid_set)]
label_train


# %%
label_train[label_train.ori == 0]


# %%
label_train[label_train.ori == 1]


# %%
label.to_csv('../input/pairwise/label.csv')


# %%
px.pie(label.groupby(['relevance']).size().reset_index(name='counts'), names='relevance', values='counts')


# %%
px.pie(label[(label.relevance != 1)&(label.relevance !=0)].groupby(['relevance']).size().reset_index(name='counts'), names='relevance', values='counts')


# %%
px.pie(ndf.groupby(['relevance']).size().reset_index(name='counts'), names='relevance', values='counts')


# %%
px.pie(ndf[(ndf.relevance!=1)&(ndf.relevance!=0)].groupby(['relevance']).size().reset_index(name='counts'), names='relevance', values='counts')


# %%
px.pie(label_train.groupby(['relevance']).size().reset_index(name='counts'), names='relevance', values='counts')


# %%
px.pie(label_valid.groupby(['relevance']).size().reset_index(name='counts'), names='relevance', values='counts')


# %%
label_valid.to_csv('../input/pairwise/label_valid0.csv', index=False)


# %%
label_train.to_csv('../input/pairwise/label_train0.csv', index=False)


# %%
label_train


# %%
label_new = ndf[ndf.ori == 0]
ic(len(label_new))
label_new.to_csv('../input/pairwise/label_new.csv', index=False)

for i in range(2):
  label_valid = label[label.hash_group == i]
  label_valid.to_csv(f'../input/pairwise/label_valid{i}.csv', index=False)
  valid_set = set(label_valid['query']) | set(label_valid['candidate'])
  label_train = label[label.hash_group != i]
  # label_train = pd.concat([label_train, ndf[ndf.ori==0]], axis=0)
  label_train = label_train[~label_train['query'].isin(valid_set) & ~label_train['candidate'].isin(valid_set)]
  ic(len(label_train))
  label_train.to_csv(f'../input/pairwise/label_train{i}.csv', index=False)
  label_new_train = label_new[~label_new['query'].isin(valid_set) & ~label_new['candidate'].isin(valid_set)]
  ic(len(label_new_train))
  label_new_train.to_csv(f'../input/pairwise/label_new_train{i}.csv', index=False)
  label_new2_train = label_new_train[(label_new_train.relevance!=1)&(label_new_train.relevance!=0)]
  ic(len(label_new2_train))
  label_new2_train.to_csv(f'../input/pairwise/label_new2_train{i}.csv', index=False)


# %%


