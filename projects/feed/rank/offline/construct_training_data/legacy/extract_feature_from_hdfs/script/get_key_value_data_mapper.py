#encoding:gbk

import sys

one_hot_feature_dict = {}
with open(sys.argv[1]) as f:
    for line in f:
        line = line.strip()
        st = line.split("\t")
        if len(st) != 2:
            continue
        if st[1] == "1": #0:real value 1:one hot 2:embed
            one_hot_feature_dict.setdefault(st[0], {})

for line in sys.stdin:
    line = line.strip()
    st = line.split("\t")
    st = st[1:]
    for fe in st:
        fest = fe.split("\a")
        if len(fest) != 2:
            continue
        if fest[0] not in one_hot_feature_dict:
            continue
        flist = fest[1].split("\b")
        for f in flist:
            name="ONEHOT\t%s\a%s" % (fest[0],f)
            print(name)
