#encoding:gbk

import sys

one_hot_feature_dict = {}
one_feature_dict = {}
#one_feature_min = {}
#one_feature_max = {}
with open(sys.argv[1]) as f:
    for line in f:
        line = line.strip()
        st = line.split("\t")
        if len(st) != 2:
            continue
        if st[1] == "1":
            one_hot_feature_dict.setdefault(st[0], {})
        elif st[1] == "0":
            one_feature_dict[st[0]] = 1

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
            one_hot_feature_dict[fest[0]].setdefault(f, 0)


f_serial = open(sys.argv[2], "w")
serial_num = 1
for k in one_feature_dict.keys():
    f_serial.write(k + "\t" + str(serial_num) + "\t" + str(serial_num + 1) + "\t" + str(serial_num + 2) + "\n")
    serial_num = serial_num + 3

for k1 in one_hot_feature_dict.keys():
    print k1 + "\t" + str(len(one_hot_feature_dict[k1].keys()))
    for k2 in one_hot_feature_dict[k1].keys():
        f_serial.write(k1 + "\a" + k2 + "\t" + str(serial_num) + "\n")
        serial_num = serial_num + 1
        #f.write(k1 + "\a" + k2 + "\t" + str(serial_num) + "\n")
        #serial_num = serial_num + 1
