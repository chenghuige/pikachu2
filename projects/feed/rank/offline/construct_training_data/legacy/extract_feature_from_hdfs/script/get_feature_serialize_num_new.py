#encoding:gbk

import sys

one_hot_feature_dict = {}
one_feature_dict = {}
with open(sys.argv[1]) as f:
    for line in f:
        line = line.strip()
        st = line.split("\t")
        if len(st) != 2:
            continue
        if st[1] == "0":
            one_feature_dict[st[0]] = 1

with open(sys.argv[2]) as f:
    for line in f:
        line = line.strip()
        st = line.split("\t")
        if st[0] != "ONEHOT":
            continue
        fe = st[1]
        count = int(st[2])
        fest = fe.split("\a")
        if len(fest) != 2:
            continue
        if fest[0] not in one_hot_feature_dict:
            one_hot_feature_dict.setdefault(fest[0], {})
        one_hot_feature_dict[fest[0]].setdefault(fest[1], 0)


f_serial = open(sys.argv[3], "w")
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
