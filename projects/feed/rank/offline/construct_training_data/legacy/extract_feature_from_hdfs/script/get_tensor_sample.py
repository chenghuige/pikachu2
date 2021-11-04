#coding=gbk

import sys
import math

feature_dict = {}
f = open(sys.argv[1], "r")
for line in f:
    line = line.rstrip()
    st = line.split("\t")
    if len(st) < 0:
        continue
    feature_dict[st[0]] = int(st[1])
f.close()

interval_dict = {}
f = open(sys.argv[2], "r")
for line in f:
    line = line.rstrip()
    st = line.split("\t")
    if len(st) != 3:
        continue
    interval_dict.setdefault(st[0], [])
    interval_dict[st[0]].append(float(st[1]))
    interval_dict[st[0]].append(float(st[2]))
f.close()

serial_dict = {}
f = open(sys.argv[3], "r")
for line in f:
    line = line.rstrip()
    st = line.split("\t")
    if len(st) < 2:
        continue
    serial_dict.setdefault(st[0], [])
    for s in st[1:]:
        serial_dict[st[0]].append(int(s))

f.close()
total_len = 28
'''total_len 28 or 29
  if 28 the last is usr_history and is NULL
  if 28 the 28th is CLICKTIMESPAN th 29th is usr_history'''
#f=open("./data/youtu/sparse_matrix_data/2019042915_23/part-00063","r")
#for line in f:
for line in sys.stdin:
    line = line.rstrip()
    st = line.split("\t")
    if len(st) < total_len:
        continue
    fix_continue_pos = ""
    fix_one_pos = ""
    embedding_pos = ""
    weight = 1.0
    for i in range(3, total_len-1):
        stt = st[i].split("\a")
        if len(stt) != 2:
            continue
        if stt[0] == "NOWREADDURATION":
            weight = float(stt[1])
            if weight > interval_dict["NOWREADDURATION"][1]:
                weight = interval_dict["NOWREADDURATION"][1]
        if stt[0] not in feature_dict:
            continue
        if feature_dict[stt[0]] < 0:
            continue
        if feature_dict[stt[0]] == 0:
            if stt[0] not in interval_dict:
                continue
            val = float(stt[1])
            if val < interval_dict[stt[0]][0]:
                val = interval_dict[stt[0]][0]
            if val > interval_dict[stt[0]][1]:
                val = interval_dict[stt[0]][1]
            val = (val - interval_dict[stt[0]][0])/(interval_dict[stt[0]][1] - interval_dict[stt[0]][0])
            if fix_continue_pos == "":
                fix_continue_pos = str(serial_dict[stt[0]][0]) + "\b" + str(math.sqrt(val)) + "\a" + str(serial_dict[stt[0]][1]) + "\b" + str(val) + "\a" + str(serial_dict[stt[0]][2]) + "\b" + str(math.pow(val, 2))
            else:
                fix_continue_pos = fix_continue_pos + "\a" + str(serial_dict[stt[0]][0]) + "\b" + str(math.sqrt(val)) + "\a" + str(serial_dict[stt[0]][1]) + "\b" + str(val) + "\a" + str(serial_dict[stt[0]][2]) + "\b" + str(math.pow(val, 2))
        elif feature_dict[stt[0]] == 1:
            if st[i] not in serial_dict:
                continue
            if fix_one_pos == "":
                fix_one_pos = str(serial_dict[st[i]][0]) + "\b1.0"
            else:
                fix_one_pos = fix_one_pos + "\a" + str(serial_dict[st[i]][0]) + "\b1.0"
        elif feature_dict[stt[0]] == 2:
            if embedding_pos == "":
                embedding_pos = st[i]
            else:
                embedding_pos = embedding_pos + "\t" + st[i]
    
    if len(st) == total_len:
        if fix_continue_pos == "":
            fix_continue_pos = str(serial_dict["CLICKTIMESPAN"][0]) + "\b1.0" + "\a" + str(serial_dict["CLICKTIMESPAN"][1]) + "\b1.0" + "\a" + str(serial_dict["CLICKTIMESPAN"][2]) + "\b1.0"
        else:
            fix_continue_pos = fix_continue_pos + "\a" + str(serial_dict["CLICKTIMESPAN"][0]) + "\b1.0" + "\a" + str(serial_dict["CLICKTIMESPAN"][1]) + "\b1.0" + "\a" + str(serial_dict["CLICKTIMESPAN"][2]) + "\b1.0"
        print st[0] + "\t" + st[2] + "\t" + str(weight) + "\t" + st[1] + "\t" + fix_continue_pos + "\t" + fix_one_pos + "\t" + embedding_pos + "\t" + st[total_len-1] #st[total_len-1] is NULL
    else:
        stt = st[total_len-1].split("\a") #CLICKTIMESPAN
        try:
            val = float(stt[1])
        except:
            #print("ERROR %s" % (line) )
            continue
        if val < interval_dict[stt[0]][0]:
            val = interval_dict[stt[0]][0]
        if val > interval_dict[stt[0]][1]:
            val = interval_dict[stt[0]][1]
        val = (val - interval_dict[stt[0]][0])/(interval_dict[stt[0]][1] - interval_dict[stt[0]][0])
        if fix_continue_pos == "":
            fix_continue_pos = str(serial_dict[stt[0]][0]) + "\b" + str(math.sqrt(val)) + "\a" + str(serial_dict[stt[0]][1]) + "\b" + str(val) + "\a" + str(serial_dict[stt[0]][2]) + "\b" + str(math.pow(val, 2))
        else:
            fix_continue_pos = fix_continue_pos + "\a" + str(serial_dict[stt[0]][0]) + "\b" + str(math.sqrt(val)) + "\a" + str(serial_dict[stt[0]][1]) + "\b" + str(val) + "\a" + str(serial_dict[stt[0]][2]) + "\b" + str(math.pow(val, 2))
        print st[0] + "\t" + st[2] + "\t" + str(weight) + "\t" + st[1] + "\t" + fix_continue_pos + "\t" + fix_one_pos + "\t" + embedding_pos + "\t" + st[total_len]
