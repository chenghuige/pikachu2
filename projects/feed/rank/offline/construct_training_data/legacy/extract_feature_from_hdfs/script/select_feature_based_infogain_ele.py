# coding=gbk
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import math
import datetime
model_name = ""


def collect_feature_freq_from_feature_list():
    feature_freq = {}
    entropy = 0
    classify_freq = {}
    for line in sys.stdin:
        line = line.strip().decode("gbk", "ignore")
        if line == "":
            continue
        line_tuple = line.split("\t")
        if len(line_tuple) < 2:
            continue
        feature = line_tuple[0]
        if feature == "TOTAL_SAMPLES":
            classify_list = line_tuple[1:]
            for item in classify_list:
                item_tuple = item.split(":")
                if len(item_tuple) != 2:
                    continue
                (classify, freq) = item_tuple
                try:
                    freq = int(freq)
                except:
                    continue
                classify_freq[classify] = freq
            continue
        feature_freq.setdefault(feature, {})
        classify_list = line_tuple[1:]
        for item in classify_list:
            item_tuple = item.split(":")
            if len(item_tuple) != 2:
                continue
            (classify, freq) = item_tuple
            try:
                freq = int(freq)
            except:
                continue
            feature_freq[feature][classify] = freq
    return feature_freq, classify_freq


def calculate_entropy(freq_list):
    s = sum(freq_list)
    if s == 0:
        return 0.0
    entropy = 0.0
    for v in freq_list:
        p = v/float(s)
        if p != 0:
            entropy += -p*math.log(p, 2)
    return entropy


def calculate_info_gain(feature_freq_in_classify, classify_freq, entropy):
    appear_list = []
    disappear_list = []
    sum = 0
    appear = 0
    disappear = 0
    for classify in classify_freq:
        sum += classify_freq[classify]
        appear += feature_freq_in_classify.get(classify, 0)
        disappear += classify_freq[classify] - \
            feature_freq_in_classify.get(classify, 0)
        appear_list.append(feature_freq_in_classify.get(classify, 0))
        disappear_list.append(
            classify_freq[classify]-feature_freq_in_classify.get(classify, 0))
    return entropy - ((appear/(sum+0.0))*calculate_entropy(appear_list) + (disappear/(sum+0.0))*calculate_entropy(disappear_list))


def main():
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 现在
    #print("start", nowTime)
    feature_freq, classify_freq = collect_feature_freq_from_feature_list()
    entropy = calculate_entropy(classify_freq.values())
    for feature in feature_freq:
        ele_freq = feature_freq[feature]
        gain = calculate_info_gain(ele_freq, classify_freq, entropy)
        print(feature.encode("gbk", "ignore") + "\t" + str(sum(ele_freq.values())) + "\t" + str(gain))


if __name__ == "__main__":
    main()
