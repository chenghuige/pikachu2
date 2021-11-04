# coding=gbk

import sys
import math

model_name = ""


def collect_feature_freq_from_feature_list():
    feature_freq = {}
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


def rerank_feature_by_info_gain(feature_freq, classify_freq):
    entropy = calculate_entropy(classify_freq.values())
    feature_list = [(calculate_info_gain(feature_freq[feature], classify_freq, entropy), sum(
        feature_freq[feature].values()), feature) for feature in feature_freq]
    # feature_list.sort(reverse=True)
    return feature_list


def main():
    (feature_freq, classify_freq) = collect_feature_freq_from_feature_list()
    print >> sys.stderr, len(feature_freq)
    print >> sys.stderr, len(classify_freq)
    feature_list = rerank_feature_by_info_gain(feature_freq, classify_freq)
    for item in feature_list:
        (info_gain, freq, feature) = item
        s = feature + "\t" + str(freq) + "\t" + str(info_gain)
        print >> sys.stdout, s.encode("gbk", "ignore")


if __name__ == "__main__":
    main()
