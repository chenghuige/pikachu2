#coding=gbk

import sys

def main():
    last_feature = ""
    feature_freq_dict = {}
    for line in sys.stdin:
        line = line.strip().decode("gbk", "ignore")
        if line == "":
            continue
        line_tuple = line.split("\t")
        if len(line_tuple) < 3:
            continue
        feature = line_tuple[0]
        click = line_tuple[1]
        try:
            freq = int(line_tuple[2])
        except:
            continue
        if feature != last_feature:
            if last_feature != "":
                s = last_feature + "\t"
                for c in feature_freq_dict:
                    s += c + ":" + str(feature_freq_dict[c]) + "\t"
                print >> sys.stdout, s.strip().encode("gbk", "ignore")
            last_feature = feature
            feature_freq_dict = {}
        if click not in feature_freq_dict:
            feature_freq_dict[click] = 0
        feature_freq_dict[click] += freq

    if last_feature != "":
        s = last_feature + "\t"
        for c in feature_freq_dict:
            s += c + ":" + str(feature_freq_dict[c]) + "\t"
        print >> sys.stdout, s.strip().encode("gbk", "ignore")


if __name__ == "__main__":
    main()
