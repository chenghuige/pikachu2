# coding=gbk

import sys


def main():
    if len(sys.argv) < 3:
        print >> sys.stderr, "Usage python " + \
            sys.argv[0] + " feature_infogain index_file"
        sys.exit(-1)
    feature_infogain_file = sys.argv[1]
    feature_indx_file = sys.argv[2]
    feature_list = []
    with open(feature_infogain_file, "r") as fp:
        for line in fp:
            line = line.strip().decode("gbk", "ignore")
            if line == "":
                continue
            line_tuple = line.split("\t")
            if len(line_tuple) < 3:
                continue
            (feature, freq, info_gain) = line_tuple[:3]
            if int(freq) < 10:
                continue
            if float(info_gain) < 0.00000001:
                continue
            feature_list.append(feature)
    feature_list.sort()
    index_fd = open(feature_indx_file, "w")
    index = 0
    last_category = ""
    last_index = 0

    for feature in feature_list:
        cur_category = feature.split('\a')[0]
        index += 1
        s = feature + "\t" + str(index)
        index_fd.write("%s\n" % s.encode("gbk", "ignore"))
    index_fd.close()


if __name__ == "__main__":
    main()
