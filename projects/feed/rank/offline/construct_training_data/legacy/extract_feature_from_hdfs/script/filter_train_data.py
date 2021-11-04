# coding=gbk

import sys
import json
import traceback
import math
import copy
import collections
import hashlib
import os
reload(sys)
sys.setdefaultencoding('gbk')


def load_badcase_mid(filter_mid_file):
    filter_mid_set = set()
    with open(filter_mid_file, "r") as fp:
        for line in fp:
            line = line.strip().decode("gbk", "ignore")
            line_tuple = line.split("\t")
            if len(line_tuple) < 1:
                continue
            mid = line_tuple[0]
            filter_mid_set.add(mid)
    return filter_mid_set

NUM_PRES = 10
def main():
    if len(sys.argv) < 2:
        print >> sys.stderr, "Usage python " + \
            sys.argv[0] + " badmid_data_file"
        sys.exit(-1)
    badmid_data_file = sys.argv[1]
    model_name = sys.argv[2]
    filter_mid_set = load_badcase_mid(badmid_data_file)

    for line in sys.stdin:
        try:
            line = line.strip().decode("gbk", "ignore")
            if line == "":
                continue
            line_tuple = line.split("\t")
            (click, mid, docid, product, abtestid, unlike, all_interest_cnt,
             ty, dur, show_time) = line_tuple[:NUM_PRES]
            if mid in filter_mid_set:
                continue
            # if unlike == "1":
            #     continue
            if ty != "0":
                continue
            # if int(all_interest_cnt) < 30:
            #     continue
            # cur_door = 20
            cur_door = 0
            if dur == "None" and click == "1":
                continue
            if dur != "None":
                if int(dur) > cur_door:
                    line_tuple[0] = "1"
                elif int(dur) < cur_door:
                    line_tuple[0] = "0"
            else:
                line_tuple[0] = "0"

            line = "\t".join(line_tuple)
            print >> sys.stdout, line.encode("gbk", "ignore")
        except:
            continue


if __name__ == "__main__":
    main()
