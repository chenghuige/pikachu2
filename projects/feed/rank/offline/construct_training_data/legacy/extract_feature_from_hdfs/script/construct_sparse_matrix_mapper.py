# coding=gbk
__author__ = "ouyuanbiao"

import sys
import json


def load_meta(feature_meta_file):
    with open(feature_meta_file, "r") as fp:
        for line in fp:
            line = line.strip().decode("gbk", "ignore")
            if line == "":
                continue
            line_tuple = line.split("-")
            if len(line_tuple) != 4:
                continue
            base_end_index = int(line_tuple[0])
            doc_end_index = int(line_tuple[1]) + base_end_index
            acc_end_index = int(line_tuple[2]) + doc_end_index
            top_end_index = int(line_tuple[3]) + acc_end_index
    return (base_end_index, doc_end_index, acc_end_index, top_end_index)


def modify_emb_index(cur_index, end_index):
    if cur_index == 0:
        cur_index = end_index
    else:
        cur_index += 1
    return cur_index


def load_feature_index(feature_index_file):
    feature_index_dict = {}
    with open(feature_index_file, "r") as fp:
        for line in fp:
            line = line.strip().decode("gbk", "ignore")
            if line == "":
                continue
            line_tuple = line.split("\t")
            if len(line_tuple) != 2:
                continue
            feature_index_dict[line_tuple[0]] = int(line_tuple[1])
    return feature_index_dict

SPLIT_IDX = 9

def main():
    model_name = ""
    if len(sys.argv) < 2:
        print >> sys.stderr, "Usage python " + \
            sys.argv[0] + " badmid_data_file"
        sys.exit(-1)
    if len(sys.argv) >= 3:
        model_name = sys.argv[2]
    split_idx = SPLIT_IDX
    feature_index_file = sys.argv[1]
    feature_index_dict = load_feature_index(feature_index_file)
    # for din model
    feature_meta_file = ""
    (base_end_index, doc_end_index, acc_end_index, top_end_index) = (0, 0, 0, 0)
    for line in sys.stdin:
        line = line.strip().decode("gbk", "ignore")
        if line == "":
            continue
        line_tuple = line.split("\t")
        if len(line_tuple) <= split_idx:
            continue
        split_idx_feature = split_idx + 1
        #right = True
        multi_sample = 1
        label_inv = False
        dur = "0"
        if len(line_tuple) < split_idx_feature:
            continue
        (click, uid, doc_id, product, abtest_id, unlike, all_interest_cnt,
        ty, dur, show_time) = line_tuple[:split_idx_feature]
        if label_inv:
            click = str(1 - int(click))
        feature_list = line_tuple[split_idx_feature:]
        index_list = []
        relationship_list = []
        for feature in feature_list:
            feature_tuple = feature.split(":\b")
            feature = feature_tuple[0]
            if (feature not in feature_index_dict and "din" not in model_name) or ("din" in model_name and "EM_MEM" not in feature and feature not in feature_index_dict):
                continue
            if "din" in model_name and "EM_MEM" in feature:
                name = feature.split(":\f")
                doc_id = feature_index_dict.get(name[1], 0)
                account_id = feature_index_dict.get(name[2], 0)
                topic_id = feature_index_dict.get(name[3], 0)
                if doc_id != ""  or account_id != "" or topic_id != "":
                    relationship_list.append(str(modify_emb_index(doc_id, base_end_index))+"-" +
                                                str(modify_emb_index(account_id, doc_end_index) + 1) + "-" + \
                        str(modify_emb_index(topic_id, acc_end_index) + 2))
            else:
                index = feature_index_dict[feature]
                value = feature_tuple[1] if len(feature_tuple) > 1 else 1
                item = (index, value)
                index_list.append(item)

        index_list.sort(key=lambda k: k[0])
        line_tuple[0] = click
        s = "\t".join(line_tuple[:split_idx-1])
        if click == "0":
            dur = "0"
        if dur == "None":
            dur = "0"
        # run here
        if model_name != "newmse_dduurr_realtime" and ("sgsapp" in model_name):
            s = '\t'.join([uid, doc_id, show_time, abtest_id, unlike, all_interest_cnt, click, dur])
        last_index = -1
        for item in index_list:
            (index, value) = item
            if index == last_index:
                continue
            last_index = index
            s += "\t" + str(index) + ":" + str(value)
        while (multi_sample):
            print >> sys.stdout, s.strip().encode("gbk", "ignore")
            multi_sample -= 1


if __name__ == "__main__":
    main()
