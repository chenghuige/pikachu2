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

MAX_TOPIC_SIZE = 8
MAX_TAG_SIZE = 8
MAX_KW_SIZE = 15
MAX_TOPIC_KW_SIZE = 15
MAX_ACCOUNT_SIZE = 15


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


def get_tp_kw_feature(behavior):
    feature_list = []
    recent_interest = behavior.get("recent_interest", "")
    long_interest = behavior.get("long_interest", "")
    topic_kw_map_r = recent_interest.get("topic_kw_map", "")
    topic_kw_map_l = long_interest.get("topic_kw_map", "")
    article_info = behavior.get("article_info", "")
    cross_word = article_info.get("cross_word", "")
    for ele in cross_word:
        score = topic_kw_map_r.get(ele, 0.0)
        score += 0.5 * topic_kw_map_l.get(ele, 0.0)
        if score > 0.005:
            feature = "MTPKW" + "\a" + ele + ":\b" + str(score)
            feature_list.append(feature)

    return feature_list


def num_cut_and_scale_to_int(num, small, big, scale):
    if num > big:
        num = big
    if num < small:
        num = small
    num = int(num * scale)
    return num


class durFeature:
    def __init__(self, behavior):
        p_score = behavior.get("p_score")
        q_score = behavior.get("q_score")
        p_cnt = behavior.get("i_len")
        w_cnt = behavior.get("c_len")
        avg_dur = behavior.get("avg_dur")

        self.p_score = num_cut_and_scale_to_int(p_score, 0, 1, 100)
        self.q_score = num_cut_and_scale_to_int(q_score, 0, 1, 100)
        self.p_cnt = num_cut_and_scale_to_int(p_cnt, 0, 50, 1)
        self.w_cnt = num_cut_and_scale_to_int(w_cnt, 0, 5000, 0.01)
        self.avg_dur = num_cut_and_scale_to_int(avg_dur, 0, 300, 1)


def get_user_interest_count(user_interest_cnt):
    sum = 0
    try:
        acc = user_interest_cnt.get("acc")
        top = user_interest_cnt.get("top")
        tag = user_interest_cnt.get("tag")
        kw = user_interest_cnt.get("kw")
        sum = acc + top + tag + kw
    except:
        sum = 0
    return sum


def replace_and_add_with_score(ori_feature_tup, key, dur_feature, ele):
    feature_q_s = ori_feature_tup[0].replace(
        key, key + "QUALS") + str(dur_feature.q_score) + ":\b" + ori_feature_tup[1]
    feature_p_c = ori_feature_tup[0].replace(
        key, key + "PCNT") + str(dur_feature.p_cnt) + ":\b" + ori_feature_tup[1]
    feature_w_c = ori_feature_tup[0].replace(
        key, key + "WCNT") + str(dur_feature.w_cnt) + ":\b" + ori_feature_tup[1]
    feature_p_s = ori_feature_tup[0].replace(
        key, key + "PORNS") + str(dur_feature.p_score) + ":\b" + ori_feature_tup[1]
    feature_dur = ori_feature_tup[0].replace(
        key, key + "DUR") + str(dur_feature.avg_dur) + ":\b" + ori_feature_tup[1]
    new_feature = [ele, feature_q_s, feature_p_c,
                   feature_w_c, feature_p_s, feature_dur]
    return new_feature


def replace_and_add_with_cross(ori_feature, key, dur_feature):
    feature_q_s = ori_feature.replace(
        key, key + "QUALS") + "\a" + str(dur_feature.q_score)
    #feature_p_c = ori_feature.replace(key, key + "PCNT") + "\a" + str(dur_feature.p_cnt)
    #feature_w_c = ori_feature.replace(key, key + "WCNT") + "\a" + str(dur_feature.w_cnt)
    #feature_p_s = ori_feature.replace(key, key + "PORNS") + "\a" + str(dur_feature.p_score)
    #feature_dur = ori_feature.replace(key, key + "DUR") + "\a" + str(dur_feature.avg_dur)
    #new_feature = [feature_q_s, feature_p_c, feature_w_c, feature_p_s, feature_dur]
    new_feature = [feature_q_s]
    return new_feature


def replace_and_add(ori_feature, key, dur_feature):
    feature_q_s = ori_feature.replace(
        key, key + "QUALS") + str(dur_feature.q_score)
    feature_p_c = ori_feature.replace(
        key, key + "PCNT") + str(dur_feature.p_cnt)
    feature_w_c = ori_feature.replace(
        key, key + "WCNT") + str(dur_feature.w_cnt)
    feature_p_s = ori_feature.replace(
        key, key + "PORNS") + str(dur_feature.p_score)
    feature_dur = ori_feature.replace(
        key, key + "DUR") + str(dur_feature.avg_dur)
    new_feature = [ori_feature, feature_q_s, feature_p_c,
                   feature_w_c, feature_p_s, feature_dur]
    return new_feature


def in_set(ele, candi):
    for cur in candi:
        if cur in ele:
            return True
    return False

# def get_current_cycle_model_profile(cycle_jason, cycle_name):


def bubble_sort(nums_ori, dim):
    nums = copy.deepcopy(nums_ori)
    for i in range(len(nums) - 1):
        for j in range(len(nums)-i-1):
            if nums[j][dim] < nums[j+1][dim]:
                nums[j], nums[j+1] = nums[j+1], nums[j]
    return nums


def get_top_n(num, tuple_list):
    click_tuple_list = bubble_sort(tuple_list, 1)
    show_tuple_list = bubble_sort(tuple_list, 2)
    dur_tuple_list = bubble_sort(tuple_list, 3)
    index = 0
    result_dict = collections.OrderedDict()
    while index < len(click_tuple_list) and index < num:
        result_dict[click_tuple_list[index][0]] = click_tuple_list[index]
        result_dict[show_tuple_list[index][0]] = show_tuple_list[index]
        result_dict[dur_tuple_list[index][0]] = dur_tuple_list[index]
        index += 1

    id_list = list()
    sorted_id_list = result_dict.keys()
    if len(sorted_id_list) > num:
        sorted_id_list = sorted_id_list[:num]
    for _id in sorted_id_list:
        id_list.append(_id)
    return id_list


def padding_list(ori_list, padding_size, value):
    while len(ori_list) < padding_size:
        ori_list.append(value)
    return ori_list


def extract_sepcific_cycle_profile(cycle_name, feature_account):
    account_list, topic_list, tag_list, kw_list, topic_kw_list = list(
    ), list(), list(), list(), list()
    cycle_jason = feature_account.get(cycle_name)
    if not cycle_jason:
        return account_list, topic_list, tag_list, kw_list, topic_kw_list
    article_jason = cycle_jason.get("article")
    if not article_jason:
        return account_list, topic_list, tag_list, kw_list, topic_kw_list
    topic = article_jason.get("topic")
    tag = article_jason.get("tag")
    account = article_jason.get("acc")
    kw = article_jason.get("kw")
    topic_kw = article_jason.get("txk")
    #total = cycle_jason.get("total")
    topic_list = list()
    for val in topic:
        topic_list.append(cycle_name+"_topic\a"+str(val[0]))
        if len(topic_list) == MAX_TOPIC_SIZE:
            break

    tag_list = list()
    for val in tag:
        tag_list.append(cycle_name+"_tag\a"+str(val[0]))
        if len(tag_list) == MAX_TAG_SIZE:
            break

    kw_list = list()
    for val in kw:
        kw_list.append(cycle_name+"_kw\a"+str(val[0]))
        if len(kw_list) == MAX_KW_SIZE:
            break

    topic_kw_list = list()
    for val in topic_kw:
        topic_kw_list.append(cycle_name+"_txk\a"+str(val[0]))
        if len(topic_kw_list) == MAX_TOPIC_KW_SIZE:
            break

    account_list = list()
    for val in account:
        account_list.append(cycle_name+"_acc\a"+str(val[0]))
        if len(account_list) == MAX_ACCOUNT_SIZE:
            break

    return account_list, topic_list, tag_list, kw_list, topic_kw_list


def main():
    # if len(sys.argv) < 2:
    #    print >> sys.stderr, "Usage python " + sys.argv[0] + " badmid_data_file"
    #    sys.exit(-1)
    model_name = sys.argv[1]

    for line in sys.stdin:
        try:
            line = line.strip().decode("gbk", "ignore")
            if line == "":
                continue
            line_tuple = line.split("\t")
            if len(line_tuple) < 3:
                print >> sys.stderr, "hehe less than 3 columns"
                continue
            (_, mid, docid, _, info) = line_tuple[0:5]
            behavior = json.loads(info)
            unlike = "0"
            if "unlike"in behavior:
                unlike = "1"
            user_interest_cnt = behavior.get("interest_cnt")
            try:
                ty = str(user_interest_cnt.get("ty"))
            except:
                ty = "1"
            all_interest_cnt = str(get_user_interest_count(user_interest_cnt))
            try:
                dur = str(behavior.get("dur"))
            except:
                dur = "None"
            try:
                show_time = str(behavior.get("show_time"))
            except:
                show_time = "0"
            abtestid = behavior.get("abtestid")

            dur_feature = durFeature(behavior)
            feature_q_score = "IQUALS\a" + str(dur_feature.q_score)
            feature_w_cnt = "IWCNT\a" + str(dur_feature.w_cnt)
            feature_p_cnt = "IPCNT\a" + str(dur_feature.p_cnt)
            feature_p_score = "IPORNS\a" + str(dur_feature.p_score)
            feature_avg_dur = "IAVGDUR\a" + str(dur_feature.avg_dur)

            user_info = behavior.get("user_info")
            qq_profile = user_info.get("qq_profile")
            profile_flag = 0
            if qq_profile:
                feature_sex = "SEX\a" + str(qq_profile.get("sex"))
                feature_edu = "EDU\a" + str(qq_profile.get("edu"))
                feature_age = "AGE\a" + str(qq_profile.get("age"))
                profile_flag = 1
            article_info = behavior.get("article_info")
            article_hot = article_info.get("article_hot")
            #feature_common_num = "COMNUM\a" + str(article_hot.get("commment_num"))
            #feature_read_num = "READNUM\a" + str(article_hot.get("read_num"))
            #feature_show_num = "SHOWNUM\a" + str(article_hot.get("show_num"))
            feature_favor_num = "FAVORNUM\a" + \
                str(article_hot.get("favor_num"))
            ##feature_read_dur = "READDUR\a" + str(article_hot.get("read_dur"))
            #feature_share_num = "SHARENUM\a" + str(article_hot.get("share_num"))
            feature_account = user_info.get("base_interest")
            cycle_profile_id = ""
            cycle_profile_click = ""
            cycle_profile_show = ""
            cycle_profile_dur = ""
            if feature_account:
                account_list, topic_list, tag_list, kw_list, topic_kw_list = extract_sepcific_cycle_profile(
                    "long_term", feature_account)
                cycle_profile_id += "\t".join(account_list) + "\t"+"\t".join(topic_list)+"\t"+"\t".join(
                    tag_list)+"\t"+"\t".join(kw_list)+"\t"+"\t".join(topic_kw_list)

                #account_list, topic_list, tag_list, kw_list, topic_kw_list = extract_sepcific_cycle_profile("last_session", feature_account)
                #cycle_profile_id += "\t"+"\t".join(account_list) +"\t"+"\t".join(topic_list)+"\t"+"\t".join(tag_list)+"\t"+"\t".join(kw_list)+"\t"+"\t".join(topic_kw_list)

                #account_list, topic_list, tag_list, kw_list, topic_kw_list = extract_sepcific_cycle_profile("last_refresh", feature_account)
                #cycle_profile_id += "\t"+"\t".join(account_list) +"\t"+"\t".join(topic_list)+"\t"+"\t".join(tag_list)+"\t"+"\t".join(kw_list)+"\t"+"\t".join(topic_kw_list)

                account_list, topic_list, tag_list, kw_list, topic_kw_list = extract_sepcific_cycle_profile(
                    "last_day", feature_account)
                cycle_profile_id += "\t"+"\t".join(account_list) + "\t"+"\t".join(
                    topic_list)+"\t"+"\t".join(tag_list)+"\t"+"\t".join(kw_list)+"\t"+"\t".join(topic_kw_list)

                account_list, topic_list, tag_list, kw_list, topic_kw_list = extract_sepcific_cycle_profile(
                    "last_seven_day", feature_account)
                cycle_profile_id += "\t"+"\t".join(account_list) + "\t"+"\t".join(
                    topic_list)+"\t"+"\t".join(tag_list)+"\t"+"\t".join(kw_list)+"\t"+"\t".join(topic_kw_list)

                account_list, topic_list, tag_list, kw_list, topic_kw_list = extract_sepcific_cycle_profile(
                    "last_three_day", feature_account)
                cycle_profile_id += "\t"+"\t".join(account_list) + "\t"+"\t".join(
                    topic_list)+"\t"+"\t".join(tag_list)+"\t"+"\t".join(kw_list)+"\t"+"\t".join(topic_kw_list)

            # if "cross" in model_name:
            #    tp_kw_feature_list = get_tp_kw_feature(behavior)
            temp = []
            index = -1
            lr_no_set = ["ATRD", "ATCTR", "PTCT", "ATCMT", "ATCMTRPY",
                         "ATCMTLIKE", "ATSWCL", "ATSWSA", "ATSWFV", "ATCLSA", "ATCLFV"]
            for ele in line_tuple:
                index += 1
                if "lr" in model_name:
                    if ("CRWATID" in ele or "ICBRW" in ele or "ICFRW" in ele or in_set(ele, lr_no_set)):
                        continue
                    temp.append(ele)
                    continue
                if "MITAT" in ele:
                    key = "MITAT"
                    old = ele.split(":\b")
                    if len(old) == 2:
                        new_feature = replace_and_add_with_score(
                            old, key, dur_feature, ele)
                        line_tuple[index] = new_feature[0]
                        temp.extend(new_feature[1: len(new_feature)])
                if "recall" in model_name or "learn_rate" in model_name or "no_hot" in model_name or "big" in model_name or "hour" in model_name or "userprofile" in model_name:
                    if "CRWATID" in ele:
                        # del(line_tuple[index])
                        key = "CRWATID"
                        new_feature = replace_and_add_with_cross(
                            ele, key, dur_feature)
                        line_tuple[index] = new_feature[0]
                        #temp.extend(new_feature[1: len(new_feature)])
                    elif "ICBRW" in ele:
                        key = "ICBRW"
                        new_feature = replace_and_add(ele, key, dur_feature)
                        line_tuple[index] = new_feature[0]
                        temp.extend(new_feature[1: len(new_feature)])
                    elif "ICFRW" in ele:
                        key = "ICFRW"
                        new_feature = replace_and_add(ele, key, dur_feature)
                        line_tuple[index] = new_feature[0]
                        temp.extend(new_feature[1: len(new_feature)])
            if "lr" in model_name:
                line = "\t".join(temp)
                print >> sys.stdout, line.encode("gbk", "ignore")
                continue
            if len(temp) > 0:
                line_tuple.extend(temp)

            line = "\t".join(line_tuple[0:4]) + "\t" + abtestid + "\t" + unlike + "\t" + all_interest_cnt + "\t" + ty + "\t" + dur + "\t" + show_time + "\t" + "\t".join(
                line_tuple[5:]) + "\t" + feature_q_score + "\t" + feature_w_cnt + "\t" + feature_p_cnt + "\t" + feature_p_score + "\t" + feature_avg_dur
            if profile_flag:
                line += "\t" + feature_sex + "\t" + feature_edu + \
                    "\t" + feature_age + "\t" + feature_favor_num
            if cycle_profile_id.replace("\t", ""):
                line += "\t" + cycle_profile_id

            # if "cross" in model_name:
            #    line += "\t" + "\t".join(tp_kw_feature_list)
            print >> sys.stdout, line.encode("gbk", "ignore")
        except:
            continue


if __name__ == "__main__":
    main()
