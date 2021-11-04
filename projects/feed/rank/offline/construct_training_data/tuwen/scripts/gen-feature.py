from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import json
import traceback
import math
import copy
import collections
import hashlib
import os
from config import *

MAX_TOPIC_SIZE = 8
MAX_TAG_SIZE = 8
MAX_KW_SIZE = 15
MAX_TOPIC_KW_SIZE = 15
MAX_ACCOUNT_SIZE = 15

model_name = 'sgsapp'

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

        self.ori_video_time = 0

        if MARK == VIDEO_MARK:
            video_time = behavior.get("article_info").get("video_time")
            vt = 0
            if video_time:
                if ":" in video_time:
                    for item in video_time.split(":"):
                        vt *= 60
                        vt += int(item)
            self.ori_video_time = vt
            self.video_time = num_cut_and_scale_to_int(vt, 0, 3600, 0.1)


def get_user_interest_count(user_interest_cnt):
    sum = 0
    try:
        acc = user_interest_cnt.get("acc", 0)
        top = user_interest_cnt.get("top", 0)
        tag = user_interest_cnt.get("tag", 0)
        kw = user_interest_cnt.get("kw", 0)
        sum = acc + top + tag + kw
    except:
        sum = 0
    return sum


def replace_and_add_with_score(ori_feature_tup, key, dur_feature, ele):
    feature_q_s = ori_feature_tup[0].replace(key, key + "QUALS") + str(dur_feature.q_score) + ":\b" + ori_feature_tup[1]
    feature_p_c = ori_feature_tup[0].replace(key, key + "PCNT") + str(dur_feature.p_cnt) + ":\b" + ori_feature_tup[1]
    feature_w_c = ori_feature_tup[0].replace(key, key + "WCNT") + str(dur_feature.w_cnt) + ":\b" + ori_feature_tup[1]
    feature_p_s = ori_feature_tup[0].replace(key, key + "PORNS") + str(dur_feature.p_score) + ":\b" + ori_feature_tup[1]
    feature_dur = ori_feature_tup[0].replace(key, key + "DUR") + str(dur_feature.avg_dur) + ":\b" + ori_feature_tup[1]
    new_feature = [ele, feature_q_s, feature_p_c, feature_w_c, feature_p_s, feature_dur]
    if MARK == VIDEO_MARK:
        feature_video = ori_feature_tup[0].replace(key, key + "VDTM") + str(dur_feature.video_time) + ":\b" + ori_feature_tup[1]
        new_feature.extend([feature_video])
    return new_feature


def replace_and_add_with_cross(ori_feature, key, dur_feature):
    feature_q_s = ori_feature.replace(key, key + "QUALS") + "\a" + str(dur_feature.q_score)
    # if MARK != TUWEN_MARK:
    #     # TODO now to be the same as online tuwen process
    #     feature_p_c = ori_feature.replace(key, key + "PCNT") + "\a" + str(dur_feature.p_cnt)
    #     feature_w_c = ori_feature.replace(key, key + "WCNT") + "\a" + str(dur_feature.w_cnt)
    #     feature_p_s = ori_feature.replace(key, key + "PORNS") + "\a" + str(dur_feature.p_score)
    #     feature_dur = ori_feature.replace(key, key + "DUR") + "\a" + str(dur_feature.avg_dur)
    #     new_feature = [feature_q_s,feature_p_c,feature_w_c,feature_p_s,feature_dur]
    # else:
    #   new_feature = [feature_q_s]
    new_feature = [feature_q_s]
    if MARK == VIDEO_MARK:
        feature_video = ori_feature.replace(key, key + "VDTM") + "\a" + str(dur_feature.video_time)
        new_feature.extend([feature_video])
    return new_feature

def replace_and_add(ori_feature, key, dur_feature):
    feature_q_s = ori_feature.replace(key, key + "QUALS") + str(dur_feature.q_score)
    feature_p_c = ori_feature.replace(key, key + "PCNT") + str(dur_feature.p_cnt)
    feature_w_c = ori_feature.replace(key, key + "WCNT") + str(dur_feature.w_cnt)
    feature_p_s = ori_feature.replace(key, key + "PORNS") + str(dur_feature.p_score)
    feature_dur = ori_feature.replace(key, key + "DUR") + str(dur_feature.avg_dur)
    new_feature = [ori_feature, feature_q_s, feature_p_c, feature_w_c, feature_p_s, feature_dur]
    if MARK == VIDEO_MARK:
        feature_video = ori_feature.replace(key, key + "VDTM") + "\a" + str(dur_feature.video_time)
        new_feature.extend([feature_video])
    return new_feature

def in_set(ele, candi):
    for cur in candi:
        if cur in ele:
            return True
    return False

def bubble_sort(nums_ori, dim):
    nums = copy.deepcopy(nums_ori)
    for i in range(len(nums) - 1):
        for j in range(len(nums) - i - 1):
            if nums[j][dim] < nums[j + 1][dim]:
                nums[j], nums[j + 1] = nums[j + 1], nums[j]
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

def extract_user_history(user_info, show_time):
    out = ""
    history_list = []
    tw_history_list = []
    vd_history_list = []

    for item in user_info.get("read_history", []):
        if len(history_list) >= 100:
            break
        if int(item.get("dur", "0")) < 1:
            continue
        # if int(show_time) > int(item.get("op_time")):
        #     continue
        if item.get("action", "") == "0":  # video's action is '0'
            vd_history_list.append(item["_id"])
        elif item.get("action", "") == "6": # tuwen's action is '6'
            tw_history_list.append(item["_id"])
        history_list.append("doc_%s" % item["_id"])  # read_history

    if len(history_list) > 0:
        out = "\b".join(history_list)
    else:
        out = "doc_NULL"

    tw_history_list.append("NULL")
    vd_history_list.append("NULL")
    
    return out, "\b".join(tw_history_list), "\b".join(vd_history_list)

def extract_user_interest_map(interest_dict):  # `interest_dict` aka `behavior.get("user_info")`` in this script
    long_interest_dict = {}
    recent_interest_dict = {}
    topic_dict = {}
    kw_dict = {}
    tag_dict = {}
    if "recent_interest" in interest_dict:
        recent_interest_dict = interest_dict["recent_interest"]
    if "topic_map" in recent_interest_dict:
        topic_dict = recent_interest_dict["topic_map"]
    if "kw_map" in recent_interest_dict:
        kw_dict = recent_interest_dict["kw_map"]
    if "tag_map" in recent_interest_dict:
        tag_dict = recent_interest_dict["tag_map"]

    if "long_interest" in interest_dict:
        long_interest_dict = interest_dict["long_interest"]
    if "topic_map" in long_interest_dict:
        topic_dict_tmp = long_interest_dict["topic_map"]
        for k in topic_dict_tmp.keys():
            if k in topic_dict:
                topic_dict[k] = topic_dict[k] + topic_dict_tmp[k] * 0.5
            else:
                topic_dict[k] = topic_dict_tmp[k] * 0.5
    if "kw_map" in long_interest_dict:
        kw_dict_tmp = long_interest_dict["kw_map"]
        for k in kw_dict_tmp.keys():
            if k in kw_dict:
                kw_dict[k] = kw_dict[k] + kw_dict_tmp[k] * 0.5
            else:
                kw_dict[k] = kw_dict_tmp[k] * 0.5
    if "tag_map" in long_interest_dict:
        tag_dict_tmp = long_interest_dict["tag_map"]
        for k in tag_dict_tmp.keys():
            if k in tag_dict:
                tag_dict[k] = tag_dict[k] + tag_dict_tmp[k] * 0.5
            else:
                tag_dict[k] = tag_dict_tmp[k] * 0.5

    tp_out = ""

    topic_li = []
    for k in topic_dict.keys():
        if topic_dict[k] >= 0.05:
            if "id" in k:  # skip strange keys
                continue
            topic_li.append("tp_%s" % k)  # topic
    if len(topic_li) > 0:
        tp_out += "\b".join(topic_li)
    else:
        tp_out += "tp_NULL"

    kw_out = ""
    kw_li = []
    para = len(kw_dict) / 1000.0
    # if len(kw_dict) > 60:
    #    para = 0.1
    if para > 0.1:
        para = 0.1
    if para < 0.02:
        para = 0.02
    for k in kw_dict.keys():
        if kw_dict[k] >= para:
            kw_li.append("kw_%s" % k)  # keyword & tag
    tag_li = []
    for k in tag_dict.keys():
        if tag_dict[k] >= para:
            tag_li.append("kw_%s" % k)  # keyword & tag
    if len(kw_li) > 0 or len(tag_li) > 0:
        # klist = list(set(tag_li) | set(kw_li))
        klist = tag_li
        s = set(tag_li)
        for k in kw_li:
            if k in s:
                continue
            klist.append(k)
        kw_out += "\b".join(klist)
    else:
        kw_out += "kw_NULL"

    return tp_out, kw_out


def extract_article_text_feature(behavior):
    article_info = behavior.get("article_info", {})
    text_feature_list = article_info.get("text_feature_list",[])
    
    tp_set = set()
    kw_tag_set = set()

    for item in text_feature_list:
        if item[0]==1:
            tp_set.add(item[1])
        elif item[0]==2 or item[0]==3:
            kw_tag_set.add(item[1])
    
    doc_tp_out = "art_tp_NULL"
    if len(tp_set)>0:
        doc_tp_out = "art_tp_%s" % list(tp_set)[0]

    doc_kw_out = "art_kw_NULL"
    if len(kw_tag_set)>0:
        doc_kw_out = "\b".join(map(lambda x: "art_kw_%s" % x ,kw_tag_set))

    return doc_tp_out,doc_kw_out


def extract_sepcific_cycle_profile(cycle_name, feature_account, key_name):
    account_list, topic_list, tag_list, kw_list, topic_kw_list = list(), list(), list(), list(), list()
    cycle_jason = feature_account.get(cycle_name)
    if not cycle_jason:
        return account_list, topic_list, tag_list, kw_list, topic_kw_list
    article_jason = cycle_jason.get(key_name)
    if not article_jason:
        return account_list, topic_list, tag_list, kw_list, topic_kw_list
    topic = article_jason.get("topic")
    tag = article_jason.get("tag")
    account = article_jason.get("acc")
    kw = article_jason.get("kw")
    topic_kw = article_jason.get("txk")
    # total = cycle_jason.get("total")
    topic_list = list()

    prefix = "vd_" if key_name == "video" else ""

    for val in topic:
        topic_list.append(prefix + cycle_name + "_topic\a" + str(val[0]))
        if len(topic_list) == MAX_TOPIC_SIZE:
            break

    tag_list = list()
    for val in tag:
        tag_list.append(prefix + cycle_name + "_tag\a" + str(val[0]))
        if len(tag_list) == MAX_TAG_SIZE:
            break

    kw_list = list()
    for val in kw:
        kw_list.append(prefix + cycle_name + "_kw\a" + str(val[0]))
        if len(kw_list) == MAX_KW_SIZE:
            break

    topic_kw_list = list()
    for val in topic_kw:
        topic_kw_list.append(prefix + cycle_name + "_txk\a" + str(val[0]))
        if len(topic_kw_list) == MAX_TOPIC_KW_SIZE:
            break

    account_list = list()
    for val in account:
        account_list.append(prefix + cycle_name + "_acc\a" + str(val[0]))
        if len(account_list) == MAX_ACCOUNT_SIZE:
            break

    return account_list, topic_list, tag_list, kw_list, topic_kw_list

def extract_dev_env_info(behavior):
    dev_env_features=[]
    dev_info = behavior.get("dev_info")
    env_info = behavior.get("env_info")
    if env_info:
        os = env_info.get("os","")
        network = env_info.get("network","")
        if os:
            dev_env_features.append("os\a%s" % os)
        if network:
            dev_env_features.append("network\a%s" % network)
    if dev_info:
        brand = dev_info.get("brand","").strip().lower()
        screen_width = dev_info.get("screen_width",0)
        if brand:
            dev_env_features.append("brand\a%s" % brand)
        if screen_width:
            dev_env_features.append("screen_width\a%s" % screen_width)
    return dev_env_features

def parse(line):
    try:
        # THIS is a must other wise json loads will try to use ascii to decode to unicode
        reload(sys)
        sys.setdefaultencoding('utf8')
        if line == "":
            return ''
        line_tuple = line.split("\t")
        if len(line_tuple) < 3:
            print("hehe less than 3 columns", file=sys.stderr)
            return ''
        (click, mid, docid, product, info) = line_tuple[0:5]
        behavior = json.loads(info)
        user_show = str(behavior.get("user_show", "1"))
        unlike = "0"
        if "unlike" in behavior:
            unlike = "1"
        user_interest_cnt = behavior.get("interest_cnt")
        ty = str(user_interest_cnt.get("ty", "1"))
        all_interest_cnt = str(get_user_interest_count(user_interest_cnt))
        dur = str(behavior.get("dur", "0"))
        show_time = str(behavior.get("show_time", "0"))
        impression_time = str(behavior.get("impression_time", "0"))
        
        abtestid = behavior.get("abtestid")
        ori_lr_score = str(behavior.get("ori_lr_score", -1.))
        lr_score = str(behavior.get("lr_score", -1.))
        position = str(behavior.get("position", -1))

        if ty != MARK:
            return ''

        dur_feature = durFeature(behavior)
        feature_q_score = "IQUALS\a" + str(dur_feature.q_score)
        feature_w_cnt = "IWCNT\a" + str(dur_feature.w_cnt)
        feature_p_cnt = "IPCNT\a" + str(dur_feature.p_cnt)
        feature_p_score = "IPORNS\a" + str(dur_feature.p_score)
        feature_avg_dur = "IAVGDUR\a" + str(dur_feature.avg_dur)

        if MARK == VIDEO_MARK:
            feature_video_time = "VDTM\a" + str(dur_feature.video_time)

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
        feature_favor_num = "FAVORNUM\a" + str(article_hot.get("favor_num"))
        feature_account = user_info.get("base_interest")
        cycle_profile_id = ""
        cycle_profile_click = ""
        cycle_profile_show = ""
        cycle_profile_dur = ""
        if feature_account:
            account_list, topic_list, tag_list, kw_list, topic_kw_list = extract_sepcific_cycle_profile("long_term", feature_account, "article")
            cycle_profile_id += "\t".join(account_list) + "\t" + "\t".join(topic_list) + "\t" + "\t".join(tag_list) + "\t" + "\t".join(kw_list) + "\t" + "\t".join(topic_kw_list)

            account_list, topic_list, tag_list, kw_list, topic_kw_list = extract_sepcific_cycle_profile("last_day", feature_account, "article")
            cycle_profile_id += "\t" + "\t".join(account_list) + "\t" + "\t".join(topic_list) + "\t" + "\t".join(tag_list) + "\t" + "\t".join(kw_list) + "\t" + "\t".join(topic_kw_list)

            account_list, topic_list, tag_list, kw_list, topic_kw_list = extract_sepcific_cycle_profile("last_seven_day", feature_account, "article")
            cycle_profile_id += "\t" + "\t".join(account_list) + "\t" + "\t".join(topic_list) + "\t" + "\t".join(tag_list) + "\t" + "\t".join(kw_list) + "\t" + "\t".join(topic_kw_list)

            account_list, topic_list, tag_list, kw_list, topic_kw_list = extract_sepcific_cycle_profile("last_three_day", feature_account, "article")
            cycle_profile_id += "\t" + "\t".join(account_list) + "\t" + "\t".join(topic_list) + "\t" + "\t".join(tag_list) + "\t" + "\t".join(kw_list) + "\t" + "\t".join(topic_kw_list)

            if MARK == VIDEO_MARK:
                account_list, topic_list, tag_list, kw_list, topic_kw_list = extract_sepcific_cycle_profile("long_term", feature_account, "video")
                cycle_profile_id += "\t" + "\t".join(account_list) + "\t" + "\t".join(topic_list) + "\t" + "\t".join(tag_list) + "\t" + "\t".join(kw_list) + "\t" + "\t".join(topic_kw_list)
                account_list, topic_list, tag_list, kw_list, topic_kw_list = extract_sepcific_cycle_profile("last_day", feature_account, "video")
                cycle_profile_id += "\t" + "\t".join(account_list) + "\t" + "\t".join(topic_list) + "\t" + "\t".join(tag_list) + "\t" + "\t".join(kw_list) + "\t" + "\t".join(topic_kw_list)
                account_list, topic_list, tag_list, kw_list, topic_kw_list = extract_sepcific_cycle_profile("last_seven_day", feature_account, "video")
                cycle_profile_id += "\t" + "\t".join(account_list) + "\t" + "\t".join(topic_list) + "\t" + "\t".join(tag_list) + "\t" + "\t".join(kw_list) + "\t" + "\t".join(topic_kw_list)
                account_list, topic_list, tag_list, kw_list, topic_kw_list = extract_sepcific_cycle_profile("last_three_day", feature_account, "video")
                cycle_profile_id += "\t" + "\t".join(account_list) + "\t" + "\t".join(topic_list) + "\t" + "\t".join(tag_list) + "\t" + "\t".join(kw_list) + "\t" + "\t".join(topic_kw_list)

        temp = []
        index = -1
        for ele in line_tuple:
            index += 1
            if "MITAT" in ele:
                key = "MITAT"
                old = ele.split(":\b")
                if len(old) == 2:
                    new_feature = replace_and_add_with_score(old, key, dur_feature, ele)
                    line_tuple[index] = new_feature[0]
                    temp.extend(new_feature[1: len(new_feature)])
            if "CRWATID" in ele:
                # del(line_tuple[index])
                key = "CRWATID"
                new_feature = replace_and_add_with_cross(ele, key, dur_feature)
                line_tuple[index] = new_feature[0]
                temp.extend(new_feature[1: len(new_feature)])
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

        if len(temp) > 0:
            line_tuple.extend(temp)

        if click == '1' and dur == '0':
            click = '0'
            
        doc, tw_history, vd_history = extract_user_history(user_info, show_time)
        tp, kw = extract_user_interest_map(user_info)
        doc_tp,doc_kw = extract_article_text_feature(behavior)

        # dev_env_features = extract_dev_env_info(behavior)

        tokens = "\b".join([doc, tp, kw, doc_tp, doc_kw])

        article_page_time = str(article_info.get("pt",0))
        rea= str(user_interest_cnt.get(REA, "000"))
        json_data = {
            ARTICLE_PT: article_page_time,
            TW_HISTORY: tw_history,
            VD_HISTORY: vd_history,
            REA: rea,
            # DEV_INFO: dev_env_features
        }

        json_data_str = json.dumps(json_data, ensure_ascii=True)

        # output  here, click here must be in pos 2 for dedup TODO might use json for lr_score, postion ...
        line = "\t".join([mid, docid, click, product, abtestid, show_time, unlike, all_interest_cnt, ty, dur, \
                          user_show, ori_lr_score, lr_score, position, str(dur_feature.ori_video_time), impression_time, tokens, json_data_str] + \
                         line_tuple[5:] + [feature_q_score, feature_w_cnt, feature_p_cnt, feature_p_score, feature_avg_dur])

        if profile_flag:
            line += "\t" + feature_sex + "\t" + feature_edu + \
                    "\t" + feature_age + "\t" + feature_favor_num

        if cycle_profile_id.replace("\t", ""):
            line += "\t" + cycle_profile_id

        if MARK==VIDEO_MARK:
            line += "\t" + feature_video_time

        # if dev_env_features:
        #     line += "\t"+"\t".join(dev_env_features)


        line = '\t'.join([x for x in line.split('\t') if x.strip()])
        return line
    except Exception:
        # return traceback.format_exc() + ' | encoding:' +  'None' if not sys.stdout.encoding else sys.stdout.encoding
        return ''


if __name__ == "__main__":
    ifile = sys.argv[1]
    ofile = sys.argv[2]
    model_name = sys.argv[3]

    compress = None if not COMPRESS else 'com.hadoop.compression.lzo.LzopCodec'

    from pyspark import SparkConf, SparkContext

    conf = SparkConf() \
        .set("spark.ui.showConsoleProgress", "true") \
        .set('spark.hive.mapred.supports.subdirectories', 'true') \
        .set('spark.hadoop.mapreduce.input.filsleinputformat.input.dir.recursive', 'true') \
        .set('spark.executor.memory', '3g') \
        .set("spark.default.parallelism", '500') \
        .set('spark.dynamicAllocation.enabled', 'true') \
        .set('spark.port.maxRetries', '100') 

    sc = SparkContext(conf=conf)
    # sc.setLogLevel("WARN")
    d = sc.textFile(ifile)
    d = d.map(parse).filter(lambda x: x != '')
    # print(d.take(10))
    # exit(0)
    # d.cache()

    # print('Num instances:', d.count())
    # d = d.map(parse).take(3)
    # print(d)
    # if DEBUG:
    #   print('Num instances after gen_feature: ', d.count()) 
    try:
        d.saveAsTextFile(ofile, compressionCodecClass=COMPRESS)
    except Exception as e:
        pass
