#coding=gbk

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

MAX_TOPIC_SIZE=8
MAX_TAG_SIZE=8
MAX_KW_SIZE=15
MAX_TOPIC_KW_SIZE=15
MAX_ACCOUNT_SIZE=15

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
    recent_interest = behavior.get("recent_interest","")
    long_interest = behavior.get("long_interest","")
    topic_kw_map_r = recent_interest.get("topic_kw_map", "")
    topic_kw_map_l = long_interest.get("topic_kw_map", "")
    article_info = behavior.get("article_info", "")
    cross_word = article_info.get("cross_word", "")
    for ele in cross_word:
        score = topic_kw_map_r.get(ele,0.0)
        score += 0.5 * topic_kw_map_l.get(ele,0.0)
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
    feature_q_s = ori_feature_tup[0].replace(key, key + "QUALS") + str(dur_feature.q_score) + ":\b" + ori_feature_tup[1]
    feature_p_c = ori_feature_tup[0].replace(key, key + "PCNT") + str(dur_feature.p_cnt) + ":\b" + ori_feature_tup[1]
    feature_w_c = ori_feature_tup[0].replace(key, key + "WCNT") + str(dur_feature.w_cnt) + ":\b" + ori_feature_tup[1]
    feature_p_s = ori_feature_tup[0].replace(key, key + "PORNS") + str(dur_feature.p_score) + ":\b" + ori_feature_tup[1]
    feature_dur = ori_feature_tup[0].replace(key, key + "DUR") + str(dur_feature.avg_dur) + ":\b" + ori_feature_tup[1]
    new_feature = [ele, feature_q_s, feature_p_c, feature_w_c, feature_p_s, feature_dur] 
    return new_feature

def replace_and_add_with_cross(ori_feature, key, dur_feature):
    feature_q_s = ori_feature.replace(key, key + "QUALS") + "\a" + str(dur_feature.q_score)
    feature_p_c = ori_feature.replace(key, key + "PCNT") + "\a" + str(dur_feature.p_cnt)
    feature_w_c = ori_feature.replace(key, key + "WCNT") + "\a" + str(dur_feature.w_cnt)
    feature_p_s = ori_feature.replace(key, key + "PORNS") + "\a" + str(dur_feature.p_score)
    feature_dur = ori_feature.replace(key, key + "DUR") + "\a" + str(dur_feature.avg_dur) 
    new_feature = [ori_feature, feature_q_s, feature_p_c, feature_w_c, feature_p_s, feature_dur] 
    return new_feature
 
def replace_and_add(ori_feature, key, dur_feature):
    feature_q_s = ori_feature.replace(key, key + "QUALS") + str(dur_feature.q_score)
    feature_p_c = ori_feature.replace(key, key + "PCNT") + str(dur_feature.p_cnt)
    feature_w_c = ori_feature.replace(key, key + "WCNT") + str(dur_feature.w_cnt)
    feature_p_s = ori_feature.replace(key, key + "PORNS") + str(dur_feature.p_score)
    feature_dur = ori_feature.replace(key, key + "DUR") + str(dur_feature.avg_dur) 
    new_feature = [ori_feature, feature_q_s, feature_p_c, feature_w_c, feature_p_s, feature_dur] 
    return new_feature
      
def in_set(ele, candi):
    for cur in candi:
	if cur in ele:
	    return True
    return False

#def get_current_cycle_model_profile(cycle_jason, cycle_name):

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
    click_list = list()
    show_list = list()
    dur_list = list()
    sorted_id_list = result_dict.keys()
    if len(sorted_id_list) > num:
        sorted_id_list  = sorted_id_list[:num]
    for _id in sorted_id_list:
        id_list.append(_id)
        if result_dict[_id][1] < 0:
            click_list.append("0.0")
        else:
            click_list.append(str(math.log(result_dict[_id][1]+1)/20))
        if result_dict[_id][2] < 0:
            show_list.append("0.0")
        else:
            show_list.append(str(math.log(result_dict[_id][2]+1)/20))
        if result_dict[_id][3] < 0:
            dur_list.append("0.0")
        else:
            dur_list.append(str(math.log(result_dict[_id][3]+1)/20))
    return id_list, click_list, show_list, dur_list

def padding_list(ori_list, padding_size, value):
    while len(ori_list) < padding_size:
        ori_list.append(value)
    return ori_list

def extract_sepcific_cycle_profile(cycle_name, feature_account):
    account_list, account_click, account_show, account_dur, topic_list, topic_click, topic_show, topic_dur, tag_list, tag_click, tag_show, tag_dur, kw_list, kw_click, kw_show, kw_dur, topic_kw_list, topic_kw_click, topic_kw_show, topic_kw_dur = list(), list(), list(), list(), list(), list(), list(), list(), list(), list(), list(), list(), list(), list(), list(), list(), list(), list(), list(), list()
    cycle_jason = feature_account.get(cycle_name)
    if not cycle_jason:
        topic_click = padding_list(topic_click, MAX_TOPIC_SIZE, "0.0")
        topic_show = padding_list(topic_show, MAX_TOPIC_SIZE, "0.0")
        topic_dur = padding_list(topic_show, MAX_TOPIC_SIZE, "0.0")
        tag_click = padding_list(tag_click, MAX_TAG_SIZE, "0.0")
        tag_show = padding_list(tag_show, MAX_TAG_SIZE, "0.0")
        tag_dur = padding_list(tag_dur, MAX_TAG_SIZE, "0.0")
        kw_click = padding_list(kw_click, MAX_KW_SIZE, "0.0")
        kw_show = padding_list(kw_show, MAX_KW_SIZE, "0.0")
        kw_dur = padding_list(kw_dur, MAX_KW_SIZE, "0.0")
        topic_kw_click = padding_list(topic_kw_click, MAX_TOPIC_KW_SIZE, "0.0")
        topic_kw_show = padding_list(topic_kw_show, MAX_TOPIC_KW_SIZE, "0.0")
        topic_kw_dur = padding_list(topic_kw_dur, MAX_TOPIC_KW_SIZE, "0.0")
        account_click = padding_list(account_click, MAX_ACCOUNT_SIZE, "0.0")
        account_show = padding_list(account_show, MAX_ACCOUNT_SIZE, "0.0")
        account_dur = padding_list(account_dur, MAX_ACCOUNT_SIZE, "0.0")
        return account_list, account_click, account_show, account_dur, topic_list, topic_click, topic_show, topic_dur, tag_list, tag_click, tag_show, tag_dur, kw_list, kw_click, kw_show, kw_dur, topic_kw_list, topic_kw_click, topic_kw_show, topic_kw_dur
    topic = cycle_jason.get("topic")
    tag = cycle_jason.get("tag")
    account = cycle_jason.get("account")
    kw = cycle_jason.get("kw")
    topic_kw = cycle_jason.get("topic_kw")
    #total = cycle_jason.get("total")
    topic_tuple_list = list()
    for key, val in topic.iteritems():
        topic_tuple_list.append((str(key), int(val.get("click")), int(val.get("show")), float(val.get("duration"))))
    topic_list, topic_click, topic_show, topic_dur = get_top_n(MAX_TOPIC_SIZE, topic_tuple_list)
    for i in range(len(topic_list)):
        topic_list[i] = cycle_name+"_topic\a"+str(topic_list[i])
    topic_click = padding_list(topic_click, MAX_TOPIC_SIZE, "0.0")
    topic_show = padding_list(topic_show, MAX_TOPIC_SIZE, "0.0")
    topic_dur = padding_list(topic_show, MAX_TOPIC_SIZE, "0.0")

    tag_tuple_list = list()
    for key, val in tag.iteritems():
        tag_tuple_list.append((str(key), int(val.get("click")), int(val.get("show")), float(val.get("duration"))))
    tag_list, tag_click, tag_show, tag_dur = get_top_n(MAX_TAG_SIZE, tag_tuple_list)
    for i in range(len(tag_list)):
        tag_list[i] = cycle_name+"_tag\a"+str(tag_list[i])
    tag_click = padding_list(tag_click, MAX_TAG_SIZE, "0.0")
    tag_show = padding_list(tag_show, MAX_TAG_SIZE, "0.0")
    tag_dur = padding_list(tag_dur, MAX_TAG_SIZE, "0.0")

    kw_tuple_list = list()
    for key, val in kw.iteritems():
        kw_tuple_list.append((str(key), int(val.get("click")), int(val.get("show")), float(val.get("duration"))))
    kw_list, kw_click, kw_show, kw_dur = get_top_n(MAX_KW_SIZE, kw_tuple_list)
    for i in range(len(kw_list)):
        kw_list[i] = cycle_name+"_kw\a"+str(kw_list[i])
    kw_click = padding_list(kw_click, MAX_KW_SIZE, "0.0")
    kw_show = padding_list(kw_show, MAX_KW_SIZE, "0.0")
    kw_dur = padding_list(kw_dur, MAX_KW_SIZE, "0.0")

    topic_kw_tuple_list = list()
    for key, val in topic_kw.iteritems():
        topic_kw_tuple_list.append((str(key), int(val.get("click")), int(val.get("show")), float(val.get("duration"))))
    topic_kw_list, topic_kw_click, topic_kw_show, topic_kw_dur = get_top_n(MAX_TOPIC_KW_SIZE, topic_kw_tuple_list)
    for i in range(len(topic_kw_list)):
        topic_kw_list[i] = cycle_name+"_topic_kw\a"+str(topic_kw_list[i])
    topic_kw_click = padding_list(topic_kw_click, MAX_TOPIC_KW_SIZE, "0.0")
    topic_kw_show = padding_list(topic_kw_show, MAX_TOPIC_KW_SIZE, "0.0")
    topic_kw_dur = padding_list(topic_kw_dur, MAX_TOPIC_KW_SIZE, "0.0")

    account_tuple_list = list()
    for key, val in account.iteritems():
        account_tuple_list.append((str(key), int(val.get("click")), int(val.get("show")), float(val.get("duration"))))
    account_list, account_click, account_show, account_dur = get_top_n(MAX_ACCOUNT_SIZE, account_tuple_list)
    for i in range(len(account_list)):
        account_list[i] = cycle_name+"_account\a"+str(account_list[i])
    account_click = padding_list(account_click, MAX_ACCOUNT_SIZE, "0.0")
    account_show = padding_list(account_show, MAX_ACCOUNT_SIZE, "0.0")
    account_dur = padding_list(account_dur, MAX_ACCOUNT_SIZE, "0.0")

    return account_list, account_click, account_show, account_dur, topic_list, topic_click, topic_show, topic_dur, tag_list, tag_click, tag_show, tag_dur, kw_list, kw_click, kw_show, kw_dur, topic_kw_list, topic_kw_click, topic_kw_show, topic_kw_dur

#def get_cycle_profile(behavior):
#    account_list, account_click, account_show, account_dur, topic_list, topic_click, topic_show, topic_dur, tag_list, tag_click, tag_show, tag_dur, kw_list, kw_click, kw_show, kw_dur, topic_kw_list, topic_kw_click, topic_kw_show, topic_kw_dur = extract_sepcific_cycle_profile("long_term_dict")
#    extract_sepcific_cycle_profile("last_session")
#    extract_sepcific_cycle_profile("last_refresh")
#    extract_sepcific_cycle_profile("last_day")
#    extract_sepcific_cycle_profile("last_seven_day")
#    extract_sepcific_cycle_profile("last_three_day")

def main():
    if len(sys.argv) < 2:
        print >> sys.stderr, "Usage python " + sys.argv[0] + " badmid_data_file"
        sys.exit(-1)
    badmid_data_file = sys.argv[1]
    model_name = sys.argv[2]
    filter_mid_set = load_badcase_mid(badmid_data_file)
    filter_docid_set = ()
    if len(sys.argv) >= 4:
    	baddocid_data_file = sys.argv[3]
    	filter_docid_set = load_badcase_mid(baddocid_data_file)

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
            if mid in filter_mid_set:
                continue
            if docid in filter_docid_set:
	        continue
	    behavior = json.loads(info)
            user_interest_cnt = behavior.get("interest_cnt")
            cid = behavior.get("cid","")
	    
	    if "no_hot" in model_name and cid != "1" :
	        continue    
            try:
                ty = user_interest_cnt.get("ty")
            except:
                continue
            if ty != 0:
                continue
            all_interest_cnt = get_user_interest_count(user_interest_cnt) 
            if all_interest_cnt < 30:
                continue
            cur_door = 20
            dur = behavior.get("dur")
            if dur > cur_door:
                line_tuple[0] = "1"
            else:
                line_tuple[0] = "0"
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
            feature_show_num = "SHOWNUM\a" + str(article_hot.get("show_num"))
            feature_favor_num = "FAVORNUM\a" + str(article_hot.get("favor_num"))
            #feature_read_dur = "READDUR\a" + str(article_hot.get("read_dur"))
            feature_share_num = "SHARENUM\a" + str(article_hot.get("share_num"))
            feature_account = behavior.get("feature_account")
            cycle_profile_id = ""
            cycle_profile_click = ""
            cycle_profile_show = ""
            cycle_profile_dur = ""
            if feature_account:
                account_list, account_click, account_show, account_dur, topic_list, topic_click, topic_show, topic_dur, tag_list, tag_click, tag_show, tag_dur, kw_list, kw_click, kw_show, kw_dur, topic_kw_list, topic_kw_click, topic_kw_show, topic_kw_dur = extract_sepcific_cycle_profile("long_term_dict", feature_account)
                cycle_profile_id += "\t".join(account_list) +"\t"+"\t".join(topic_list)+"\t"+"\t".join(tag_list)+"\t"+"\t".join(kw_list)+"\t"+"\t".join(topic_kw_list)
                cycle_profile_click += "cycle_profile_click\a"+",".join(account_click)+","+",".join(topic_click)+","+",".join(tag_click)+","+",".join(kw_click)+","+",".join(topic_kw_click)
                cycle_profile_show += "cycle_profile_show\a"+",".join(account_show)+","+",".join(topic_show)+","+",".join(tag_show)+","+",".join(kw_show)+","+",".join(topic_kw_show)
                cycle_profile_dur += "cycle_profile_dur\a"+",".join(account_dur)+","+",".join(topic_dur)+","+",".join(tag_dur)+","+",".join(kw_dur)+","+",".join(topic_kw_dur)

                account_list, account_click, account_show, account_dur, topic_list, topic_click, topic_show, topic_dur, tag_list, tag_click, tag_show, tag_dur, kw_list, kw_click, kw_show, kw_dur, topic_kw_list, topic_kw_click, topic_kw_show, topic_kw_dur = extract_sepcific_cycle_profile("last_session", feature_account)
                cycle_profile_id += "\t"+"\t".join(account_list) +"\t"+"\t".join(topic_list)+"\t"+"\t".join(tag_list)+"\t"+"\t".join(kw_list)+"\t"+"\t".join(topic_kw_list)
                cycle_profile_click += ","+",".join(account_click)+","+",".join(topic_click)+","+",".join(tag_click)+","+",".join(kw_click)+","+",".join(topic_kw_click)
                cycle_profile_show += ","+",".join(account_show)+","+",".join(topic_show)+","+",".join(tag_show)+","+",".join(kw_show)+","+",".join(topic_kw_show)
                cycle_profile_dur += ","+",".join(account_dur)+","+",".join(topic_dur)+","+",".join(tag_dur)+","+",".join(kw_dur)+","+",".join(topic_kw_dur)

                account_list, account_click, account_show, account_dur, topic_list, topic_click, topic_show, topic_dur, tag_list, tag_click, tag_show, tag_dur, kw_list, kw_click, kw_show, kw_dur, topic_kw_list, topic_kw_click, topic_kw_show, topic_kw_dur = extract_sepcific_cycle_profile("last_refresh", feature_account)
                cycle_profile_id += "\t"+"\t".join(account_list) +"\t"+"\t".join(topic_list)+"\t"+"\t".join(tag_list)+"\t"+"\t".join(kw_list)+"\t"+"\t".join(topic_kw_list)
                cycle_profile_click += ","+",".join(account_click)+","+",".join(topic_click)+","+",".join(tag_click)+","+",".join(kw_click)+","+",".join(topic_kw_click)
                cycle_profile_show += ","+",".join(account_show)+","+",".join(topic_show)+","+",".join(tag_show)+","+",".join(kw_show)+","+",".join(topic_kw_show)
                cycle_profile_dur += ","+",".join(account_dur)+","+",".join(topic_dur)+","+",".join(tag_dur)+","+",".join(kw_dur)+","+",".join(topic_kw_dur)

                account_list, account_click, account_show, account_dur, topic_list, topic_click, topic_show, topic_dur, tag_list, tag_click, tag_show, tag_dur, kw_list, kw_click, kw_show, kw_dur, topic_kw_list, topic_kw_click, topic_kw_show, topic_kw_dur = extract_sepcific_cycle_profile("last_day", feature_account)
                cycle_profile_id += "\t"+"\t".join(account_list) +"\t"+"\t".join(topic_list)+"\t"+"\t".join(tag_list)+"\t"+"\t".join(kw_list)+"\t"+"\t".join(topic_kw_list)
                cycle_profile_click += ","+",".join(account_click)+","+",".join(topic_click)+","+",".join(tag_click)+","+",".join(kw_click)+","+",".join(topic_kw_click)
                cycle_profile_show += ","+",".join(account_show)+","+",".join(topic_show)+","+",".join(tag_show)+","+",".join(kw_show)+","+",".join(topic_kw_show)
                cycle_profile_dur += ","+",".join(account_dur)+","+",".join(topic_dur)+","+",".join(tag_dur)+","+",".join(kw_dur)+","+",".join(topic_kw_dur)

                account_list, account_click, account_show, account_dur, topic_list, topic_click, topic_show, topic_dur, tag_list, tag_click, tag_show, tag_dur, kw_list, kw_click, kw_show, kw_dur, topic_kw_list, topic_kw_click, topic_kw_show, topic_kw_dur = extract_sepcific_cycle_profile("last_seven_day", feature_account)
                cycle_profile_id += "\t"+"\t".join(account_list) +"\t"+"\t".join(topic_list)+"\t"+"\t".join(tag_list)+"\t"+"\t".join(kw_list)+"\t"+"\t".join(topic_kw_list)
                cycle_profile_click += ","+",".join(account_click)+","+",".join(topic_click)+","+",".join(tag_click)+","+",".join(kw_click)+","+",".join(topic_kw_click)
                cycle_profile_show += ","+",".join(account_show)+","+",".join(topic_show)+","+",".join(tag_show)+","+",".join(kw_show)+","+",".join(topic_kw_show)
                cycle_profile_dur += ","+",".join(account_dur)+","+",".join(topic_dur)+","+",".join(tag_dur)+","+",".join(kw_dur)+","+",".join(topic_kw_dur)

                account_list, account_click, account_show, account_dur, topic_list, topic_click, topic_show, topic_dur, tag_list, tag_click, tag_show, tag_dur, kw_list, kw_click, kw_show, kw_dur, topic_kw_list, topic_kw_click, topic_kw_show, topic_kw_dur = extract_sepcific_cycle_profile("last_three_day", feature_account)
                cycle_profile_id += "\t"+"\t".join(account_list) +"\t"+"\t".join(topic_list)+"\t"+"\t".join(tag_list)+"\t"+"\t".join(kw_list)+"\t"+"\t".join(topic_kw_list)
                cycle_profile_click += ","+",".join(account_click)+","+",".join(topic_click)+","+",".join(tag_click)+","+",".join(kw_click)+","+",".join(topic_kw_click)
                cycle_profile_show += ","+",".join(account_show)+","+",".join(topic_show)+","+",".join(tag_show)+","+",".join(kw_show)+","+",".join(topic_kw_show)
                cycle_profile_dur += ","+",".join(account_dur)+","+",".join(topic_dur)+","+",".join(tag_dur)+","+",".join(kw_dur)+","+",".join(topic_kw_dur)
                
            
            #if "cross" in model_name: 
            #    tp_kw_feature_list = get_tp_kw_feature(behavior) 
            temp = []
            index = -1
	    lr_no_set = ["ATRD", "ATCTR", "PTCT", "ATCMT", "ATCMTRPY", "ATCMTLIKE", "ATSWCL", "ATSWSA", "ATSWFV", "ATCLSA", "ATCLFV"]
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
                        new_feature = replace_and_add_with_score(old, key, dur_feature, ele)
                        line_tuple[index] = new_feature[0]
                        temp.extend(new_feature[1: len(new_feature)])
	        if "recall" in model_name or "learn_rate" in model_name or "no_hot" in model_name or "big" in model_name or "orifeature" in model_name or "userprofile" in model_name:
                    if "CRWATID" in ele:
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
	    if "lr" in model_name:
	        line = "\t".join(temp)
	        print >> sys.stdout, line.encode("gbk", "ignore")
	        continue
            if len(temp) > 0:
                line_tuple.extend(temp)
            if profile_flag:
                line = "\t".join(line_tuple) + \
                   "\t" + feature_q_score + "\t"+ feature_w_cnt + "\t" + feature_p_cnt + "\t" + feature_p_score + "\t" + feature_avg_dur + "\t" + feature_sex + "\t" + feature_edu + "\t" + feature_age + "\t" + feature_share_num + "\t" + feature_show_num + "\t" + feature_favor_num
            else:
                line = "\t".join(line_tuple) + \
                   "\t" + feature_q_score + "\t"+ feature_w_cnt + "\t" + feature_p_cnt + "\t" + feature_p_score + "\t" + feature_avg_dur + "\t" + feature_share_num + "\t" + feature_show_num + "\t" + feature_favor_num
            if cycle_profile_id.replace("\t", ""):
                line += "\t" + cycle_profile_id + "\t" + cycle_profile_click + "\t" + cycle_profile_show + "\t" + cycle_profile_dur 
            
	    #if "cross" in model_name:
            #    line += "\t" + "\t".join(tp_kw_feature_list)
	    print >> sys.stdout, line.encode("gbk", "ignore")
        except:
            continue

if __name__ == "__main__":
    main()
