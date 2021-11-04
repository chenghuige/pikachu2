#coding=gbk

import sys
import json
import traceback
import math
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
	    if "recall" in model_name or "learn_rate" in model_name or "no_hot" in model_name or "big" in model_name or "orifeature" in model_name:
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
            
	#if "cross" in model_name:
        #    line += "\t" + "\t".join(tp_kw_feature_list)
	print >> sys.stdout, line.encode("gbk", "ignore")

if __name__ == "__main__":
    main()
