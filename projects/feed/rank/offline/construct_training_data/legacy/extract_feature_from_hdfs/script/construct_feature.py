#coding=gbk

import sys
import json
import traceback
import math


SEX_SET = set((u"男性", u"女性"))
AGE_SET = set(("0_18", "19_23", "24_30", "31_40", "41_50", "51_999"))
EDU_SET = set((u"小学", u"初中", u"高中", u"大学生", u"硕士", u"博士"))

FEATURE_SET = set()

acc_click_interval = [0, 100, 1000]
acc_show_interval = [0, 100, 1000]
acc_favor_interval = [0, 100, 1000]
acc_share_interval = [0, 100, 1000]

pagetime_clicktime_interval = [3, 6, 12, 24, 36, 48, 60, 72]
ctr_score_interval = [0.01,0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

art_app_share_interval = [2,4,8,16,32,64,128,256,512,1024,2048]
art_app_show_interval = [2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,10000] 
art_app_read_interval = [2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,10000] 
art_app_readduration_interval = [2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,10000] 
art_app_favor_interval = [2,4,8,16,32,64,128,256,512,1024,2048] 
art_app_comment_interval = [2,4,8,16,32,64,128,256,512,1024,2048] 
art_app_cmt_reply_interval = [2,4,8,16,32,64,128,256,512,1024,2048] 
art_app_cmt_like_interval = [2,4,8,16,32,64,128,256,512,1024,2048] 
art_news_comment_interval = [7,16,24,37,52,69,93,120,156,207,279,263,510,766,1192,1778,2889,5315,13280,208506]
art_news_participant_interval = [12,24,39,60,83,112,145,190,283,454,659,1112,1649,2568,3949,6055,12349,24525,55210,294619,3193264]
art_sogourank_pv_interval = [200,400,600,800]

recall_word_interval = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

interest_match_interval = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

def load_interval(interval_file):
    global acc_click_interval, acc_show_interval, acc_favor_interval, acc_share_interval, pagetime_clicktime_interval, art_app_share_interval, art_app_show_interval, art_app_read_interval, art_app_readduration_interval, art_app_favor_interval, art_news_comment_interval, art_news_participant_interval, art_sogourank_pv_interval
    json_str = ""
    with open(interval_file, "r") as fp:
        for line in fp:
            json_str += line.strip().decode("gbk", "ignore")
    try:
        interval_dict = json.loads(json_str);
    except:
        return
    acc_click_interval = interval_dict["acc_click_interval"]
    acc_show_interval = interval_dict["acc_show_interval"]
    acc_favor_interval = interval_dict["acc_share_interval"]
    acc_share_interval = interval_dict["acc_share_interval"]
    pagetime_clicktime_interval = interval_dict["pagetime_clicktime_interval"]
    art_app_share_interval = interval_dict["art_app_share_interval"]
    art_app_show_interval = interval_dict["art_app_show_interval"]
    art_app_read_interval = interval_dict["art_app_read_interval"]
    art_app_readduration_interval = interval_dict["art_app_readduration_interval"]
    art_app_favor_interval = interval_dict["art_app_favor_interval"]
    art_news_comment_interval = interval_dict["art_news_comment_interval"]
    art_news_participant_interval = interval_dict["art_news_participant_interval"]
    art_sogourank_pv_interval = interval_dict["art_sogourank_pv_interval"]

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

def construct_cross_feature(prefix, separator, feature_list):
    return "C" + prefix + separator + separator.join(feature_list)

def construct_cross_feature_with_value(prefix, separator, feature_list, value):
    return "C" + prefix + separator + separator.join(feature_list) + ":\b" + str(value)

def construct_match_feature(prefix, separator, feature):
    return "M" + prefix + separator + feature

def construct_match_feature_with_value(prefix, separator, feature, value):
    return "M" + prefix + separator + feature + ":\b" + str(value)

def construct_match_discrete_feature(prefix, separator, feature, value, interval):
    level = get_discrete_level(value, interval)
    return "M" + prefix + separator + feature + separator + str(level)

def construct_dismatch_feature(prefix, separator, feature):
    return "N" + prefix + separator + feature

def construct_independ_feature(prefix, separator, feature):
    return "I" + prefix + separator + feature

def get_discrete_level(value, interval):
    level = 0
    for i in range(len(interval)):
        if value <= interval[i]:
            break
        level += 1
    if i == len(interval):
        level += 1
    return level

def construct_discrete_feature_with_original(prefix, separator, feature, value, interval):
    level = get_discrete_level(value, interval)
    return "D" + prefix + separator + feature + separator + str(level)

def construct_discrete_feature(prefix, separator, value, interval):
    level = get_discrete_level(value, interval)
    return "D" + prefix + separator + str(level)

def construct_cross_discrete_feature(prefix, separator, value1, interval1, value2, interval2):
    level1 = get_discrete_level(value1, interval1)
    level2 = get_discrete_level(value2, interval2)
    return "X" + prefix + separator + str(level1) + separator + str(level2)

def merge_long_recent_interest(interest):
    long_interest = interest.get("long_interest",{})
    recent_interest = interest.get("recent_interest",{})
    coeff = 0.5
    merge_interest = {"keyword":{}, "tag":{}, "account":{}, "topic":{}, "lda":{}}
    item = long_interest
    cnt = 0
    try:
        topic_map = item.get("topic_map",{})
        for topic in topic_map:
            if topic not in merge_interest["topic"]:
                merge_interest["topic"][topic] = 0.0
            score = topic_map[topic]
	    cnt += 1
            merge_interest["topic"][topic] += coeff*score
        kw_map = item.get("kw_map",{})
        for kw in kw_map:
            if kw not in merge_interest["keyword"]:
                merge_interest["keyword"][kw] = 0.0
            score = kw_map[kw]
	    cnt += 1
            merge_interest["keyword"][kw] += coeff*score
        tag_map = item.get("tag_map",{})
        for tag in tag_map:
            if tag not in merge_interest["tag"]:
                merge_interest["tag"][tag] = 0.0
            score = tag_map[tag]
	    cnt += 1
            merge_interest["tag"][tag] += coeff*score 
        acc_map = item.get("account_map",{})
        for acc in acc_map:
            if acc not in merge_interest["account"]:
                merge_interest["account"][acc] = 0.0
            score = acc_map[acc]
	    cnt += 1
            merge_interest["account"][acc] += coeff*score
      	lda_map = item.get("lda_map",{})
        for lda_ori in lda_map:
	    lda = int(lda_ori.replace("lda_",""))
            if lda not in merge_interest["lda"]:
                merge_interest["lda"][lda] = 0.0
            score = lda_map[lda_ori]
            merge_interest["lda"][lda] += coeff*score
   
    except:
        traceback.print_exc()
    item = recent_interest
    try:
        topic_map = item.get("topic_map",{})
        for topic in topic_map:
            if topic not in merge_interest["topic"]:
                merge_interest["topic"][topic] = 0.0
            score = topic_map[topic]
            merge_interest["topic"][topic] += score
	    cnt += 1


        kw_map = item.get("kw_map",{})
        for kw in kw_map:
            if kw not in merge_interest["keyword"]:
                merge_interest["keyword"][kw] = 0.0
            score = kw_map[kw]
            merge_interest["keyword"][kw] += score
	    cnt += 1

        tag_map = item.get("tag_map",{})
        for tag in tag_map:
            if tag not in merge_interest["tag"]:
                merge_interest["tag"][tag] = 0.0
            score = tag_map[tag]
            merge_interest["tag"][tag] += score 
	    cnt += 1

        acc_map = item.get("account_map",{})
        for acc in acc_map:
            if acc not in merge_interest["account"]:
                merge_interest["account"][acc] = 0.0
            score = acc_map[acc]
	    cnt += 1
            merge_interest["account"][acc] += score
      	lda_map = item.get("lda_map",{})
        for lda_ori in lda_map:
	    lda = int(lda_ori.replace("lda_",""))
            if lda not in merge_interest["lda"]:
                merge_interest["lda"][lda] = 0.0
            score = lda_map[lda_ori]
            merge_interest["lda"][lda] += coeff*score
    except:
        traceback.print_exc()
    return merge_interest, cnt

def article_feature_for_din(article_info):
    feature_list = []
    doc_id = article_info.get("_id", "")
    if doc_id != "":
        feature_list.append("z1"+construct_independ_feature("HISTORY_ID", "\a", doc_id));
    '''
    account_id = article_info.get("account_openid","")
    if account_id !="":
        feature_list.append("z2"+construct_independ_feature("HISTORY_AC","\a",account_id));
    topic_word_arr = article_info.get("keywords_content","")
    topic_word = ""
    if len(topic_word_arr) != 0:
	print topic_word_arr,article_info
        topic_word = topic_word_arr[0]
    else:
	topic_word_arr = article_info.get("keywords_secondary","")
	if len(topic_word_arr) != 0:
            topic_word = topic_word_arr[0]	
    if topic_word != "":
        feature_list.append("z3"+construct_independ_feature("HISTORY_TP", "\a", topic_word))
    '''
    account_id = ""
    topic_word = "" 
    feature_list.append('zZZZEM_MEM'+':\f'+ \
        "z1"+construct_independ_feature('HISTORY_ID','\a',doc_id)+':\f'+ \
        "z2"+construct_independ_feature('HISTORY_AC','\a',account_id)+':\f'+ \
        "z3"+construct_independ_feature('HISTORY_TP','\a',topic_word))

    video_sig = article_info.get("video_sig", '-1')
    feature_list.append(construct_independ_feature("ATVI", "\a", str(video_sig)))

    original_sig = article_info.get("original_sig", '-1')
    feature_list.append(construct_independ_feature("ATOR", "\a", str(original_sig)))

    return feature_list

def article_feature(article_info):
    feature_list = []
    topic = article_info.get("topic1", "NULL")
    topic = "NULL" if topic == "" else topic
    feature_list.append(construct_independ_feature("ATTP", "\a", topic))
    keyword_list = article_info.get("keywords_content", [])
    keyword_list.sort(key=lambda k:len(k), reverse=True)
    kw_s = ""
    for kw in keyword_list:
        if kw in kw_s:
            continue
        feature_list.append(construct_independ_feature("ATKW", "\a", kw))
        kw_s += kw + "\a"

    keywords_secondary_list = article_info.get("keywords_secondary", [])
    keywords_secondary_list.sort(key=lambda k: len(k), reverse=True)
    kw_s = ""
    for kw in keywords_secondary_list:
        if kw in kw_s:
            continue
        feature_list.append(construct_independ_feature("ATKWSE", "\a", kw))
        kw_s += kw + "\a"
    
    video_sig = article_info.get("video_sig", '-1')
    original_sig = article_info.get("original_sig", '-1')
    locate_enable = article_info.get("locate_enable", '-1')
    source_type = article_info.get("source_type", '-1')
    account_weight = article_info.get("account_weight", '-1')

    feature_list.append(construct_independ_feature("ATVI", "\a", str(video_sig)))
    feature_list.append(construct_independ_feature("ATOR", "\a", str(original_sig)))
    feature_list.append(construct_independ_feature("ATLO", "\a", str(locate_enable)))
    feature_list.append(construct_independ_feature("ATSO", "\a", str(source_type)))
    feature_list.append(construct_independ_feature("ATAC", "\a", str(account_weight)))

    return feature_list

def qqprofile_cross_article(interest, article_info):
    cross_feature_list = []
    qq_profile = interest.get("qq_profile", {})
    sex = qq_profile.get("sexual", "NULL")
    sex = "NULL" if sex not in SEX_SET else sex
    age = qq_profile.get("age", "NULL")
    age = "NULL" if age not in AGE_SET else age
    edu = qq_profile.get("education", "NULL")
    edu = "NULL" if edu not in EDU_SET else edu
    doc_id = article_info.get("_id", "")
    if doc_id == "":
        return cross_feature_list
    topic = article_info.get("topic1", "NULL")
    topic = "NULL" if topic == "" else topic
    cross_feature_list.append(construct_cross_feature("SXATTP", "\a", (sex, topic)))
    cross_feature_list.append(construct_cross_feature("AGATTP", "\a", (age, topic)))
    cross_feature_list.append(construct_cross_feature("EDATTP", "\a", (edu, topic)))
    return cross_feature_list 

def interest2kta_set(merge_interest, int_type):
    interest_dict = merge_interest.get(int_type, {})
    result_dict = {}
    result_set = set()
    for key in interest_dict:
        result_dict[key] = interest_dict[key]
        result_set.add(key)
    return result_dict, result_set

def article2kta_set(article_info, art_type):
    result = set()
    value = article_info.get(art_type, [])
    if isinstance(value, unicode) or isinstance(value, str):
        result.add(value)
        return result
    value.sort(key=lambda k:len(k), reverse=True)
    item_s = ""
    for item in value:
        if item in item_s:
            continue
        result.add(item)
        item_s += item + "\a"
    return result

def article_match_interest(article_info, merge_interest, modelname):
    match_feature_list = []
    match_type_list = ( ("topic", "topic1", "ITATTP"),
                        ("keyword", "keywords_content", "ITATKW"),
                        ("tag", "tag_list", "ITATTG"),
                        ("account", "account_openid", "ITATAC"),
                        )
    for match_type in match_type_list:
        (int_type, art_type, match_prefix) = match_type
        int_dict, int_set = interest2kta_set(merge_interest, int_type)
        art_set = article2kta_set(article_info, art_type)
        match_set = int_set & art_set
	num = 0
        for item in match_set:
	    num += 1
	    if "one_hot" in modelname:
           	 match_feature_list.append(construct_match_feature(match_prefix, "\a", item+str(int(int_dict[item]/0.2))))
      	    else:
            	match_feature_list.append(construct_match_feature_with_value(match_prefix, "\a", item, int_dict[item]))
    lda = article_info.get("lda500")
    if "lda" in modelname:
        for ele in lda:
       	    if ele[0] in merge_interest["lda"]:
	        match_feature_list.append(construct_match_feature_with_value("ITATLD", "\a", "lda_"+str(ele[0]), merge_interest["lda"][ele[0]] + ele[1])) 
    if "t_k" in modelname:
	tp = article_info.get("topic1")
	tp_u = merge_interest.get("topic")
	kw_u = merge_interest.get("keyword")
	tp_u_v = 0
	if tp in tp_u:
	    tp_u_v = tp_u[tp]
	    for ele in article_info.get("keywords_content"):
		if ele in kw_u:
		    match_feature_list.append(construct_cross_feature_with_value("TPC", "\a", (tp, ele), tp_u_v + kw_u[ele])) 

    return match_feature_list

def article_match_subscribe_interest(article_info, subscribe_interest):
    match_feature_list = []
    match_type_list = ( ("kw_dict", "topic1", "SUBITATTP"),
                        ("kw_dict", "keywords_content", "SUBITATKW"),
                        ("kw_dict", "tag_list", "SUBITATTG"),
                        ("acc_dict", "account_openid", "SUBITATAC"),
                        )
    for match_type in match_type_list:
        (int_type, art_type, match_prefix) = match_type
        int_dict, int_set = interest2kta_set(subscribe_interest, int_type)
        art_set = article2kta_set(article_info, art_type)
        #print >> sys.stderr,int_set
        #print >> sys.stderr,art_set
        match_set = int_set & art_set
        for item in match_set:
            match_feature_list.append(construct_match_feature(match_prefix, "\a", item))
    return match_feature_list

def article_match_cross_interest(article_info, merge_interest, modelname):
    match_feature_list = []
    match_type_list = (("topic", "topic1", "ITATCOTP"),)
    for match_type in match_type_list:
        (int_type, art_type, match_prefix) = match_type 
        int_dict, int_set = interest2kta_set(merge_interest, int_type)
        int_list = sorted(int_dict.iteritems(), key=lambda d: d[1], reverse=True)
        art_set = article2kta_set(article_info, art_type)
        match_set = int_set & art_set 
        for mitem in match_set:
            number = 0
            for item, itemvalue in int_list:
                if number >= 3:
                    break;
                if item in match_set:
                    number+=1
                    continue
		if "xxxxone_hot" in modelname:
            	    match_feature_list.append(construct_cross_feature(match_prefix, "\a", (mitem+str(int(int_dict[mitem]/0.2)), item+str(int(itemvalue/0.2)))))
                else:
                    match_feature_list.append(construct_cross_feature_with_value(match_prefix, "\a", (mitem, item), itemvalue + int_dict[mitem]))
		number += 1
    return match_feature_list

def account_feature(account_info):
    acc_feature_list = []
    account_openid = account_info.get("account_openid", "")
    if account_openid == "":
        return acc_feature_list
    acc_feature_list.append(construct_independ_feature("ACID", "\a", account_openid))
    return acc_feature_list

def temporal_feature(ts, feedback_info, article_info):
    feature_list = []
    page_time = article_info.get("page_time", 0)
    diff = (ts - page_time)/3600.0
    feature_list.append(construct_discrete_feature("PTCT", "\a", diff, pagetime_clicktime_interval))

    #location = article_info.get("locate",'NULL')
    #feature_list.append(construct_independ_feature("LOC","\a",location))
    return feature_list

def cal_ctr_score(read_num,show_num,share_num):
    if read_num == -1 or show_num == -1:
        return -1
    if read_num > show_num or share_num > show_num:
        show_num = max(read_num,share_num) * 3
    n = show_num
    p = (read_num+share_num*0.5)*1.0/(n+1)
    if p > 1.0:
        p = 1.0
    z = 1.0
    if n < 5 and p ==0.0:
        score = 0.01
    else:
        score = (p + 1.0/(2*n)* z**2 - z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)))/(1 + 1.0/n * z**2)
    return score

def ctr_feature(hot_info):
    feature_list = []
    app_share_num = hot_info.get("app_share_num", 0)
    app_show_num = hot_info.get("app_show_num", 0)
    app_read_num = hot_info.get("app_read_num", 0)
    score = cal_ctr_score(app_read_num,app_show_num,app_share_num)
    feature_list.append(construct_discrete_feature('ATCTR','\a',score,ctr_score_interval))
    feature_list.append(construct_discrete_feature("ATRD", "\a", app_read_num, art_app_read_interval))
    
    return feature_list

def hot_feature(hot_info):
    feature_list = []
    app_share_num = hot_info.get("app_share_num", 0)
    app_show_num = hot_info.get("app_show_num", 0)
    app_read_num = hot_info.get("app_read_num", 0)
    app_favor_num = hot_info.get("app_favor_num", 0)
    app_read_duration = hot_info.get("app_read_duration", 0)
    news_comment_num = hot_info.get("news_comment_num", 0)
    app_comment_num = hot_info.get("comment_num", 0)
    app_cmt_reply_num = hot_info.get("comment_reply_num", 0)
    app_cmt_like_num = hot_info.get("comment_like_num", 0)
    feature_list.append(construct_discrete_feature("ATCMT","\a",app_comment_num,art_app_comment_interval))
    feature_list.append(construct_discrete_feature("ATCMTRPY","\a",app_cmt_reply_num,art_app_cmt_reply_interval))
    feature_list.append(construct_discrete_feature("ATCMTLIKE","\a",app_cmt_like_num,art_app_cmt_like_interval))

    feature_list.append(construct_cross_discrete_feature("ATSWCL", "\a", \
	app_show_num, art_app_show_interval, app_read_num, art_app_read_interval))
    feature_list.append(construct_cross_discrete_feature("ATSWSA", "\a", \
	app_show_num, art_app_show_interval, app_share_num, art_app_share_interval))
    feature_list.append(construct_cross_discrete_feature("ATSWFV", "\a", \
	app_show_num, art_app_show_interval, app_favor_num, art_app_favor_interval))
    feature_list.append(construct_cross_discrete_feature("ATCLSA", "\a", \
	app_read_num, art_app_read_interval, app_share_num, art_app_share_interval))
    feature_list.append(construct_cross_discrete_feature("ATCLFV", "\a", \
	app_read_num, art_app_read_interval, app_favor_num, art_app_favor_interval))
    return feature_list

m_cf_prefix = ["TITLE_","TAG_","TOPIC_","ACC_","DOCID_","LOC","APP_","NOVEL_","QUERY_"]

def _classifyTargetWord(target_word):
    for prefix in m_cf_prefix:
        if target_word.startswith(prefix):
            return "CF"
    return "CB"

def recall_word_feature(feedback_info, model_name):
    feature_list = []
    rec_reason = feedback_info.get("rec_reason", -1)
    if rec_reason == -1:
        return feature_list
    try:
        rec_reason = int(rec_reason)
    except:
        return feature_list
    target_word = feedback_info.get("target_word", "")
    if target_word == "":
        return feature_list
    if model_name == "newmse_rec_sp":
	if rec_reason == 915 or rec_reason == 942:
	    feature_list.append(construct_independ_feature("RCRR", "\a", str(rec_reason)))	
    elif model_name == "newmse_rec_all" or model_name == "sgsapp_rec_all":
	feature_list.append(construct_independ_feature("RCRR", "\a", str(rec_reason)))
    if _classifyTargetWord(target_word) == "CB":
        feature_list.append(construct_independ_feature("CBRW", "\a", target_word))
    else:
        if "#" in target_word:
            (target_word, score) = target_word.split("#")
            try:
                score = float(score)
            except:
                return feature_list
            feature_list.append(construct_discrete_feature_with_original("CFRW", "\a", target_word, score, recall_word_interval))
        else:
            feature_list.append(construct_independ_feature("CFRW", "\a", target_word))
    return feature_list

def recall_cross_docid(feedback_info, article_info, model_name):
    feature_list = []
    rec_reason = feedback_info.get("rec_reason", -1)
    if rec_reason == -1:
        return feature_list
    try:
        rec_reason = int(rec_reason)
    except:
        return feature_list
    target_word = feedback_info.get("target_word", "")
    if _classifyTargetWord(target_word) == "CF":
        if "#" in target_word:
            (target_word, score) = target_word.split("#")
    if target_word == "":
        return feature_list
    doc_id = article_info.get("_id", "")
    if doc_id == "":
        return feature_list
    feature_list.append(construct_cross_feature("RWATID", "\a", (target_word, doc_id)))
    return feature_list

def classfication_with_read_completion(feedback_info, click, read_duration):
    try:
        read_completion_rate = float(feedback_info.get("read_completion_rate", -1))
    except:
        return click
    if read_duration == -1 and read_completion_rate == 0:
        return 1
    if read_duration > 0 and read_duration < 3.0:
        return 0
    if read_completion_rate < 0.01:
        read_completion_rate = 0
    return read_completion_rate
    
def output_sample( click, dur, mid, doc_id, product, feature_list):
    global FEATURE_SET
    for feature in feature_list:
        FEATURE_SET.add(feature)
    s =  str(click) + "\t" + str(dur) + "\t" + mid + "\t" + doc_id + "\t" + product + "\t" + "\t".join(feature_list)
    print >> sys.stdout, s.encode("gbk", "ignore")

def read_history(user_info, act_ts):
    feature_list = []
    history_json = user_info.get("user_read_history","")
    if not history_json:
        return feature_list
    i = 0
    account_map=[]
    topic_map=[]
    for i in range(len(history_json)):
        ele = history_json[i]
        if not ele:
           continue
        ts = ele.get("op_time",-1)
        doc_id = ele.get("_id","")
        account_id = ele.get("account_openid","")
        topic_word = ""
        topic_word_arr = ele.get("keywords_content","")
        if topic_word_arr!="":
           topic_word = topic_word_arr.split(",")[0]
        topic_word = topic_word.strip("KS_")
        if ts < act_ts-10 and ts!=-1:
           feature_list.append("z1"+construct_independ_feature('HISTORY_ID','\a',doc_id))
           feature_list.append('zZZZMEM_MEM'+':\f'+ \
                "z1"+construct_independ_feature('HISTORY_ID','\a',doc_id)+':\f'+ \
                "z2"+construct_independ_feature('HISTORY_AC','\a',account_id)+':\f'+ \
                "z3"+construct_independ_feature('HISTORY_TP','\a',topic_word))
           if account_id in account_map:
                continue
           else:
                feature_list.append("z2"+construct_independ_feature('HISTORY_AC','\a',account_id))
                account_map.append(account_id)
           if topic_word in topic_map:
                continue
           else:
                feature_list.append("z3"+construct_independ_feature('HISTORY_TP','\a',topic_word))
                topic_map.append(topic_word)
    return  feature_list
def history_match_doc(user_info, article_info, act_ts):
    feature_list = []
    history_json = user_info.get("user_read_history","")
    if not history_json:
        return feature_list
    i = 0
    topic_map=dict()
    for i in range(len(history_json)):
        ele = history_json[i]
        if not ele:
           continue
        ts = ele.get("op_time",-1)
        topic_word = ""
        topic_word_arr = ele.get("keywords_content","")
        if topic_word_arr!="":
           topic_word = topic_word_arr.split(",")[0]
        topic_word = topic_word.strip("KS_")
        if ts < act_ts-10 and ts!=-1 and ts > act_ts - 3600:
            if topic_word in topic_map:
		topic_map[topic_word] += 0.1
	    else:
		topic_map[topic_word] = 1
        if len(topic_map) > 10:
	    break
    art_kw = []
    topic_word_arr = article_info.get("keywords_content","")
    for ele in topic_word_arr:
       art_kw.append(ele.strip("KS_"))
    for ele in art_kw:
	if ele in topic_map:
	    feature_list.append(construct_match_feature_with_value("HKW", "\a", ele, topic_map[ele]))
    return feature_list

def qqprofile(interest):
    feature_list = []
    qq_profile = interest.get("qq_profile", {})
    sex = qq_profile.get("sexual", "NULL")
    age = qq_profile.get("age", "NULL")
    edu = qq_profile.get("education", "NULL")
    feature_list.append(construct_independ_feature("SEX", "\a", sex))
    feature_list.append(construct_independ_feature("AGE", "\a", age))
    feature_list.append(construct_independ_feature("EDU", "\a", edu))
    return feature_list

def nfm_dur_weight_output_sample(click, mid, doc_id, product, article_dur, u_dur, feature_list):
    global FEATURE_SET
    for feature in feature_list:
        FEATURE_SET.add(feature)
    s = str(click) + "\t" + mid + "\t" + doc_id + "\t" + product + "\t" +str(article_dur)+"\t"+str( u_dur) + "\t" + "\t".join(feature_list)
    print >> sys.stdout, s.encode("gbk", "ignore")

def get_dur_feature(dur_info):
    q_score = dur_info["q_score"]
    p_score = dur_info["p_score"]
    w_cnt = dur_info["w_cnt"]
    p_cnt = dur_info["p_cnt"]
    avg_dur = dur_info["avg_dur"]
    feature_q_score = "IQUALS\a" + str(q_score)
    feature_w_cnt = "IWCNT\a" + str(w_cnt)
    feature_p_cnt = "IPCNT\a" + str(p_cnt)
    feature_p_score = "IPORNS\a" + str(p_score)
    feature_avg_dur = "IAVGDUR\a" + str(avg_dur) 
    return [feature_q_score, feature_w_cnt, feature_p_cnt, feature_p_score, feature_avg_dur]

def get_dur_info(article_info, hot_info):
    q_score = article_info.get("quality_score",0)
    p_score = article_info.get("porn_score",0)
    p_cnt = article_info.get("content_img_num",0)
    w_cnt = article_info.get("content_len",0)
    avg_dur = article_info.get("app_read_duration_double",0)
    if q_score > 1:
        q_score = 1
    if q_score < 0:
        q_score = 0
    q_score = int(q_score * 100)
    if w_cnt > 5000:
        w_cnt = 5000
    w_cnt = w_cnt / 100
    if p_cnt > 50:
        p_cnt = 50
    if p_cnt < 0:
        p_cnt = 0
    if p_score < 0:
        p_score = 0
    if p_score > 1:
        p_score = 1
    p_score = int(p_score * 100)
    if avg_dur > 300:
        avg_dur = 300
    if avg_dur < 0:
        avg_dur = 0
    avg_dur = int(avg_dur)
    dur_info = dict()
    dur_info["p_score"] = p_score
    dur_info["q_score"] = q_score
    dur_info["avg_dur"] = avg_dur
    dur_info["p_cnt"] = p_cnt
    dur_info["w_cnt"] = w_cnt
    return dur_info

def main():
    model_name = ""
    product_need = "shida"

    if len(sys.argv) < 2:
        print >> sys.stderr, "Usage python " + sys.argv[0] + " badmid_data_file"
        sys.exit(-1)
    badmid_data_file = sys.argv[1]
    filter_mid_set = load_badcase_mid(badmid_data_file)
    load_interval("./interval.json")
    if len(sys.argv) >= 3 :
        model_name = sys.argv[2]
    if model_name == "sgsapp_nfm" or model_name == "sgsapp_wd" or "sgsapp" in model_name:
        product_need = "sgsapp"
    elif "newmse" in model_name:
	product_need = "newmse"
    for line in sys.stdin:
        line = line.strip().decode("gbk", "ignore")
        if line == "":
            continue
        line_tuple = line.split("\t")
        if len(line_tuple) < 9:
            print >> sys.stderr, "hehe less than 9 columns"
            continue
        if len(line_tuple) > 9:
            (mid, doc_id, ts, feedback_info, interest, hot_info, article_info, account_info, click, product) = line_tuple[0:10]
        if product_need not in product:
             continue
        if mid in filter_mid_set:
            continue
        try:
            ts_int = int(ts)
            click_int = int(click)
        except:
            continue
        ts = ts_int
        click = click_int
        json_list = [feedback_info, interest, hot_info, article_info, account_info]
        legal_json = True
        for i in range(len(json_list)):
            try:
                json_list[i] = json.loads(json_list[i])
            except:
                legal_json = False
                break
        if legal_json is False:
            continue
        (feedback_info, interest, hot_info, article_info, account_info) = json_list
        read_duration = feedback_info.get("read_duration", -1.0)
        topic = article_info.get("topic1", "NULL")
        if topic == "outer_video":
            continue
        account_openid = account_info.get("account_openid", "")
        if account_openid == "zhihu":
            continue
        if article_info.get("video_time","") != "":
            continue
        # if click == 1 and read_duration < 20:
	    # click = 0
        #     #click = classfication_with_read_completion(feedback_info, click, read_duration)
	(merge_interest,cnt) = merge_long_recent_interest(interest)
        if cnt < 30:
	    print "cold pop"
	    continue
	subscribe_interest = interest.get("subscribe_interest",{})
        feature_list = []
	dur_info = get_dur_info(article_info, hot_info)
	q_score = dur_info["q_score"]
    	p_score = dur_info["p_score"]
    	w_cnt = dur_info["w_cnt"]
    	p_cnt = dur_info["p_cnt"]
    	avg_dur = dur_info["avg_dur"]
    
        try:
            feature_list.extend(article_feature(article_info))
            feature_list.extend(account_feature(account_info))
            feature_list.extend(qqprofile_cross_article(interest, article_info))
            feature_list.extend(article_match_interest(article_info, merge_interest, model_name))
            feature_list.extend(article_match_subscribe_interest(article_info, subscribe_interest))
            feature_list.extend(article_match_cross_interest(article_info,merge_interest,model_name))
            feature_list.extend(temporal_feature(ts, feedback_info, article_info))
            feature_list.extend(hot_feature(hot_info))
            feature_list.extend(ctr_feature(hot_info))
            feature_list.extend(recall_word_feature(feedback_info, model_name))
            feature_list.extend(recall_cross_docid(feedback_info, article_info, model_name))
            feature_list.extend(get_dur_feature(dur_info))
	    index = -1
	    temp = []
	    for ele in feature_list:
                index += 1
                if "MITAT" in ele:
                    old = ele.split(":\b")
                    if len(old) == 2:
                        feature = old[0].replace("MITAT","MITATQUALS")+str(q_score) + ":\b" + old[1]
                        feature_list[index] = feature
                        feature = old[0].replace("MITAT","MITATPCNT")+str(p_cnt) + ":\b" + old[1]
                        temp.append(feature)
                        feature = old[0].replace("MITAT","MITATWCNT")+str(w_cnt) + ":\b" + old[1]
                        temp.append(feature)
                        feature = old[0].replace("MITAT","MITATPORNS")+str(p_score) + ":\b" + old[1]
                        temp.append(feature)
                        feature = old[0].replace("MITAT","MITATDUR")+str(avg_dur) + ":\b" + old[1]
                        temp.append(feature)
	    if len(temp) != 0:
	    	feature_list.extend(temp)	
	except Exception as e:
    	    print("type error: " + str(e))
        if feature_list:
            output_sample(click, read_duration, mid, doc_id, product, feature_list)


if __name__ == "__main__":
    main()
