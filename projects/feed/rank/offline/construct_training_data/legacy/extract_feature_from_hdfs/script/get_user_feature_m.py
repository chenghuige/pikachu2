#coding=gbk

import sys
import json
import math
import time

model_name = ""
product_need = "sgsapp"
import pdb
#pdb.set_trace()
#line = open("/tmp/a.txt")
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
    #if mid in filter_mid_set:
    #    continue
    try:
        ts_int = int(ts)
        click_int = int(click)
    except:
        continue
    article_show_time = ts_int
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
    (feedback_info, interest_dict, hot_info, article_info, account_info) = json_list
    topic = article_info.get("topic1", "NULL")
    if topic == "outer_video":
        continue
    if article_info.get("video_time","") != "":
        continue
    
    read_duration = feedback_info.get("read_duration", -1.0)
    
    location = feedback_info.get("location", "")
    stt = location.split(",")
    if len(stt) >= 3:
        location = stt[0] + "," + stt[1]
    if len(location) < 1:
        location = "NULL"
    
    rec_reason = feedback_info.get("rec_reason", "NULL")

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

    history_json = interest_dict.get("user_read_history","")
    i = 0
    his_docid=[]
    his_span_time = 0
    for i in range(len(history_json)):
        ele = history_json[i]
        if not ele:
            continue

        act = ele.get("action",-1)
        if act == 0: # action=0 is video
            continue

        h_doc_id = ele.get("_id","")
        if h_doc_id == doc_id:
            continue
        h_doc_id=h_doc_id.strip("\"").strip() # maybe have \t \" mid:88c7da3c2e1e6450765bf26dde105761e06669d15373
        op_ts = ele.get("op_time",-1)
        delt_time = article_show_time - op_ts        
        if delt_time < 10:
            continue
        if his_span_time == 0:
            his_span_time=delt_time
        
        his_docid.append(h_doc_id)


    article_profile=hot_info
    article_share_num = 0.0
    article_comment_num = 0.0
    article_comment_reply_num = 0.0
    article_show_num = 0.0
    article_comment_like_num = 0.0
    article_read_num = 0.0
    article_read_duration = 0.0
    article_favor_num = 0.0
    try:
        article_share_num = float(article_profile["app_share_num"])
        article_comment_num = float(article_profile["comment_num"])
        article_comment_reply_num = float(article_profile["comment_reply_num"])
        article_show_num = float(article_profile["app_show_num"])
        article_comment_like_num = float(article_profile["comment_like_num"])
        article_read_num = float(article_profile["app_read_num"])
        article_read_duration = float(article_profile["app_read_duration"])
        article_favor_num = float(article_profile["app_favor_num"])
    except:
        continue
    
    article_profile=article_info
    
    article_topic = ""
    article_tag_list = []
    article_keywords_secondary = []
    article_keywords_content = []
    article_page_time = 0
    article_account_openid = ""
    article_video_time = 1
    article_source_type = ""
    article_account_weight = ""
    try:
        article_topic = article_profile["topic1"]
        article_tag_list = article_profile["tag_list"]
        article_keywords_secondary = article_profile["keywords_secondary"]
        article_keywords_content = article_profile["keywords_content"]
        article_page_time = int(article_profile["page_time"])
        article_account_openid = article_profile["account_openid"]
        if len(article_account_openid) < 1:
            article_account_openid = "NULL"
        '''video_time = article_profile["video_time"]
        video_time_st = video_time.split(":")
        for i in range(len(video_time_st)):
            article_video_time = article_video_time + math.pow(60, len(video_time_st) - i - 1) * int(video_time_st[i])'''
        article_source_type = str(article_profile["source_type"])
        article_account_weight = str(article_profile["account_weight"])
    except:
        continue

    article_click = click
    if article_click == "1" and read_duration < 1.0:
        continue
    out = article_click
    #时间特征
    timearray = time.localtime(article_show_time)
    weekday = timearray.tm_wday
    timespan = timearray.tm_hour
    out = out + "\t" + "WEEK\a" + str(weekday) + "\t" + "TIMESPAN\a" + str(timespan)

    #召回词特征
    #out = out + "\t" + "TARGET_WORD\a" + target_word

    #地域特征
    #out = out + "\t" + "LOCATION\a" + location

    #用户topic特征
    topic_li = []
    for k in topic_dict.keys():
        if k.find("id") != -1: #filter dirty data {"id":"5c4421368e477c1202d31089":0.0728
            continue
        if topic_dict[k] >= 0.05:
            topic_li.append(k)
    if len(topic_li) > 0:
        out = out + "\t" + "USERTOPIC\a"
        out = out + "\b".join(topic_li)
    else:
        out = out + "\t" + "USERTOPIC\aNULL"
    #klist = topic_dict.keys()
    #out = out + "\b".join(klist)
    #用户kw, tag特征
    kw_li = []
    para = len(kw_dict) / 1000.0
    #if len(kw_dict) > 60:
    #    para = 0.1
    if para > 0.1:
        para = 0.1
    if para < 0.02:
        para = 0.02
    for k in kw_dict.keys():
        if kw_dict[k] >= para:
            kw_li.append(k)
    tag_li = []
    for k in tag_dict.keys():
        if tag_dict[k] >= para:
            tag_li.append(k)
    if len(kw_li) > 0 or len(tag_li) > 0:
        out = out + "\t" + "USERKWTAG\a"
        #klist = list(set(tag_li) | set(kw_li))
        klist = tag_li
        s = set(tag_li)
        for k in kw_li:
            if k in s:
                continue
            klist.append(k)
        out = out + "\b".join(klist)
    else:
        out = out + "\t" + "USERKWTAG\aNULL"
    #klist = list(set(kw_dict.keys()) | set(tag_dict.keys()))
    #out = out + "\b".join(klist)


    #文章阅读,分享,展现等信息
    article_time_span = (article_show_time - article_page_time)/(60.0 * 60.0)
    if article_time_span < 1.0:
        article_time_span = 1.0
    #if article_show_num < 1.0:
    #    article_show_num = 1.0
    #if article_read_num < 1.0:
    #    article_read_num = 1.0/article_time_span;
    share_ctr = article_share_num/(article_show_num if article_show_num > 0.0 else 1.0)
    comment_ctr = article_comment_num/(article_show_num if article_show_num > 0.0 else 1.0)
    comment_reply_ctr = article_comment_reply_num/(article_show_num if article_show_num > 0.0 else 1.0)
    comment_like_ctr = article_comment_like_num/(article_show_num if article_show_num > 0.0 else 1.0)
    read_ctr = article_read_num/(article_show_num if article_show_num > 0.0 else 1.0)
    if article_video_time < 1.0:
        article_video_time = 1.0
    #read_duration_ctr = ((1.0 - 1.0/(article_read_num + 1.0)) * article_read_duration + 1.0/(article_read_num + 1.0) * article_video_time)/float(article_video_time)
    '''read_duration_ctr = article_read_duration / article_video_time'''
    favor_ctr = article_favor_num/(article_show_num if article_show_num > 0.0 else 1.0)
    out = out + "\t" + "ARTICLETIMESPAN\a" + str(article_time_span)
    out = out + "\t" + "SHOWNUM\a" + str(article_show_num)
    out = out + "\t" + "READNUM\a" + str(article_read_num)
    out = out + "\t" + "SHARENUM\a" + str(article_share_num)
    out = out + "\t" + "COMMENTNUM\a" + str(article_comment_num)
    out = out + "\t" + "COMMENTREPLYNUM\a" + str(article_comment_reply_num)
    out = out + "\t" + "COMMENTLIKENUM\a" + str(article_comment_like_num)
    #out = out + "\t" + "READDURATION\a" + str(article_read_duration)
    out = out + "\t" + "NOWREADDURATION\a" + str(read_duration) #for get_tensor_sample.py weight
    #out = out + "\t" + "VIDEOTIME\a" + str(article_video_time)
    out = out + "\t" + "FAVORNUM\a" + str(article_favor_num)
    out = out + "\t" + "SHARECTR\a" + str(share_ctr)
    out = out + "\t" + "COMMENTCTR\a" + str(comment_ctr)
    out = out + "\t" + "COMMENTREPLYCTR\a" + str(comment_reply_ctr)
    out = out + "\t" + "COMMENTLIKECTR\a" + str(comment_like_ctr)
    out = out + "\t" + "READCTR\a" + str(read_ctr)
    '''out = out + "\t" + "READDURATIONCTR\a" + str(read_duration_ctr)'''
    out = out + "\t" + "FAVORCTR\a" + str(favor_ctr)

    #推荐理由信息
    if len(rec_reason) > 0:
        out = out + "\t" + "RECREASON\a" + rec_reason
    else:
        out = out + "\t" + "RECREASON\aNULL"

    #文章来源信息
    if len(article_source_type) > 0:
        out = out + "\t" + "SOURCETYPE\a" + article_source_type
    else:
        out = out + "\t" + "SOURCETYPE\aNULL"

    #文章账号等级
    if len(article_account_weight) > 0:
        out = out + "\t" + "ACCOUNTWEIGHT\a" + article_account_weight
    else:
        out = out + "\t" + "ACCOUNTWEIGHT\aNULL"

    #article topic 信息
    if len(article_topic) > 0:
        out = out + "\t" + "ARTICLETOPIC\a" + article_topic
    else:
        out = out + "\t" + "ARTICLETOPIC\aNULL"


    #article kw 信息
    kw_list = list(set(article_keywords_secondary) | set(article_keywords_content) | set(article_tag_list))
    if len(kw_list) > 0:
        out = out + "\t" + "ARTICLEKW\a"
    else:
        out = out + "\t" + "ARTICLEKW\aNULL"
    out = out + "\b".join(kw_list)
    
    #加入mid信息
    try:
        out = mid + "\a" + str(article_show_time) + "\t" + doc_id + "\t" + out
    except:
        continue

    if len(his_docid) != 0:
        out = out + "\t" + "CLICKTIMESPAN\a" + str(his_span_time) + "\t" + " ".join(his_docid)
    else:
        out = out + "\tNULL"

    print out.encode("gbk","ignore")
