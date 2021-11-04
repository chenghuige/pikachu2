#coding=gbk
__author__ = "ouyuanbiao"

import sys
import json



#TODO: account discretization
acc_click_interval = [0.0, 100.0, 1000.0]
acc_show_interval = [0.0, 100.0, 1000.0]
acc_favor_interval = [0.0, 100.0, 1000.0]
acc_share_interval = [0.0, 100.0, 1000.0]


#discretization time interval
#pagetime_clicktime_interval = [3600*3.0, 3600*6.0, 3600*12.0, 3600*24.0, 3600*36.0, 3600*48.0, 3600*60.0, 3600*72.0]
pagetime_clicktime_interval = [3.0, 6.0, 12.0, 24.0, 36.0, 48.0, 60.0, 72.0]
ctr_score_interval = [0.01,0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

#discretization hot parameters
art_app_collect_interval = [2.0, 7.0, 20.0, 50.0, 75.0, 146.0] #等频分割
art_app_share_interval = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0] #指数分割
art_app_show_interval = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0, 10000.0] #指数分割
art_app_read_interval = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0, 10000.0] #指数分割
art_app_readduration_interval = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0, 10000.0] #指数分割
art_app_favor_interval = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0] #指数分割
art_app_comment_interval = [2.0,4.0,8.0,16.0,32.0,64.0,128.0,256.0,512.0,1024.0,2048.0] 
art_app_cmt_reply_interval = [2.0,4.0,8.0,16.0,32.0,64.0,128.0,256.0,512.0,1024.0,2048.0] 
art_app_cmt_like_interval = [2.0,4.0,8.0,16.0,32.0,64.0,128.0,256.0,512.0,1024.0,2048.0] 
art_news_comment_interval = [7.0, 16.0, 24.0, 37.0, 52.0, 69.0, 93.0, 120.0, 156.0, 207.0, 263.0, 279.0, 510.0, 766.0, 1192.0, 1778.0, 2889.0, 5315.0, 13280.0, 208506.0]# 等频分割
art_news_participant_interval = [12.0, 24.0, 39.0, 60.0, 83.0, 112.0, 145.0, 190.0, 283.0, 454.0, 659.0, 1112.0, 1649.0, 2568.0, 3949.0, 6055.0, 12349.0, 24525.0, 55210.0, 294619.0, 3193264.0]# 等频分割
art_sogourank_pv_interval = [200.0, 400.0, 600.0, 800.0] #等距分割，该数据很稀疏


#TODO: discretization cf recall_word
recall_word_interval = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


#TODO: discretization interest match 
interest_match_interval = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


result = { "acc_click_interval": acc_click_interval, 
           "acc_show_interval": acc_show_interval, 
           "acc_favor_interval": acc_favor_interval, 
           "acc_share_interval": acc_share_interval,
           "pagetime_clicktime_interval": pagetime_clicktime_interval,
           "ctr_score_interval":ctr_score_interval,
           "art_app_collect_interval": art_app_collect_interval,
           "art_app_share_interval": art_app_share_interval,
           "art_app_show_interval": art_app_show_interval,
           "art_app_read_interval": art_app_read_interval,
           "art_app_readduration_interval": art_app_readduration_interval,
           "art_app_favor_interval": art_app_favor_interval,
           "art_news_comment_interval": art_news_comment_interval,
           "art_news_participant_interval": art_news_participant_interval,
           "art_sogourank_pv_interval": art_sogourank_pv_interval,
           "recall_word_interval": recall_word_interval,
           "art_app_comment_interval":art_app_comment_interval,
           "art_app_cmt_reply_interval":art_app_cmt_reply_interval,
           "art_app_cmt_like_interval":art_app_cmt_like_interval,
        }

print >> sys.stdout, json.dumps(result)
