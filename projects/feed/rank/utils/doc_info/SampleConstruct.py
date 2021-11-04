#coding=gbk
import sys
import traceback
import time
import redis
import json
import config
import datetime
sys.path.append("/usr/lib64/python2.6/site-packages/")
from KV import KV
from decode_resp_log import decode_resp_log
from Hot.ttypes import *
from account_info.ttypes import *
from user_profile.ttypes import *
from serialization import *
import log
logger = log.logger

class SampleInfo(object):
    def __init__(self):
        self.article_hot_redis = redis.Redis(host=config.REDIS_ARTICLE_HOT_HOST,port=config.REDIS_ARTICLE_HOT_PORT,db=0,password=config.REDIS_ARTICLE_HOT_PW)
        self.account_hot_redis = redis.Redis(host=config.REDIS_ACCOUNT_HOT_HOST,port=config.REDIS_ACCOUNT_HOT_PORT,db=0,password=config.REDIS_ACCOUNT_HOT_PW)
        self.kv_recent_interest_conn = KV(config.KV_RECENT_INTEREST_APPID,config.KV_RECENT_INTEREST_NAMESPACE)
        self.kv_long_interest_conn = KV(config.KV_LONG_INTEREST_APPID,config.KV_LONG_INTEREST_NAMESPACE)
        self.kv_qq_profile_conn = KV(config.KV_QQ_PROFILE_APPID, config.KV_QQ_PROFILE_NAMESPACE)
        self.kv_subs_interest_conn = KV(config.KV_SUBSCRIBE_INTEREST_APPID, config.KV_SUBSCRIBE_INTEREST_NAMESPACE)
        self.kv_article_profile_conn = KV(config.KV_ARTICLE_DETAIL_APPID, config.KV_ARTICLE_DETAIL_NAMESPACE)
        #20180502 sunjun add
        self.kv_qq_applist_conn = KV(config.KV_QQ_PROFILE_APPID, config.KV_QQ_APPLIST_NAMESPACE)
        #20180607 liuyaqiong  add
        self.user_read_history_redis = redis.Redis(host=config.REDIS_READ_HISTORY_HOST,port=config.REDIS_READ_HISTORY_PORT,db=0, password=config.REDIS_READ_HISTORY_PW)
    def process(self,u_jo):
        if not u_jo:
            return False
        if "article_list" in u_jo:
            ret_info = self.get_feature_info(u_jo)
            return ret_info
        else:
            return False
            
    def get_feature_info(self,jo):
        ret_list = []
        decode_str_list = decode_resp_log(jo)
        if not decode_str_list:
            #print >> sys.stderr,"resp empty"
            return False
        mid_map={}
        article_hot_map={}
        article_detail_map={}
        article_account_map={}
        docid_list=[]
        for decode_str in decode_str_list:
            decode_str = decode_str.decode("gbk",'ignore')
            (resp,mid,tm,article_cnt,index_num,mark,title,reason,read_num,topic,keywords,pub_time,image_type,img_list,url,account,channel,art_source,account_openid,abtestid,sub_topic,userinfo,position,app_ver,aduser_flag,location,pagetime,rec_reason,adid,vulgar,sub_list,ip,recall_word,video_type,channel_id,docid,product) = decode_str.split("\t")[0:37]
            if channel_id != "1" and channel_id != "101" and channel_id != "35":
                continue

            #if mid.find("-") != -1:
            #    continue
            if not docid:
                continue
            docid_list.append(docid)
        if len(docid_list) == 0:
            return True
        
        start_time = time.time()
        logger.info("get_article hot doc len=%d" %(len(docid_list)))
        article_hot_map = self.getArticleHotFeatureList(docid_list)
        end_time = time.time()
        logger.info("get_article hot cost %f size=%d" %(end_time-start_time, len(article_hot_map)))
        
        start_time= end_time
        logger.info("get_article detail doc len=%d" %(len(docid_list)))
        article_detail_map = self.getArticleDetailFeatureList(docid_list)
        end_time = time.time()
        logger.info("get_article detail cost %f size=%d" %(end_time-start_time, len(article_detail_map)))
        
        for decode_str in decode_str_list:
            decode_str = decode_str.decode("gbk",'ignore')
            (resp,mid,tm,article_cnt,index_num,mark,title,reason,read_num,topic,keywords,pub_time,image_type,img_list,url,account,channel,art_source,account_openid,abtestid,sub_topic,userinfo,position,app_ver,aduser_flag,location,pagetime,rec_reason,adid,vulgar,sub_list,ip,recall_word,video_type,channel_id,docid,product) = decode_str.split("\t")[0:37]
            if channel_id != "1" and channel_id != "101" and channel_id != "35":
                continue

            #if mid.find("-") != -1:
            #    continue
            if not docid:
                continue

            action_dict = {}
            action_dict["location"] = location
            action_dict["target_word"] = recall_word
            action_dict["read_duration"] = -1
            action_dict["rec_reason"] = rec_reason
            action_dict["topic"] = topic
            action_feature = json.dumps(action_dict,ensure_ascii=False)
            
            if mid in mid_map:
                interest_feature = mid_map[mid]
            else:
                start_time = time.time()
                interest_dict = {}
                recent_interest = self.get_recent_interest(mid)
                end_time = time.time()
                logger.info("get_recent_interest cost %f" %(end_time-start_time))
                start_time= end_time
                if recent_interest:
                    recent_interest_str = recent_interest.encode('utf-8','ignore')
                    interest_dict["recent_interest"] = json.loads(recent_interest_str)
                long_interest = self.get_long_interest(mid)
                end_time = time.time()
                logger.info("get_long_interest cost %f" %(end_time-start_time))
                start_time= end_time
                if long_interest:
                    long_interest_str = long_interest.encode('utf-8','ignore')
                    interest_dict["long_interest"] = json.loads(long_interest_str)
                qq_profile = self.get_qq_profile(mid)
                end_time = time.time()
                logger.info("get_qq_profile cost %f" %(end_time-start_time))
                start_time= end_time
                if qq_profile:
                    qq_profile_str = qq_profile.decode("gbk","ignore").encode('utf-8','ignore')
                    interest_dict["qq_profile"] = json.loads(qq_profile_str)
                subscribe_interest = self.get_subscribe_interest(mid)
                end_time = time.time()
                logger.info("get_sub cost %f" %(end_time-start_time))
                start_time= end_time
                if subscribe_interest:
                    subscribe_interest_str = subscribe_interest.decode("gbk","ignore").encode("utf-8","ignore")
                    interest_dict["subscribe_interest"] = json.loads(subscribe_interest_str)
                #20180502 sunjun add
                qq_applist = self.get_qq_applist(mid)
                if qq_applist:
                    qq_applist_str = qq_applist.decode("gbk","ignore").encode("utf-8","ignore")
                    interest_dict["qq_applist"] = json.loads(qq_applist_str)

                user_read_history = self.getReadHistory(mid)
                if user_read_history:
                    interest_dict["user_read_history"]=user_read_history

                interest_feature = json.dumps(interest_dict,ensure_ascii=False)
                mid_map[mid] = interest_feature

            #article_hot_feature = self.getArticleHotFeature(docid)
            #end_time = time.time()
            #logger.info("get_article hot cost %f" %(end_time-start_time))
            #start_time= end_time

            article_hot_feature=""
            if docid in article_hot_map:
                article_hot_feature= article_hot_map[docid]
            else:
                logger.info("docid %s no article hot " %(docid))
            
            #article_detail_feature = self.getArticleDetailFeature(docid)
            #end_time = time.time()
            #logger.info("get_article detail cost %f" %(end_time-start_time))
            #start_time= end_time
            
            article_detail_feature=""
            if docid in article_detail_map:
                article_detail_feature= article_detail_map[docid]
            else:
                logger.info("docid %s no article detail " %(docid))

            account_detail_feature = self.getAccountDetailFeature(account_openid)
            end_time = time.time()
            logger.info("get_account detail cost %f" %(end_time-start_time))
            start_time= end_time
            
            output = mid + "\t" + docid + "\t" + tm + "\t" + action_feature + "\t" + interest_feature + "\t" + article_hot_feature + "\t" + article_detail_feature + "\t" + account_detail_feature + "\t" + index_num + "\t"  + product + "\t" + "resp"
            print output.encode("gbk","ignore")
        return True

    def getArticleDetailFeatureList(self,docid_list):
        res_map={}
        res_list=[]
        try:
            res_list = self.kv_article_profile_conn.mget(docid_list)
        except:
            traceback.print_exc()
            print >> sys.stderr,"get article_info List Error"
            return res_map
        for i in range(len(res_list)):
            res = res_list[i]
            if len(res) == 0:
                continue
            res = res.encode("gbk")
            docid = docid_list[i]
            article_dict = {}
            article_dict = self.parse_article_info(res)
            res_map[docid] = json.dumps(article_dict,ensure_ascii=False)
        return res_map

    def getArticleDetailFeature(self,doc_id):
        print >> sys.stderr,doc_id
        article_dict = {}
        try:
            res = self.kv_article_profile_conn.get(doc_id)
            if res:
                article_dict = self.parse_article_info(res)
        except:
            print >> sys.stderr,res
            traceback.print_exc()
            print >> sys.stderr,"get article_info Error"
            return ""
        return json.dumps(article_dict,ensure_ascii=False)

    def getAccountDetailFeature(self,open_id):
        accout_dict = {}
        try:
            res = self.account_hot_redis.get(open_id)
            accout_dict = self.parse_account_info(res)
        except:
            print >> sys.stderr,"get account_info Error"
            return ""
        return json.dumps(accout_dict,ensure_ascii=False)

    def parse_article_info(self,info_str):
        try:
            info_str = info_str.decode("gbk","ignore")
            info = json.loads(info_str)
        except:
            traceback.print_exc()
            print >> sys.stderr, "Error info : %s" % info_str
            return {}
        res_dic = {}
        res_dic["_id"] = info.get("_id", "")
        res_dic["video_sig"] = info.get("video_sig", -1)
        res_dic["account_openid"] = info.get("account_openid", "")
        res_dic["locate_enable"] = info.get("locate_enable", -1)
        res_dic["page_time"] = info.get("page_time", -1)
        res_dic["group_type"] = info.get("group_type", -1)
        res_dic["locate"] = info.get("locate", "")
        res_dic["topic1"] = info.get("topic1", "")
        res_dic["source_type"] = info.get("source_type", -1)
        res_dic["tag_list"] = info.get("tag_list", [])
        res_dic["keywords_content"] = info.get("keywords_content", [])
        res_dic["keywords_secondary"] = info.get("keywords_secondary", [])
        res_dic["account_weight"] = info.get("account_weight", -1)
        res_dic["original_sig"] = info.get("original_sig", -1)
        res_dic["video_time"] = info.get("video_time", "")
        #2018-04-19 sunjun add
        res_dic['content_img_list'] = info.get("content_img_list", [])
        #sunjun 20180605 add
        res_dic['lda500'] = info.get("lda500", [])
        return res_dic

    def parse_account_info(self,info_str):
        account_info = AccountInfo()
        if info_str:
            try:
                DeserializeThriftMsg(account_info, info_str)
            except:
                print >> sys.stderr, "DeserializeThriftMsg err."
                return {}
        res_dic = {}
        if account_info.account_openid:
            res_dic["account_openid"] = account_info.account_openid
        else:
            res_dic["account_openid"] = ""
        if account_info.account:
            res_dic["account"] = account_info.account
        else:
            res_dic["account"] = ""
        if account_info.account_region:
            res_dic["account_region"] = account_info.account_region
        else:
            res_dic["account_region"] = ""
        if account_info.account_type:
            res_dic["account_type"] = account_info.account_type
        else:
            res_dic["account_type"] = -1
        if account_info.level:
            res_dic["level"] = account_info.level
        else:
            res_dic["level"] = -1
        if account_info.region_level:
            res_dic["region_level"] = account_info.region_level
        else:
            res_dic["region_level"] = -1
        if account_info.weixin_avg_read:
            res_dic["weixin_avg_read"] = account_info.weixin_avg_read
        else:
            res_dic["weixin_avg_read"] = -1
        if account_info.sougourank_avg_read:
            res_dic["sougourank_avg_read"] = account_info.sougourank_avg_read
        else:
            res_dic["sougourank_avg_read"] = -1
        if account_info.subscribe_num:
            res_dic["subscribe_num"] = account_info.subscribe_num
        else:
            res_dic["subscribe_num"] = -1
        if account_info.app_avg_click:
            res_dic["app_avg_click"] = account_info.app_avg_click
        else:
            res_dic["app_avg_click"] = -1
        if account_info.app_avg_show:
            res_dic["app_avg_show"] = account_info.app_avg_show
        else:
            res_dic["app_avg_show"] = -1
        if account_info.app_avg_favor:
            res_dic["app_avg_favor"] = account_info.app_avg_favor
        else:
            res_dic["app_avg_favor"] = -1
        if account_info.app_avg_share:
            res_dic["app_avg_share"] = account_info.app_avg_share
        else:
            res_dic["app_avg_share"] = -1
        if account_info.app_avg_duration:
            res_dic["app_avg_duration"] = account_info.app_avg_duration
        else:
            res_dic["app_avg_duration"] = -1
        res_dic["account_topics"] = account_info.account_topics
        res_dic["account_tags"] = account_info.account_tags
        return res_dic

    def getReadHistory(self,mid):
        try :
           res = self.user_read_history_redis.lrange(mid,0,config.USER_HISTORY_NUM)
        except:
            traceback.print_exc()
            print >> sys.stderr, "Error info : %s" % mid
            return ""
        for i, ele in enumerate(res):
            res[i] = json.loads(ele.decode("gbk","ignore").encode("utf-8","ignore"))
        return res
    def getArticleHotFeatureList(self, docid_list):
        res_list = self.article_hot_redis.mget(docid_list)
        res_map={}
        if len(res_list) != len(docid_list):
            return res_map
        for i in range(len(res_list)):
            res = res_list[i]
            docid = docid_list[i]
            var_str=self.parse_article_hot(res)
            res_map[docid] = var_str
        return res_map
        
    def getArticleHotFeature(self,doc_id):
        res = self.article_hot_redis.get(doc_id)
        return parse_article_hot(res)
    def parse_article_hot(self, res):
        if not res:
            return ""
        feature = HotFeature()
        DeserializeThriftMsg(feature, res)
        f = {}
        f["weixin_read_num"] = feature.weixin_read_num
        f["app_read_num"] = feature.app_read_num
        f["app_show_num"] = feature.app_show_num
        f["app_read_duration"] = feature.app_read_duration
        f["app_favor_num"] = feature.app_favor_num
        f["app_collect_num"] = feature.app_collect_num
        f["app_share_num"] = feature.app_share_num
        f["news_sogourank_pv"] = feature.news_sogourank_pv
        f["news_comment_num"] = feature.news_comment_num
        f["news_participant_num"] = feature.news_participant_num
        f["comment_num"] = feature.comment_num
        f["comment_reply_num"] = feature.comment_reply_num
        f["comment_like_num"] = feature.comment_like_num
        return json.dumps(f)

    def get_interest_from_thrift(self,user_profile,interest_dict):
        cur_user_profile = user_profile.pos_info
        if cur_user_profile:
            if cur_user_profile.topic_map:
                for item in cur_user_profile.topic_map:
                    interest_dict['topic_map'].setdefault(item,cur_user_profile.topic_map[item])
            if cur_user_profile.kw_map:
                for item in cur_user_profile.kw_map:
                     interest_dict['kw_map'].setdefault(item,cur_user_profile.kw_map[item])
            if cur_user_profile.tag_map:
                for item in cur_user_profile.tag_map:
                    interest_dict['tag_map'].setdefault(item,cur_user_profile.tag_map[item])
            if cur_user_profile.account_map:
                for item in cur_user_profile.account_map:
                    interest_dict['account_map'].setdefault(item,cur_user_profile.account_map[item])
            #20180605 sunjun add
            if cur_user_profile.lda_map:
                for item in cur_user_profile.lda_map:
                    interest_dict['lda_map'].setdefault(item,cur_user_profile.lda_map[item])
        

    def get_recent_interest(self, mid):
        recent_interest_dict = {'topic_map':{},'kw_map':{},'tag_map':{},'account_map':{}, 'lda_map':{}}
        profile_str = self.kv_recent_interest_conn.get(mid)
        if not profile_str:
            print >> sys.stderr,"No record for %s's recent profile." % mid
            return False
        user_profile = RecentTermUserProfile()
        try:
            DeserializeThriftMsg(user_profile, profile_str)
        except:
            traceback.print_exc()
            print >> sys.stderr,"Deserialize %s's recent profile failed." % mid 
            return json.dumps(recent_interest_dict,ensure_ascii=False,encoding='gbk')
        if mid != user_profile.mid:
            print >> sys.stderr, "Mid different, request %s, get %s." % (mid, user_profile.mid)
            return json.dumps(recent_interest_dict,ensure_ascii=False,encoding='gbk')
        self.get_interest_from_thrift(user_profile,recent_interest_dict)
        
        recent_interest_str = json.dumps(recent_interest_dict,ensure_ascii=False,encoding='gbk')
        return recent_interest_str
    
    def get_long_interest(self, mid):
        long_interest_dict = {"topic_map":{},"kw_map":{},"tag_map":{},"account_map":{}, 'lda_map':{}}
        profile_str = self.kv_long_interest_conn.get(mid)
        if not profile_str:
            print >> sys.stderr,"No record for %s's long profile." % mid
            return False
        user_profile = LongTermUserProfile()
        try:
            DeserializeThriftMsg(user_profile, profile_str)
        except:
            print >> sys.stderr,"Deserialize %s's long profile failed." % mid 
            return json.dumps(long_interest_dict,ensure_ascii=False,encoding='gbk')
        if mid != user_profile.mid:
            print >> sys.stderr, "Mid different, request %s, get %s." % (mid, user_profile.mid)
            return json.dumps(long_interest_dict,ensure_ascii=False,encoding='gbk')
        
        self.get_interest_from_thrift(user_profile,long_interest_dict)
        
        long_interest_str = json.dumps(long_interest_dict,ensure_ascii=False,encoding='gbk')
        return long_interest_str

    def get_qq_profile(self, mid):
        return self.kv_qq_profile_conn.get(mid)
    
    #20180502 sunjun add
    def get_qq_applist(self, mid):
        return self.kv_qq_applist_conn.get(mid)

    def get_subscribe_interest(self, mid):
        return self.kv_subs_interest_conn.get(mid)
