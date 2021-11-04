#coding=gbk
import datetime as d

import time
import MySQLdb as mdb
import sys
def parse_info(tup):
    pos = "0"
    neg = "0"
    for ele in tup:
	cur = ele.split(":")
	if cur[0] == "1":
	    pos = cur[1]
	elif cur[0] == "0":
	    neg = cur[1]
    return (int(pos), int(neg))
		
def main():
    if len(sys.argv) < 3:
        print >> sys.stderr, "Usage python " + sys.argv[0] + " feature_infogain index_file" 
        sys.exit(-1)

    feature_list_file = sys.argv[1]
    feature_infogain_file = sys.argv[1]
    feature_index_file=sys.argv[2]
    feature_info_out=sys.argv[3]
    feature_index_dict =dict()
    feature_ori_dict = dict()
    model_name = ""
    
    with open(feature_list_file, "r") as fp:
        for line in fp:
            line = line.strip().decode("gbk", "ignore")
            if line == "":
                continue
            line_tuple = line.split("\t")
            if len(line_tuple) < 3:
                continue
            (feature, info1, info2) = line_tuple[:3]
	    (pos, neg) = parse_info([info1, info2])
	    if feature in feature_ori_dict:
		(pos_num, neg_num, stay) = feature_ori_dict[feature] 
		feature_ori_dict[feature] = (pos_num + pos, neg_num + neg, stay)
	    else:
		feature_ori_dict[feature] = (pos, neg, 0)
    with open(feature_index_file, "r") as fp:
        for line in fp:
            line = line.strip().decode("gbk", "ignore")
            if line == "":
                continue
            line_tuple = line.split("\t")
            if len(line_tuple) < 2:
                continue
            (feature, index ) = line_tuple[:2]
	    if feature in feature_ori_dict:
		(pos_num, neg_num, stay) = feature_ori_dict[feature] 
	    	feature_ori_dict[feature] = (pos_num, neg_num, 1)
	    else:
		feature_ori_dict[feature] = (0, 0, -1)
    
    #index_fd = open(feature_info_out, "w")
    category_feature = dict()
    for feature in feature_ori_dict:
	(pos_num, neg_num, stay) = feature_ori_dict[feature]
	prefix = feature.split("\a")[0]
        if prefix in category_feature:
	    (all_dim, all_pos, all_neg, stay_dim, stay_pos, stay_neg) = category_feature[prefix]
	    if stay == 1:
		stay_dim += 1
		stay_pos += pos_num
		stay_neg += neg_num
	    category_feature[prefix] = (all_dim + 1, all_pos + pos_num, all_neg + neg_num, stay_dim, stay_pos, stay_neg)
	else:
	    stay_dim = 0
            stay_pos = 0
	    stay_neg = 0
	    if stay == 1:
		stay_dim = 1
		stay_pos = pos_num
		stay_neg = neg_num
	    category_feature[prefix] = (1, pos_num, neg_num, stay_dim, stay_pos, stay_neg)
    total_pos = 10000
    total_neg = 10000
    update_time = time.time()
    update_time = d.datetime.now().strftime("%Y%m%d%H")
    if "TOTAL_SAMPLES" in category_feature :
        (_, total_pos, total_neg, _, _, _) = category_feature["TOTAL_SAMPLES"]   
    print total_pos, total_neg
    '''
    for feature in category_feature:
	strtemp = [str(ele) for ele in category_feature[feature]]
	s =  feature + "," + ",".join(strtemp)
        index_fd.write("%s\n" % s.encode("gbk", "ignore"))
    index_fd.close()
    '''
    db = mdb.connect(host='feed.feed_monitor.rds.sogou', user='feed_monitor', passwd='FeedMonitor2018', db='feed_monitor')
    model_name = feature_info_out
    cursor = db.cursor()
    for feature in category_feature:
	(ori_dim, ori_pos_num, ori_neg_num, choose_dim, choose_pos_num, choose_neg_num) = category_feature[feature]
	ori_per = (ori_pos_num + ori_neg_num + 0.0) / (total_pos + total_neg)  
	ori_per_pos = (ori_pos_num + 0.0) / (total_pos)
	ori_per_neg = (ori_neg_num + 0.0) / (total_neg)
	print ori_pos_num, ori_neg_num, ori_per_pos, ori_per_neg
	choose_per_num = (choose_pos_num + choose_neg_num + 0.0) / (total_pos + total_neg)
	choose_per_pos_num = (choose_pos_num + 0.0) / (total_pos)
	choose_per_neg_num = (choose_neg_num + 0.0) / (total_neg)
	choose_rate = (choose_dim + 0.0) / ori_dim 
    	sql = "insert into model_feature_check(feature_name, model_name, ori_dim, ori_pos_num, ori_neg_num, ori_per, ori_per_pos, ori_per_neg, choose_dim, choose_pos_num, choose_neg_num, choose_per_num, choose_per_pos_num, choose_per_neg_num, choose_rate, update_time) values('%s', '%s','%s', '%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s')" % (feature, model_name, ori_dim, ori_pos_num, ori_neg_num, ori_per, ori_per_pos, ori_per_neg, choose_dim, choose_pos_num, choose_neg_num, choose_per_num, choose_per_pos_num, choose_per_neg_num, choose_rate, update_time)
	cursor.execute(sql)
	db.commit()
    db.close()
	
if __name__ == "__main__":
    main()    
