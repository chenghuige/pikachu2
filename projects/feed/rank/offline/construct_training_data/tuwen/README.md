what to set when deply online:  
in config.py may need to set  
DEBUG COMPRESS MARK(for tuwen or video)
in config.sh may need to set 
join=0 as join cost a lot time 
eval, del_inter_dir

change base_feature_dir  


--------
run.sh -> run-ontime.sh -> |config.sh
                           |gen-features.sh -> function.sh;|gen_features(echo hour) -> |dedup-ori.py
			   |				   |			       |filter-ori.py
                           |                               |                           |join-show.py (now del)
                           |                               |                           |gen-feature.py
                           |                               |stats_feature(echo hour) -> |stats-feature.py
                           |                                                            |stats-field.py 
                           |deal-features.sh -> |deal_feature_index -> |stats-features.sh -> |stats-features.py
                                                |                      |                     |stats-fields.py
                                                |                      |copy_index(hash 6e108) [for online pred]
                                                |gen-tfrecords.sh

ori_path
rank/tuwen_hour_sgsapp_v1/eval
rank/tuwen_hour_sgsapp_v1/exist
rank/tuwen_hour_sgsapp_v1/gen_feature
rank/tuwen_hour_sgsapp_v1/stats_feature
rank/tuwen_hour_sgsapp_v1/stats_features
rank/tuwen_hour_sgsapp_v1/tfrecords

if you do expereiments, you may ignore deal feature index just 
1. sh gen_features.sh  2019110208 24 
2. sh gen_tfrecords.sh 2019110208 24 

