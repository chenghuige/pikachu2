mark="video"
deploy_model_name="sgsapp_video_wide_deep_hour_wdfield_interest"
model_name="video_hour_sgsapp_v1"

# dest_dir_16="/search/odin/public_data/video"
# dest_dir_15="/search/odin/public_data/chenghuige/video"
# dest_dir_8="/search/odin/public_data/newdata/video"
# dest_dir_18="/search/odin/public_data/dlrm/video-att"
dest_dir_1="/search/odin/public_data/video.dlrm.shareKwEmb"  # mkyuwen online
#dest_dir_1="/search/odin/yuwenmengke/video_addSrh_addftrl"  # mkyuwen test

# declare -A dest_dir=(["15"]="01" ["16"]="02")
declare -A dest_dir
# dest_dirs["16"]=$dest_dir_16
# dest_dirs["15"]=$dest_dir_15
# dest_dirs["8"]=$dest_dir_8
# dest_dirs["18"]=$dest_dir_18
dest_dirs["1"]=$dest_dir_1

model_pb_name="model.pb"
ROOT="/search/odin/publicData/CloudS/yuwenmengke/rank_0521"
