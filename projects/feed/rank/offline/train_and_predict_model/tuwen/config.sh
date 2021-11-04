mark="tuwen"
deploy_model_name="sgsapp_wdfield_interest_hour"
model_name="tuwen_hour_sgsapp_v1"

# dest_dir_16="/search/odin/public_data"
# dest_dir_15="/search/odin/public_data/chenghuige"
# dest_dir_8="/search/odin/public_data/newdata/tuwen"
# dest_dir_18="/search/odin/public_data/dlrm/tuwen-att"
# dest_dir_19="/search/odin/libowei/public_data"
# dest_dir_1="/search/odin/public_data/tuwen.dlrm.shareKwEmb"  # mkyuwen online
dest_dir_1="/search/odin/public_data/video.dlrm.addSrch" # mkyuwen online 先用着
dest_dir_11="/search/odin/public_data/dlrm/tuwen-att" # mkyuwen online

# declare -A dest_dir=(["15"]="01" ["16"]="02")
declare -A dest_dirs
# dest_dirs["16"]=$dest_dir_16
# dest_dirs["15"]=$dest_dir_15
# dest_dirs["8"]=$dest_dir_8
# dest_dirs["18"]=$dest_dir_18
# dest_dirs["19"]=$dest_dir_19
dest_dirs["1"]=$dest_dir_1
dest_dirs["11"]=$dest_dir_11

model_pb_name="model.pb"
ROOT="/search/odin/publicData/CloudS/yuwenmengke/rank_0521"
