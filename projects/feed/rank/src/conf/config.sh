mark=$1
start_hour=2020062400
version="${V:-1}"
product=sgsapp
product_others=newmse,shida
model_name=${mark}_hour_${product}_v${version}
root_exps=../working

src="${SRC:-clouds}"
if [[ $src == 'clouds' ]]
then
  #root="/search/odin/publicData/CloudS/libowei/rank4"
  #root="/search/odin/publicData/CloudS/yuwenmengke/rank_0522_addScore"
  #root="/search/odin/publicData/CloudS/yuwenmengke/rank_0521"
  root="/search/odin/publicData/CloudS/yuwenmengke/rank_0804_so"
  #root="/search/odin/publicData/CloudS/chenghuige/rank_0804_so"
else
  root=${src}
fi

if [[ $src == 'old' ]]
then
  root="/search/odin/publicData/CloudS/libowei/rank"
fi

cloud_root="/search/odin/publicData/CloudS/chenghuige/rank"
base_dir="${root}/${product}/data/${model_name}"

# TODO change to ${mark}/exps ?
exps_dir="${root_exps}/exps/${mark}"
cloud_dir="${cloud_root}/exps/${mark}"

#base_result_dir="/search/odin/publicData/CloudS/libowei/rank_online/infos/${mark}/8"
base_result_dir="/search/odin/publicData/CloudS/mkyuwen/rank_online/infos/video/1"

# Notice for model path or exps path you may soft linke from cloud root to local root (in the future)
# like ln -s /search/odin/publicData/CloudS/chenghuige/rank/data/tuwen_hour_sgsapp_v1/exps /home/gezi/tmp/rank
# Now do not since ClouS not ok for python logging
# 
ROOT=$root
DIR=$base_dir

