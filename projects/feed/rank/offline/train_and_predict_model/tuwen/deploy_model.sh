#!/bin/sh

source ./config.sh

model_pb=$1
abtestid=$2

# compat for old
if [[ $mark == "tuwen" && ($abtestid == 15 || $abtestid == 16) ]]
then
	 model_pb_name="wide_deep_model_sgsapp_wdfield_interest.pb"
fi

if (($# > 2))
then
train_model_hour=$3
else
#train_model_hour=`date -2hours +%Y%m%d%H_%M`
train_model_hour=`date -d "-2 hours" "+%Y%m%d%H"`
echo 'No train_model_hour'
fi

if (($# > 3))
then 
deploy_model_name=$4
fi

dest_dir=${dest_dirs[${abtestid}]}
# update model info
if [[ $mark == "tuwen" && ($abtestid == 16) ]]
then
info_name='ExpRuleInfoApp.json'
else
info_name='ExpRuleInfo.json'
fi
info_path_out="update_${info_name}_${train_model_hour}_${abtestid}"
info_path="${info_name}_${train_model_hour}_${abtestid}"
info_dir="expinfo"
mkdir -p ${info_dir}
scp "10.144.57.79:${dest_dir}/${info_name}" ./${info_path}
echo "Read_ExpInfo:"
cat ./${info_path}
python update_model_info.py $info_path $info_path_out "${deploy_model_name}.${mark}.${abtestid}" $train_model_hour 
echo "Update_ExpInfo:"
cat $info_path_out

md5sum $info_path_out | awk '{print $1}' > ./${info_path_out}.md5


public_data_dir="10.144.57.79:$dest_dir" 

#--------------------------------------
# md5sum
#-------------------------------------
md5sum ${model_pb}|awk '{print $1}' > ${model_pb}.md5
md5sum ${feature_index_field}|awk '{print $1}' > ${feature_index_field}.md5

tar_dir=./data/tar_${mark}_${abtestid}

rm -rf $tar_dir
mkdir $tar_dir
cp -rf ${model_pb}  $tar_dir/${model_pb_name}
cp -rf ${model_pb}.md5  $tar_dir/${model_pb_name}.md5
cp -rf ./${info_path_out} $tar_dir/${info_name} #rename
cp -rf ./${info_path_out}.md5 $tar_dir/${info_name}.md5

mv ${info_path} ${info_dir}
mv ${info_path_out} ${info_dir}
mv ./${info_path_out}.md5 ${info_dir}

dest_dir=${dest_dirs[${abtestid}]}

pushd .
cd $tar_dir
tar czvf ../$deploy_model_name.${abtestid}.tar.gz  ./*
popd

sync_tar()
{
  rsync -avP ./data/$deploy_model_name.${abtestid}.tar.gz ${public_data_dir}/$deploy_model_name.${abtestid}.tar.gz
}

sync_tar 
if [ $? -eq 0 ];then
ssh root@10.144.57.79 "cd $dest_dir; tar xvzf $deploy_model_name.${abtestid}.tar.gz; rm -rf $deploy_model_name.${abtestid}.tar.gz"
echo "update ok!"
fi
