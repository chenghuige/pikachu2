#!/bin/sh

model_pb=$1
feature_index_field=$2
model_name=sgsapp_video_wide_deep_hour_wdfield_interest
if (($# > 2))
then 
model_name=$3
fi

model_pb_name="model.pb"
feature_index_name="feature_index_field"

dest_dir='/search/odin/public_data/video'
#dest_dir='/search/odin/chenghuige/video'

public_data_dir="10.144.57.79:$dest_dir" 


#--------------------------------------
# md5sum
#-------------------------------------
md5sum ${model_pb}|awk '{print $1}' > ${model_pb}.md5
md5sum ${feature_index_field}|awk '{print $1}' > ${feature_index_field}.md5

tar_dir=./data/tar_video

rm -rf $tar_dir
mkdir $tar_dir
cp -rf ${model_pb}  $tar_dir/${model_pb_name}
cp -rf ${model_pb}.md5  $tar_dir/${model_pb_name}.md5
cp ${feature_index_field} $tar_dir/${feature_index_name}
cp ${feature_index_field}.md5 $tar_dir/${feature_index_name}.md5


pushd .
cd $tar_dir
tar czvf ../$model_name.tar.gz  ./*
popd

sync_model()
{
  rsync --progress ${model_pb} ${public_data_dir}/${model_pb_name}
  rsync --progress ${model_pb}.md5 ${public_data_dir}/${model_pb_name}.md5 &
}

sync_index()
{
  rsync --progress ${feature_index_field} ${public_data_dir}/${feature_index_name} 
  rsync --progress ${feature_index_field}.md5 ${public_data_dir}/${feature_index_name}.md5 &
}

sync_tar()
{
  rsync --progress ./data/$model_name.tar.gz ${public_data_dir}/$model_name.tar.gz
}


sync_tar 
if [ $? -eq 0 ];then
ssh root@10.144.57.79 "cd $dest_dir; tar xvzf $model_name.tar.gz"
fi

echo "update ok!"

