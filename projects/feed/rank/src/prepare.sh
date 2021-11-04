#sh prepare/gen-valid.sh 2019090100_23 --over_write=1 --mask_fields=3 & 
#PID1=$!
#sh prepare/gen-train.sh 2019090100_23 --over_write=1 --mask_fields=3 &
#PID2=$!
#sh prepare/gen-valid.sh 2019083003_23 --over_write=1 --mask_fields=3 &
#PID3=$!
#sh prepare/gen-train.sh 2019083003_23 --over_write=1 --mask_fields=3 &
#PID4=$!
#wait $PID1 $PID2 $PID3 $PID4

#export DIR=/home/gezi/new/temp/feed/rank/zjx_data_2/
#sh ./prepare/gen-valid.sh --padded_tfrecord=1 --over_write=1 >& /tmp/valid.log & 
#sh ./prepare/gen-train.sh --padded_tfrecord=1 --over_write=1 >& /tmp/train.log & 
export DIR=/home/gezi/new/temp/feed/rank/zjx_cross
sh ./prepare/gen-valid.sh  --over_write=1 >& /tmp/valid.log & 
sh ./prepare/gen-train.sh  --over_write=1 >& /tmp/train.log & 
wait %1 %2
