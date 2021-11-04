over_write=0
TAG1=2019082815_23
sh prepare/gen-valid.sh $TAG1 --over_write=$over_write  & 
PID1=$!
sh prepare/gen-train.sh $TAG1 --over_write=$over_write  &
PID2=$!
TAG2=2019090215_23
sh prepare/gen-valid.sh $TAG2 --over_write=$over_write  &
PID3=$!
sh prepare/gen-train.sh $TAG2 --over_write=$over_write  &
PID4=$!

wait $PID1 $PID2 $PID3 $PID4 

#rm -rf ../input/2019083003_23/model
#rm -rf ../input/2019090100_23/model 

export DIR=../input/$TAG1
sh ./train/v9/dense.sh &
export DIR=../input/$TAG2
sh ./train/v9/dense.sh &

