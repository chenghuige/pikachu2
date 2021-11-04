#!/bin/bash

#rm -rf ../input/tfrecords$1

python gen-records.py --mark=valid --records_name=tfrecords$1
for ((d=1; d<=13; d++))
do
  echo ----------------------------$d
  python gen-records.py --mark=train --records_name=tfrecords$1 --day=$d
done
