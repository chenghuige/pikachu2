rm -rf ../input/tfrecords4
python gen-records.py --mark=valid --records_name=tfrecords4
python gen-records.py --mark=train --records_name=tfrecords4
#python gen-records.py --mark=train --sortby=day --records_name=tfrecords.day

