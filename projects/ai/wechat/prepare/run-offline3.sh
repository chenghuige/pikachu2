rm -rf ../input/tfrecords3
python gen-records.py --mark=valid --records_name=tfrecords3
python gen-records.py --mark=train --records_name=tfrecords3
#python gen-records.py --mark=train --sortby=day --records_name=tfrecords.day

