# notice setting buffer 100w will almot use all mem with each process using 6g
python ./gen-records.py --mark=train --record_name=tfrecords-rand --shuffle_impressions
python ./gen-records.py --mark=train --record_name=tfrecords-rand --shuffle_impressions --train_by_day --buffer_size=100000 --day=0
