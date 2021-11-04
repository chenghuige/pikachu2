python ./gen-records.py --mark=dev
python ./gen-records.py --mark=train --neg_parts=5 --neg_part=0
python ./gen-records.py --mark=test
python ./gen-records.py --mark=train --train_by_day
python ./gen-records.py --mark=train --train_by_day --day=6
python ./gen-records.py --mark=train --neg_parts=5 --neg_part=1
python ./gen-records.py --mark=train-dev --neg_parts=5 --neg_part=0
python ./gen-records.py --mark=train --train_by_day --neg_parts=5 --neg_part=0
python ./gen-records.py --mark=train --train_by_day --day=6 --neg_parts=5 --neg_part=0
python ./gen-records.py --mark=train --train_by_day --neg_parts=5 --neg_part=1
python ./gen-records.py --mark=train --train_by_day --day=6 --neg_parts=5 --neg_part=1
python ./gen-records.py --mark=train-dev --neg_parts=5 --neg_part=1

