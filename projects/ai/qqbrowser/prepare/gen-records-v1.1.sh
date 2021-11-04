python gen-records.py --mode=valid --rv=$1 --max_title_len=66 --segmentor=sp
python gen-records.py --mode=test --rv=$1 --max_title_len=66 --segmentor=sp
python gen-records.py --mode=test_a --rv=$1 --max_title_len=66 --segmentor=sp
python gen-records.py --mode=test_b --rv=$1 --max_title_len=66 --segmentor=sp
python gen-records.py --mode=train --rv=$1 --max_title_len=66 --segmentor=sp
python gen-pairwise.py --rv=$1
#python gen-pairwise-ext.py --rv=$1

