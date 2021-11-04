python gen-records.py --mode=valid --merge_text=0 --max_title_len=40 --rv=$1 
python gen-pairwise.py --rv=$1
python gen-records.py --mode=test  --merge_text=0 --max_title_len=40 --rv=$1
python gen-records.py --mode=test_a  --merge_text=0 --max_title_len=40 --rv=$1
python gen-records.py --mode=test_b  --merge_text=0 --max_title_len=40 --rv=$1
python gen-records.py --mode=train --merge_text=0 --max_title_len=40 --rv=$1
#python gen-pairwise-ext.py --rv=$1
