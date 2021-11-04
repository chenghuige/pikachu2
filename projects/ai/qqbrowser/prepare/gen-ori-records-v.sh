#sudo rm -rf ../input/tfrecords$1
python gen-records.py --mode=valid --rv=$1 --merge_text=0
python gen-records.py --mode=test --rv=$1 --merge_text=0
python gen-records.py --mode=test_a --rv=$1 --merge_text=0
python gen-records.py --mode=test_b --rv=$1 --merge_text=0
python gen-records.py --mode=train --rv=$1 --merge_text=0
python gen-pairwise.py --rv=$1 --merge_text=0

