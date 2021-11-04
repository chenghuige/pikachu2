#rm -rf ../input/tfrecords
python gen-records.py --mode=valid --merge_text=0
python gen-records.py --mode=test  --merge_text=0
python gen-records.py --mode=train --merge_text=0
python gen-pairwise.py --merge_text=0
