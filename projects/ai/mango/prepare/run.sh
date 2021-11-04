export CUDA_VISABLE_DEVICES=-1
python ./gen-vocabs.py 
python ./dump-day-vids-uids.py 
python ./gen-bins.py 

