export CUDA_VISABLE_DEVICES=-1
python ./gen-records.py --mark=eval --toy
for (( i=30; i>0; i-- ))
do
  python ./gen-records.py --day=$i --toy
done
