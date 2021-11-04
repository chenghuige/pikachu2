export CUDA_VISABLE_DEVICES=-1
for (( i=1; i<=30; i++ ))
do
  python ./gen-records.py --day=$i
done
