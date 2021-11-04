export CUDA_VISABLE_DEVICES=-1
for (( i=$2; i>=$1; i-- ))
do
  echo $i
  python ./gen-records.py --day=$i
done
