export CUDA_VISABLE_DEVICES=-1
for (( i=$1; i<=$2; i++ ))
do
  echo $i
  python ./gen-records.py --day=$i --gen_stars_corpus
done
