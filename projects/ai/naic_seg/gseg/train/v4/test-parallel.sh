model=$1
parts=$2
root=v1
for ((i=0; i<$parts; i+=1))
do
  sh ./train/$root/test.sh --parts=$parts --part=$i --mn=$model &
done
wait
pushd .
cd ../working/$root/$model
zip results.zip ./results/*
popd
