parts=$1
root=v1
for ((i=0; i<$parts; i+=1))
do
  sh ./train/$root/test-tta-best.sh --parts=$parts --part=$i &
done
wait 
pushd .
cd ../working/$root/tta-best
zip results.zip ./results/*
popd 
