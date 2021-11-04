# 生成tfrecord 

#mkdir -p ./input/quarter
#cd ./input/quarter 
#ln -s /userhome/2020_naic_remote_sensing/semi_final/train .
#cd ../..

rm -rf ./input/quarter/tfrecords 
cd ./prepare 
python gen-records.py --small=1
cd ..

cd ./gseg 
sh run.sh  
cd ..

rm -rf ./gseg/infer/model.h5
cp ./working/ensemble/model.h5 ./gseg/infer 

cd ./gseg/infer
sh zip.sh 
cd ../.. 



