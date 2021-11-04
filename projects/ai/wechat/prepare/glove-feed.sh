pushd .
cd ../../../../third/glove/ 
make 
popd

# corpus 是直接使用feed自身信息 

bash ./glove-word.sh
bash ./glove-char.sh
bash ./glove-key.sh
bash ./glove-tag.sh

