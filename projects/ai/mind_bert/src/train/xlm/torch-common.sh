folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

bin=./train.py
if [ $TORCH ];then
  if [[ $TORCH == "1" ]]
  then
     bin=./torch-train.py
  fi
fi

RECORD_DIR=../input/tfrecords/xlm
TORCH=1 sh ./train/${v}/common.sh \
    $*
