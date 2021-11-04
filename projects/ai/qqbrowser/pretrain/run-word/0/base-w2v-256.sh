folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run-word/$v/common.sh \
  --vocab_size=100000 \
  --embedding_path=../input/w2v/sp/256/word.npy \
  --num_attention_heads=8 \
  --custom_model \
  --transformer=bert-base-chinese \
  --mname=$x \
  $*
