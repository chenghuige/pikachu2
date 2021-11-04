folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/wd.sh \
    --optimizers=bert-lazyadam,bert-lazyadam \
    --learning_rates=0.001,0.01 \
    --hash_embedding_type=SimpleEmbedding \
    --feature_dict_size=100000000 \
    --wide_only=1 \
    --model_name=$x \
    $*
    
