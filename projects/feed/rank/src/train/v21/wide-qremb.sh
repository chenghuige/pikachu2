folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/wd.sh \
    --hash_embedding_type=QREmbedding \
    --hash_combiner=mul \
    --feature_dict_size=20000000 \
    --num_feature_buckets=3000000 \
    --optimizers=bert-lazyadam,bert-lazyadam \
    --learning_rates=0.001,0.01 \
    --hash_embedding_type=SimpleEmbedding \
    --feature_dict_size=3000000 \
    --wide_only=1 \
    --model_name=$x \
    $*
    
