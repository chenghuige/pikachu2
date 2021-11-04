folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/wd.sh \
    --optimizers=bert-sgd,bert-sgd \
    --opt_momentum=0. \
    --learning_rates=0.1,0.1 \
    --num_optimizers=1 \
    --hash_embedding_type=SimpleEmbedding \
    --feature_dict_size=3000000 \
    --deep_only=1 \
    --use_onehot_emb=0 \
    --use_user_emb=1 \
    --use_doc_emb=1 \
    --use_history_emb=0 \
    --model_name=$x \
    $*
    
