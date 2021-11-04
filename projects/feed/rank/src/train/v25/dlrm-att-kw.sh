folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/dlrm-att.sh \
  --use_distribution_emb=1 \
  --use_merge_kw_emb=1 \
  --use_doc_kw_merge_emb=1 \
  --use_doc_kw_secondary_merge_emb=1 \
  --use_rel_vd_history_kw_merge_emb=1 \
  --use_tw_history_kw_merge_emb=1 \
  --use_vd_history_kw_merge_emb=1 \
  --use_vd_long_term_kw_merge_emb=1 \
  --use_tw_long_term_kw_merge_emb=1 \
  --use_long_search_kw_merge_emb=1  \
  --use_new_search_kw_merge_emb=1 \
  --use_user_kw_merge_emb=1 \
  --masked_fields='IATKW$,IATKWSE$,os$,mobile_brand,mobile_model,long_term,last,MSUB.*,^CRW.*,^ICBRW.+,^ICFRW.+' \
  --mask_mode=regex-excl \
  --use_w2v_kw_emb=0 \
  --use_total_attn=0 \
  --merge_kw_emb_pooling='avg' \
  --model_name=$x \
  $*
    
