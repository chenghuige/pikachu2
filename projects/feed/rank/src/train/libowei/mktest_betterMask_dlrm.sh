folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

mktest_model_name=$1
# --model_name=$x \

mktest_better_mask=$2
base_masked_fields='IATKW$,IATKWSE$,os$,mobile_brand,mobile_model,long_term,last,MSUB.*,^CRW.*,^ICBRW.+,^ICFRW.+'
if [ -z "$str" ]; then
    mktest_final_masked_fields=${base_masked_fields},${mktest_better_mask}
else
    echo "mktest_betterMask_dlrm.sh base_masked_fields==NULL ------------"
    exit 1
fi


sh ./train/${v}/dlrm-att.sh \
    --model_name=${mktest_model_name} \
    --field_file_path=./conf/${mark}/betterMask_fields.txt \
    --use_distribution_emb=1 --use_merge_kw_emb=1 --use_doc_kw_merge_emb=1 --use_doc_kw_secondary_merge_emb=1 --use_rel_vd_history_kw_merge_emb=1 --use_tw_history_kw_merge_emb=1 --use_vd_history_kw_merge_emb=1 --use_vd_long_term_kw_merge_emb=1 --use_tw_long_term_kw_merge_emb=1 --use_long_search_kw_merge_emb=1  --use_new_search_kw_merge_emb=1 --use_user_kw_merge_emb=1 --masked_fields=${mktest_final_masked_fields} --mask_mode=regex-excl \
    $*
    
