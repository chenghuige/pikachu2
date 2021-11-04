folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./train/${v}/dlrm.sh \
    --masked_fields='AGE|EDU|SEX|ATVI|ATOR|ATLO|ATSO|ATAC|IQUALS|IWCNT|IPCNT|IPORNS|AVGDUR|ACID|ATSWCL|ATSWSA|ATSWFV|ATCLSA|ATCLFV|ATCMT|ATCMTRPY|ATCMTLIKE|FAVORNUM' \
    --mask_mode=regex-incl \
    --deep_only=1 \
    --use_deep_val=1 \
    --fields_pooling_after_mlp='' \
    --model_name=$x \
    $*
    
