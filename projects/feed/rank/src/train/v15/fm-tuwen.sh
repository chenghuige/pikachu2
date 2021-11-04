folder=$(dirname "$0")
v=${folder##*/}

sh ./train/${v}/fm.sh \
    --click_power=1.1 \
    $*
    
