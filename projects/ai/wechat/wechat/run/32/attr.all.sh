folder=$(dirname "$0")
v=${folder##*/}
x=$(basename "$0")
echo $x
x=${x%.*}

sh ./run/$v/base.sh \
  --model=Model \
  --mean_unk \
  --feats=user,doc,feed,device,day,author,singer,song,video_time,video_time2 \
  --feats2=manual_tags,machine_tags,manual_keys,machine_keys,desc,ocr,asr \
  --mname=$x \
  $*


