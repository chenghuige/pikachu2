python ./prepare/gen-records.py \
    --input=$DIR/valid/*  \
    --feat_file_path=$DIR/feature_index \
    --field_file_path=$DIR/feat_fields.txt \
    --out_dir=$DIR/tfrecord/valid/ \
    $*
   
