bin=./gen-records.py
max_len=$1 
last_tokens=$2
padding=$3
pretrained=$4
python $bin ../input/jigsaw-multilingual-toxic-comment-classification/test.csv --max_len=$max_len  --last_tokens=$last_tokens --padding=$padding --pretrained=$pretrained
python $bin ../input/jigsaw-multilingual-toxic-comment-classification/validation.csv --max_len=$max_len  --last_tokens=$last_tokens --padding=$padding --pretrained=$pretrained
python $bin ../input/jigsaw-multilingual-toxic-comment-classification/validation.csv --max_len=$max_len  --last_tokens=$last_tokens --padding=$padding --pretrained=$pretrained  --valid_by_lang
python $bin ../input/jigsaw-multilingual-toxic-comment-classification/translate/test-en.csv --max_len=$max_len  --last_tokens=$last_tokens --padding=$padding --pretrained=$pretrained
python $bin ../input/jigsaw-multilingual-toxic-comment-classification/translate/validation-en.csv --max_len=$max_len  --last_tokens=$last_tokens --padding=$padding --pretrained=$pretrained
python $bin ../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv --max_len=$max_len  --last_tokens=$last_tokens --padding=$padding --pretrained=$pretrained
python $bin ../input/jigsaw-multilingual-toxic-comment-classification/translate/jigsaw-toxic-comment-train-google-ru-cleaned.csv --max_len=$max_len  --last_tokens=$last_tokens --padding=$padding --pretrained=$pretrained
python $bin ../input/jigsaw-multilingual-toxic-comment-classification/translate/jigsaw-toxic-comment-train-google-pt-cleaned.csv --max_len=$max_len  --last_tokens=$last_tokens --padding=$padding --pretrained=$pretrained
python $bin ../input/jigsaw-multilingual-toxic-comment-classification/translate/jigsaw-toxic-comment-train-google-fr-cleaned.csv --max_len=$max_len  --last_tokens=$last_tokens --padding=$padding --pretrained=$pretrained
python $bin ../input/jigsaw-multilingual-toxic-comment-classification/translate/jigsaw-toxic-comment-train-google-tr-cleaned.csv --max_len=$max_len  --last_tokens=$last_tokens --padding=$padding --pretrained=$pretrained
python $bin ../input/jigsaw-multilingual-toxic-comment-classification/translate/jigsaw-toxic-comment-train-google-it-cleaned.csv --max_len=$max_len  --last_tokens=$last_tokens --padding=$padding --pretrained=$pretrained
python $bin ../input/jigsaw-multilingual-toxic-comment-classification/translate/jigsaw-toxic-comment-train-google-es-cleaned.csv --max_len=$max_len  --last_tokens=$last_tokens --padding=$padding --pretrained=$pretrained
python $bin ../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv --num_records=40 --max_len=$max_len  --last_tokens=$last_tokens --padding=$padding --pretrained=$pretrained

