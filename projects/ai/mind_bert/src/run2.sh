#sh ./train/v2/unint-fintune-toxic.sh 
#sh ./train/v2/unint-fintune-toxic.sh --mode=valid --valid_en
#sh ./train/v2/unint-fintune-toxic2.sh 
#sh ./train/v2/unint-fintune-toxic2.sh --mode=valid --valid_en

sh ./train/v2/unint-sample1.sh
sh ./train/v2/unint-sample1.sh --mode=valid --valid_en
sh ./train/v2/toxic-fintune-unint.sh
sh ./train/v2/toxic-fintune-unint.sh --mode=valid --valid_en
#sh ./train/v2/toxic-fintune-unint2.sh
#sh ./train/v2/toxic-fintune-unint2.sh --mode=valid --valid_en
#sh ./train/v2/toxic-and-unint.sh
#sh ./train/v2/toxic-and-unint.sh --mode=valid --valid_en
