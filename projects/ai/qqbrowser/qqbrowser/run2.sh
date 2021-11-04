#sh $1 $*
#sh $1 --ft $*

sh $1 --online $*
sh $1 --ft --online $*
sh $1 --ft --online --mode=test $*

sh $1 $*
sh $1 --ft $*
sh $1 --ft $* --fold_=1
sh $1 --ft $* --fold_=2
sh $1 --ft $* --fold_=3
sh $1 --ft $* --fold_=4

sh ./run/11/loop.sh

