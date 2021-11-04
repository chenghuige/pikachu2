for ((i=0; i<40; i+=1))
do
  sh ./tools/create_click_title_tfrecord.sh $i &
done
