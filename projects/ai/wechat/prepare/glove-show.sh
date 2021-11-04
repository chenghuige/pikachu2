# corpus 是使用show信息制作的

days="14.5 13 15 14"

for day in $days; do
  echo "------------$day"
  sh glove-doc.sh $day
  sh glove-author.sh $day 
  sh glove-singer.sh $day
  sh glove-song.sh $day
  sh glove-user.sh $day 
done


