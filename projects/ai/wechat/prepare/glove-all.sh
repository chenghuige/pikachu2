pushd .
cd ../../../../third/glove/ 
make 
popd

sh ./glove-word.sh

days="13 14.5 14 15"

for day in $days; do
  echo "------------$day"
  sh glove-author.sh $day 
  sh glove-singer.sh $day
  sh glove-song.sh $day
done

for day in $days; do
  echo "------------doc $day"
  sh glove-doc.sh $day
done

for day in $days; do
  echo "------------user $day"
  sh glove-user.sh $day 200
done
