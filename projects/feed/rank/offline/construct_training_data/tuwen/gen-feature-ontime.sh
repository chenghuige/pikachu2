start_hour=`date -d "-1 hours" "+%Y%m%d%H"`
interval=1

sh ./gen-features.sh $start_hour $interval >> /home/gezi/tmp/rank.log 2>&1
