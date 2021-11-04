content="$*"
to="chenghuige,chenweipeng"

pubid="feed"
apiToken="880e55c982e1748add7cbb17c8e0b28b"
ts="`date +%s`000"

token=`echo -n "$pubid:$apiToken:$ts" |md5sum|awk '{print $1}'`
local_ip=`/sbin/ifconfig -a|grep inet|grep -v 127.0.0.1|grep -v inet6|awk '{print $2}'|tr -d "addr:"|head -1`

content="$content"
echo $content
curl -X POST -d "pubid=$pubid&token=$token&ts=$ts&to=$to&content=$content" https://puboa.sogou-inc.com/moa/sylla/open/v1/pns/send
