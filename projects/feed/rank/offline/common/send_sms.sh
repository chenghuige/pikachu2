#!/usr/bash
urlquote ()
{
    STR=$@
    [ "${STR}x" == "x" ] && { STR="$(cat -)"; }

    echo ${STR} | sed -e 's| |%20|g' \
    -e 's|!|%21|g' \
    -e 's|#|%23|g' \
    -e 's|\$|%24|g' \
    -e 's|%|%25|g' \
    -e 's|&|%26|g' \
    -e "s|'|%27|g" \
    -e 's|(|%28|g' \
    -e 's|)|%29|g' \
    -e 's|*|%2A|g' \
    -e 's|+|%2B|g' \
    -e 's|,|%2C|g' \
    -e 's|/|%2F|g' \
    -e 's|:|%3A|g' \
    -e 's|;|%3B|g' \
    -e 's|=|%3D|g' \
    -e 's|?|%3F|g' \
    -e 's|@|%40|g' \
    -e 's|\[|%5B|g' \
    -e 's|]|%5D|g'
}


msg="$1"
echo $msg
number="17600127400"
url="http://10.134.114.49:10000/sms?content="`urlquote $msg`"&number="`urlquote $number`
curl $url
