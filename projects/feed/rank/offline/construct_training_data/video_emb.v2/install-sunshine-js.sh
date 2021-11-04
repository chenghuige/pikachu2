#!/bin/bash

CLUSTER="sunshine-js"
SCRIPT="install-client.sh"
URL="http://gitlab.dev.sogou-inc.com/wangchengwei/scripts/raw/master/hadoop/install/install.sh"

user=`whoami`
if [ $user != "root" ];then
    echo "Only root user can start to install"
    exit 1
fi

wget  -q  -O  $SCRIPT  $URL
if [ -f $SCRIPT ];then
    sh  $SCRIPT  $CLUSTER
else
    echo "Get the install script fail"
fi

