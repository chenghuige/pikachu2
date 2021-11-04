#!/bin/bash

function LOG() {
    echo "[`date +%Y%m%d-%H:%M:%S`] $@" >&2
}

function DIE() {
    LOG $@
    exit 1
}
function error(){
    echo -e "\033[31m\033[01m$(date +'%Y-%m-%d %H:%M:%S') [ERROR]: $@\033[0m" >> /dev/stderr
}

[ X`command -v git` = X ] && DIE "git is not installed!"
sunshine=`hadoop version | grep 2.6.0-cdh5.10.0 | head -n 1 2>/dev/null`
#[ "X`command -v hadoop`" = X -o "X$sunshine" = X ] && DIE "hadoop client(sunshine) is not installed!"
if (( `ifconfig | grep eth0 | wc -l` == 0 ));then
     error "please install net-tools first (e.g yum install net-tools)"
     exit 1
fi

TIME=$(date +%Y%m%d-%H%M%S)

#uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu
TGZ_NAME="spark-2.4.3-bin-sogou"
LATEST_TGZ=$TGZ_NAME.tgz

LOCAL_INSTALL_DIR=/opt

HDFS_DIST_ROOT_DIR=/user/spark/dist
HDFS_DIST_LATEST_DIR=$HDFS_DIST_ROOT_DIR/latest

SPARK_CONFIG_GIT_REPO=http://gitlab.dev.sogou-inc.com/sogou-spark/spark-config.git
#uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu
SPARK_CONFIG_GIT_BRANCH=sogou-2.4.3-sunshine-js

LOCAL_INSTALL_TMP_ROOT=/tmp/.spark; mkdir -p $LOCAL_INSTALL_TMP_ROOT
LOCAL_INSTALL_TMP_DIR=`mktemp -d -p $LOCAL_INSTALL_TMP_ROOT`
pushd $LOCAL_INSTALL_TMP_DIR

# choose the active namenode
#uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu
SUNSHINE_NN_HFTP=viewfs://nsX

LOG "step-1: download latest spark tgz file: $LATEST_TGZ"
hadoop fs -get $SUNSHINE_NN_HFTP/$HDFS_DIST_LATEST_DIR/$LATEST_TGZ .
[ $? -ne 0 ] && DIE "fail to download $LATEST_TGZ"

LOG "step-2: decompres spark tgz file"
tar -xzvf $LATEST_TGZ

LOCAL_CONFIG_TMP_ROOT=$LOCAL_CONFIG_TMP_DIR/.config; mkdir -p $LOCAL_CONFIG_TMP_ROOT
LOCAL_CONFIG_TMP_DIR=`mktemp -d -p $LOCAL_CONFIG_TMP_ROOT`
pushd $LOCAL_INSTALL_TMP_DIR
LOG "step-3: git clone spark-config project to local"
 git clone -b $SPARK_CONFIG_GIT_BRANCH $SPARK_CONFIG_GIT_REPO
LOG "step-4: copy file to spark project"
## cd spark-config
cd spark-config
for f in $(ls conf)
do
    LOG "copy config file $f to /opt/spark/conf ..."
    cp conf/$f $LOCAL_INSTALL_TMP_DIR/$TGZ_NAME/conf
done
mkdir -p $LOCAL_INSTALL_TMP_DIR/$TGZ_NAME/test
for f in $(ls usr/bin)
do
    LOG "copy executable file $f to /usr/bin ..."
    cp usr/bin/$f /usr/bin
done
popd
rm -fr $LOCAL_CONFIG_TMP_ROOT

LOG "step-5: backup old spark project, move lastest spark project to dir $LOCAL_INSTALL_DIR/spark"
[ -d $LOCAL_INSTALL_DIR/spark ] && mv $LOCAL_INSTALL_DIR/spark $LOCAL_INSTALL_DIR/spark-bak-$TIME
mv $LOCAL_INSTALL_TMP_DIR/$TGZ_NAME $LOCAL_INSTALL_DIR/spark

popd

LOG "step-6: clean up tmp files"
rm -fr $LOCAL_INSTALL_TMP_ROOT

#LOG "step-7: add hive-site.xml softlink"
#cd $LOCAL_INSTALL_DIR/spark/conf; ln -s /opt/datadir/conf/hive-site.xml .
if [[ `cat /proc/1/cgroup | grep docker | wc -l` -eq 0 ]]
then
	echo "spark.driver.host `ifconfig eth0 | grep inet | grep -v inet6 | awk '{print $2}' | awk -F':' '{if ($1=="addr") print $2; else print $1}'`" >>/opt/spark/conf/spark-defaults.conf
fi
LOG "Spark installed succeed!!!"
