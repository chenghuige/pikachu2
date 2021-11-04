hadoop fs -rmr chg/rank/*v0/exist/gen_feature/$1*
hadoop fs -rmr chg/rank/*v0/gen_feature/*/$1*
hadoop fs -rmr chg/rank/*v0/*/$1*
rm -rf ./data/*v0/$1*
