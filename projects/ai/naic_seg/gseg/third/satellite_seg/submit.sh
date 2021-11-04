#!/bin/bash
resuts_dir=$1
use_crf=$2
test_dir=dataset/CCF-testing-Semi

#use crf
if [ ${use_crf} -eq '1' ] ; then
	if [ ! -f  ${resuts_dir}/vis_test_1_post.png ] ;then
		python utils/inference.py ${test_dir}/1.png ${resuts_dir}/vis_test_1_pred.png ${resuts_dir}/vis_test_1_post.png &
	fi
	if [ ! -f ${resuts_dir}/vis_test_2_post.png ] ;then
		python utils/inference.py ${test_dir}/2.png ${resuts_dir}/vis_test_2_pred.png ${resuts_dir}/vis_test_2_post.png &
	fi
	if [ ! -f ${resuts_dir}/vis_test_3_post.png ] ;then
		python utils/inference.py ${test_dir}/3.png ${resuts_dir}/vis_test_3_pred.png ${resuts_dir}/vis_test_3_post.png &
	fi
wait
fi

python submit.py ${resuts_dir} ${use_crf}