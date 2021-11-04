#!/bin/bash
model_name=pspnet-densenet-s1s2-crf2
start_epoch=90
end_epoch=150
interval=20
for i in $(seq ${start_epoch} ${interval} ${end_epoch})
do
	model=snapshot/${model_name}/${i}.pkl
	save_dir=results/${model_name}/epoch${i}
	mkdir -p ${save_dir}
	mkdir -p ${save_dir}/temp
	gpuids=(0 1 2)
	for j in $(seq 1 3)
	do
		test_img=dataset/CCF-testing-Semi/${j}.png
		let index=${j}-1 
		if [ -f  ${save_dir}/test_${j}_pred.png ] ;then
			echo ${save_dir}/test_${j}_pred.png 
			echo "file exists"
			continue
		fi
		CUDA_VISIBLE_DEVICES=${gpuids[${index}]} python test.py --img_path $test_img \
												--out_path ${save_dir}/test_${j}_pred.png \
												--vis_out_path ${save_dir}/vis_test_${j}_pred.png \
												--gpu 0 \
												--batch_size 8 \
												--stride 64 \
												--model_path $model \
												--input_size 256 \
												--crop_scales 192 224 256 288 320 \
												--tempdir ${save_dir}/temp &
	done
	wait
done
