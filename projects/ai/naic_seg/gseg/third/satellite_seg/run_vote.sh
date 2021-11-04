#!/bin/bash
model_name="pspnet-densenet-s1s2-crf2"
start_epoch=90
end_epoch=150
interval=20
for i in $(seq 1 3)
do
	index=0
	for epoch in $(seq ${start_epoch} ${interval} ${end_epoch})
	do 
		declare -a vote_inputs
		vote_inputs[${index}]="results/${model_name}/epoch${epoch}/test_${i}_pred.png"
		let index=index+1
	done

	vote_output="results/${model_name}/vote/test_${i}_pred.png"

	python vote.py --inputs ${vote_inputs[@]} \
			   --output ${vote_output} &

	unset inputlist
done