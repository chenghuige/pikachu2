from preprocess import write_train_list,crop,generate_trainval_list,ProjectDir
from preprocess import convert_label_to_vis,convert_vis_to_label
import os
import skimage.io as io
import numpy as np

training_data_stage1_dir=os.path.join(ProjectDir,"dataset/CCF-training")
training_data_stage2_dir=os.path.join(ProjectDir,"dataset/CCF-training-Semi")

def generate_stat(label_file_lists):
	label_list=[]
	for label_file in label_file_lists:
		label = io.imread(label_file)
		label_list = label_list + label.flatten().tolist()
	count_label = np.bincount(label_list)
	return count_label

def generate_dataset(dataset_dir,crop_size,img_list,label_list):
	img_path=os.path.join(dataset_dir,'img')
	label_path =os.path.join(dataset_dir,'label')
	visualize_gt_path = os.path.join(dataset_dir,'visualize_gt')

	if(not os.path.exists(img_path)):
		os.mkdir(img_path)
	if(not os.path.exists(label_path)):
		os.mkdir(label_path)
	if(not os.path.exists(visualize_gt_path)):
		os.mkdir(visualize_gt_path)
	for i in range(len(img_list)):
		crop(img_list[i],label_list[i],crop_size,crop_size,prefix='%d'%(i+1),save_dir=dataset_dir,crop_label=True)
	generate_trainval_list(dataset_dir)
	write_train_list(dataset_dir)

##save crf label
#convert_vis_to_label(os.path.join(training_data_stage1_dir,'1_visual_crf.png'),os.path.join(training_data_stage1_dir,'1_class_crf.png'))
#convert_vis_to_label(os.path.join(training_data_stage1_dir,'2_visual_crf.png'),os.path.join(training_data_stage1_dir,'2_class_crf.png'))

convert_vis_to_label(os.path.join(training_data_stage2_dir,'1_visual_crf.png'),os.path.join(training_data_stage2_dir,'1_class_crf.png'))
convert_vis_to_label(os.path.join(training_data_stage2_dir,'2_visual_crf.png'),os.path.join(training_data_stage2_dir,'2_class_crf.png'))
convert_vis_to_label(os.path.join(training_data_stage2_dir,'3_visual_crf.png'),os.path.join(training_data_stage2_dir,'3_class_crf.png'))



img_list_1=[os.path.join(training_data_stage1_dir,'1.png'),
			os.path.join(training_data_stage1_dir,'2.png'),
]
label_list_1=[os.path.join(training_data_stage1_dir,'1_class.png'),
			os.path.join(training_data_stage1_dir,'2_class.png'),
]

img_list_2=[os.path.join(training_data_stage2_dir,'1.png'),
			os.path.join(training_data_stage2_dir,'2.png'),
			os.path.join(training_data_stage2_dir,'3.png')
]
label_list_2=[os.path.join(training_data_stage2_dir,'1_class.png'),
			os.path.join(training_data_stage2_dir,'2_class.png'),
			os.path.join(training_data_stage2_dir,'3_class.png')
]
label_crf_list_2=[os.path.join(training_data_stage2_dir,'1_class_crf.png'),
			os.path.join(training_data_stage2_dir,'2_class_crf.png'),
			os.path.join(training_data_stage2_dir,'3_class_crf.png')
]

#dataset s2
stat=generate_stat(label_list_2)
print "dataset s2 rate: ",np.array(stat)*1.0/np.min(stat[np.nonzero(stat)])
dataset_dir=os.path.join(ProjectDir,"dataset/stage2-train")
if(not os.path.exists(dataset_dir)):
	os.mkdir(dataset_dir)
	generate_dataset(dataset_dir,320,img_list_2,label_list_2)
	print "create dataset s2..."
else:
	print "dataset s2 exists, pass!"

#dataset s1s2
stat=generate_stat(label_list_1+label_list_2)
print "stage1&stage2 rate: ",np.array(stat)*1.0/np.min(stat[np.nonzero(stat)])
dataset_dir=os.path.join(ProjectDir,"dataset/stage1&stage2-train")
if(not os.path.exists(dataset_dir)):
	os.mkdir(dataset_dir)
	print "create dataset s1s2..."
	generate_dataset(dataset_dir,320,img_list_1+img_list_2,label_list_1+label_list_2)
else:
	print "dataset s1s2 exists, pass!"

#dataset s1s2-crf
stat=generate_stat(label_list_1+label_crf_list_2)
print "crf2 stage1&stage2 rate: ",np.array(stat)*1.0/np.min(stat[np.nonzero(stat)])
dataset_dir=os.path.join(ProjectDir,"dataset/stage1&stage2-train-crf2") #[4 4 6 1 1]
if(not os.path.exists(dataset_dir)):
	os.mkdir(dataset_dir)
	print "create dataset s1s2-crf2..."
	generate_dataset(dataset_dir,320,img_list_1+img_list_2,label_list_1+label_crf_list_2)#{0: 654, 2: 568, 1: 499, 4: 91, 3: 68})
else:
	print "dataset s1s2-crf2 exists, pass!"
