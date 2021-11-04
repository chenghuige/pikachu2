from preprocess import convert_label_to_vis,convert_vis_to_label,ProjectDir
import os
import skimage.io as io
import numpy as np

training_data_stage1_dir=os.path.join(ProjectDir,"dataset/CCF-training")
training_data_stage2_dir=os.path.join(ProjectDir,"dataset/CCF-training-Semi")

def convert_stage1_to_stage2(src_path,dst_path):
	label = io.imread(src_path)
	for i in range(label.shape[0]):
		for j in range(label.shape[1]):
			if(label[i][j]==2):
				label[i][j]=4
			elif(label[i][j]==3):
				label[i][j]=2
			elif(label[i][j]==4):
				label[i][j]=3
	io.imsave(dst_path,label)

#Note! no need for this step, the given datasets have been convert

##convert label from stage1 to stage 2
#convert_stage1_to_stage2(os.path.join(training_data_stage1_dir,'1_class.png'),os.path.join(training_data_stage1_dir,'stage1_1_class.png'))
#convert_stage1_to_stage2(os.path.join(training_data_stage1_dir,'2_class.png'),os.path.join(training_data_stage1_dir,'stage1_2_class.png'))
##save visulize label for stage 1

convert_label_to_vis(os.path.join(training_data_stage1_dir,'1_class.png'),os.path.join(training_data_stage1_dir,'1_class_vis.png'))
convert_label_to_vis(os.path.join(training_data_stage1_dir,'2_class.png'),os.path.join(training_data_stage1_dir,'2_class_vis.png'))

convert_label_to_vis(os.path.join(training_data_stage2_dir,'1_class.png'),os.path.join(training_data_stage2_dir,'1_class_vis.png'))
convert_label_to_vis(os.path.join(training_data_stage2_dir,'2_class.png'),os.path.join(training_data_stage2_dir,'2_class_vis.png'))
convert_label_to_vis(os.path.join(training_data_stage2_dir,'3_class.png'),os.path.join(training_data_stage2_dir,'3_class_vis.png'))
