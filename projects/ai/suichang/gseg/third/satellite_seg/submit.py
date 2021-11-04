import numpy as np
import skimage
import skimage.io as io
import pandas as pd
from tqdm import tqdm
from utils.preprocess import encode_segmap,segmap
import sys,os

columns=["ret"]
def load_predicted_label(ID,label_path):
	labels = io.imread(label_path)
	num_rows = labels.shape[0] * labels.shape[1]
	test_preds = np.zeros((num_rows),np.uint8)

	for i in range(labels.shape[1]):
		for j in range(labels.shape[0]):
			test_preds[i*labels.shape[0]+j] = labels[j][i]

	ids = [ID for i in range(num_rows)]
	return ids,test_preds
def replace_zeros(predmap,replace_val):
	predmap[predmap==0] = replace_val

def generate_csv(result_path_dir,use_crf=False):
	#encode
	for i in tqdm(range(3)):
		prefix = i + 1
		if(use_crf):
			image_path = os.path.join(result_path_dir,"vis_test_%d_post.png"%(prefix))
			save_path  = os.path.join(result_path_dir,"test_%s_pred_post.png"%(prefix))
		else:
			image_path = os.path.join(result_path_dir,"vis_test_%d_pred.png"%(prefix))
			save_path  = os.path.join(result_path_dir,"test_%s_pred.png"%(prefix))

		img = io.imread(image_path)
		encode_mask = encode_segmap(img)

		if(use_crf==False):
			encode_mask[encode_mask==0] = 1 #replace with plant 
			color_mask = segmap(encode_mask)
			io.imsave(os.path.join(result_path_dir,"test_%s_pred_replace.png"%(prefix)),color_mask)

		io.imsave(save_path,encode_mask)
		idx,pred = load_predicted_label(prefix,save_path)
		submission = pd.DataFrame(pred,columns=columns)
		submission.insert(0,'ID',idx)
		submission.to_csv(os.path.join(result_path_dir,'%d.csv'%(prefix)),index=False)

def generate_csv_stage2(result_path_dir,use_crf=False,use_replace=False):
	for i in tqdm(range(3)):
		prefix = i + 1
		if(use_crf):
			image_path = os.path.join(result_path_dir,"vis_test_%d_post.png"%(prefix))
			save_path  = os.path.join(result_path_dir,"test_%s_pred_post.png"%(prefix))
		else:
			image_path = os.path.join(result_path_dir,"vis_test_%d_pred.png"%(prefix))
			save_path  = os.path.join(result_path_dir,"test_%s_pred.png"%(prefix))

		img = io.imread(image_path)
		encode_mask = encode_segmap(img)

		if(use_crf==False and use_replace==True):
			encode_mask[encode_mask==0] = 1 #replace with plant 
			color_mask = segmap(encode_mask)
			io.imsave(os.path.join(result_path_dir,"test_%s_pred_replace.png"%(prefix)),color_mask)

		io.imsave(save_path,encode_mask)
		idx,pred = load_predicted_label(prefix,save_path)
		submission = pd.DataFrame(np.reshape(pred,(1,-1)))
		submission.to_csv(os.path.join(result_path_dir,'%d.csv'%(prefix)),index=False,header=False)


if __name__=='__main__':

	if(len(sys.argv)<2):
		result_dir = '.'
		use_crf = False
	else:
		result_dir = sys.argv[1]
		use_crf = int(sys.argv[2])
	#generate zipfile and remove csv file
	generate_csv_stage2(result_dir,use_crf=use_crf,use_replace=False)
