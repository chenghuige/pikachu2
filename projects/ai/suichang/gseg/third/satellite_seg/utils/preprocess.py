import skimage.io as io
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter

ProjectDir="/home/lab-xiong.jiangfeng/Projects/satellite_seg"

def get_color_labels():
	return np.asarray([[0,0,0], [128,0,0], [0,128,0], [128,128,0], [0,0,128],[255,255,255]])

def encode_segmap(mask):
	mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for i, label in enumerate(get_color_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = i
        label_mask = label_mask.astype(int)
        return label_mask

def auto_stride(patch_label):
#x0:x1:x2:x3:x4
	#most_count_label= Counter(patch_label.flatten().tolist()).most_common(1)[0][0]
	most_count_label = np.argmax(np.bincount(patch_label.flatten().tolist()))
	if(most_count_label==0):
		stride = 256
	elif(most_count_label==1):
		stride = 32 #oversampling the most class in the testset(plant)
	elif(most_count_label==2):
		stride = 32
	elif(most_count_label==3):
		stride = 256
	elif(most_count_label==4):
		stride = 32
	else:
		print "Unknown label"
	return stride

def segmap(mask):
	label_colours = get_color_labels()
	r = mask.copy()
	g = mask.copy()
	b = mask.copy()
	for l in range(0, label_colours.shape[0]):
		r[mask == l] = label_colours[l, 0]
 		g[mask == l] = label_colours[l, 1]
		b[mask == l] = label_colours[l, 2]

		rgb = np.zeros((mask.shape[0], mask.shape[1], 3),dtype=np.uint8)
		rgb[:, :, 0] = r
		rgb[:, :, 1] = g
		rgb[:, :, 2] = b
	return rgb
def save_visual_gt(save_dir,mask,prefix,index):
	color_mask = segmap(mask)
	io.imsave(os.path.join(save_dir,prefix+"_%05d.png"%(index)),color_mask)

def crop(image_path,img_class_path,crop_size_w,crop_size_h,prefix,save_dir,crop_label=False):
	raw_img = io.imread(image_path,dtype=np.uint8)
	raw_img_class = io.imread(img_class_path,dtpe=np.uint8)
	h,w = raw_img.shape[0],raw_img.shape[1]
	#stride_h = 15
	#stride_w = 15
	#n_rows = int(np.ceil((h - crop_size_h)/stride_h)) + 1
	#n_cols = int(np.ceil((w - crop_size_w)/stride_w)) + 1
	index = 0
	x2,y2 = 0,0
	x0,y0 = 0,0
	while(y2<h):
		while(x2<w):
			x1 = x0
			x2 = x1 + crop_size_w
			y1 = y0
			y2 = y1 +crop_size_h

			print x1,y1,x2,y2

			if(x2>w or y2>h):
				x2 = min(x2,w)
				y2 = min(y2,h)
				if((x2-x1)>10 and (y2-y1)>10):
					backgroud = np.zeros((crop_size_h,crop_size_w,raw_img.shape[2]),dtype=np.uint8)
					backgroud[:y2-y1,:x2-x1] = raw_img[y1:y2,x1:x2]
					patch = backgroud

					backgroud_label = np.zeros((crop_size_h,crop_size_w),dtype=np.uint8)
					backgroud_label[:y2-y1,:x2-x1] = raw_img_class[y1:y2,x1:x2]
					patch_label = backgroud_label
				else:
					break
			else:
				patch = raw_img[y1:y2,x1:x2]
				patch_label = raw_img_class[y1:y2,x1:x2]
			#stride_h = auto_stride(patch_label)
			stride_h = crop_size_h
			stride_w = crop_size_w
			#print "current stride: ",stride_h
			x0 = x1 + stride_w

			io.imsave(os.path.join(save_dir,'img',prefix+"_%05d.png"%(index)),patch)
			io.imsave(os.path.join(save_dir,'label',prefix+"_%05d.png"%(index)),patch_label)
			save_visual_gt(os.path.join(save_dir,'visualize_gt'),patch_label,prefix,index)
			index = index + 1
		x0,x1,x2 = 0,0,0
		y0 = y1 + stride_h


def generate_trainval_list(pathdir):
	labels_img_paths = os.listdir(os.path.join(pathdir,'label'))
	labels_count_list=dict()
	for labels_img_path in tqdm(labels_img_paths):
		label = io.imread(os.path.join(pathdir,'label',labels_img_path))
		most_count_label= np.argmax(np.bincount(label.flatten().tolist()))
		labels_count_list[labels_img_path] = most_count_label
	values= labels_count_list.values()
	count_dict= Counter(values)
	print count_dict


def write_train_list(pathdir):
	labels_img_paths = os.listdir(os.path.join(pathdir,'label'))
	num_sets = len(labels_img_paths)
	indexs = range(num_sets)
	np.random.shuffle(indexs)
	train_set_num = 0.95 * num_sets
	train_f = open(os.path.join(pathdir,'train.txt'),'w')
	val_f = open(os.path.join(pathdir,'val.txt'),'w')
	trainval_f = open(os.path.join(pathdir,'trainval.txt'),'w')
	for index in range(num_sets):
		if(index<train_set_num):
			print >>train_f,labels_img_paths[indexs[index]]
		else:
			print >>val_f,labels_img_paths[indexs[index]]
		print >>trainval_f,labels_img_paths[indexs[index]]
	train_f.close()
	val_f.close()
	trainval_f.close()

def save_gt_vis(input_path,output_path):
	raw_img_class = io.imread(input_path,dtpe=np.uint8)
	color_mask = segmap(raw_img_class)
	io.imsave(output_path,color_mask)

def convert_label_to_vis(src_path,dst_path):
	label = io.imread(src_path)
	io.imsave(dst_path,segmap(label))
	
def convert_vis_to_label(src_path,dst_path):
	vis = io.imread(src_path)
	io.imsave(dst_path,encode_segmap(vis))
