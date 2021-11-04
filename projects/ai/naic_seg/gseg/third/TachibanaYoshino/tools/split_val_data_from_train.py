import os
import random
random.seed(1)
import shutil
from tqdm import tqdm


def check_folder(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def move_file(old_path, new_path):
    print(old_path)
    print(new_path)
    filelist = os.listdir(old_path) #列出该目录下的所有文件,listdir返回的文件列表是不包含路径的。
    print(filelist)
    for file in filelist:
        src = os.path.join(old_path, file)
        dst = os.path.join(new_path, file)
        shutil.move(src, dst)

# get val data from train data
def split(dir):
    data =[]
    for x in os.listdir(dir):
        if x.endswith('tif'):
            data.append(os.path.join(dir,x))
    return data

if __name__=='__main__':
    train_dir = '../data/train/image'
    train_label_dir = '../data/train/label'
    val_dir = '../data/val/image'
    val_label_dir = '../data/val/label'
    check_folder(val_label_dir)
    check_folder(val_dir)
    val_ratio = 0.1
    res = split(train_dir)
    random.shuffle(res)
    print(len(res))
    print(res)
    for i in tqdm(range(int(len(res) * val_ratio))):
        shutil.move(res[i],res[i].replace(train_dir,val_dir)) # tif images
        # png images
        shutil.move(res[i].replace(train_dir,train_label_dir).replace('tif','png'),res[i].replace(train_dir,val_label_dir).replace('tif','png'))


