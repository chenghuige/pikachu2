import os

class Config():

    def __init__(self, flag):
        self.n_classes = 8
        self.get_attr(flag)

    def get_attr(self,flag):

        if flag.lower() == 'train':
            self.epoch = 200
            self.batch_size = 16
            self.lr = 0.002
            self.train_data_path = 'data/train/image'
            self.train_label_path = 'data/train/label'
            self.val_data_path = 'data/val/image'
            self.val_label_path = 'data/val/label'
            self.data_augment = True
            self.weight_path = 'weights/aspp'
            self.train_number = self.get_train_number()
            self.val_number = self.get_val_number()
            self.steps_per_epoch = self.train_number//self.batch_size if self.train_number % self.batch_size ==0 else self.train_number//self.batch_size +1
            self.validation_steps= self.val_number//self.batch_size if self.val_number % self.batch_size ==0 else self.val_number// self.batch_size +1

        if flag.lower() == 'test':
            self.batch_size = 32
            self.data_path = 'data/image_A'
            self.weight_path = 'weights/DenseNet121_DeepLabV3Plus/weights-200-0.4155-0.7409.h5'
            self.output_path= 'results'
            self.image_number = self.get_test_number()
            self.steps = self.image_number//self.batch_size if self.image_number % self.batch_size ==0 else self.image_number//self.batch_size +1

    def check_folder(self,dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def get_train_number(self):
        res = 0
        for dir_entry in os.listdir(self.train_data_path):
            if os.path.isfile(os.path.join(self.train_data_path, dir_entry)):
                res += 1
        return res

    def get_val_number(self):
        res = 0
        for dir_entry in os.listdir(self.val_data_path):
            if os.path.isfile(os.path.join(self.val_data_path, dir_entry)):
                res += 1
        return res

    def get_test_number(self):
        res = 0
        for dir_entry in os.listdir(self.data_path):
            if os.path.isfile(os.path.join(self.data_path, dir_entry)):
                res += 1
        return res






