# -- coding: utf-8 --
import os
from tensorflow.keras.callbacks import ModelCheckpoint

class GhostTrainer(object):
    def __init__(self,model,data,batchsizes,epochs):
        self.model = model
        self.data = data
        self.batchsizes = batchsizes
        self.epochs = epochs

    def train(self):
        x_train = self.data[0][0]
        y_train = self.data[0][1]

        print("training ==========~~~~~~~~=======")

        modelpath = 'weight/GhostNet_model.h5'
        checkpoint = ModelCheckpoint(modelpath,monitor='val_acc',verbose=1,save_best_only=True,save_weights_only=True,mode='max')
        callback_list = [checkpoint]

        self.model.fit(
            x_train, y_train,
            batch_size=self.batchsizes,
            epochs=self.epochs,
            verbose=1,
            callbacks=callback_list,
            validation_split=0.2)

        #self.model.save(os.path.join("weight", "GhostNet_model.h5"))
        #self.model.save_weights(os.path.join("weight", "GhostNet_model.h5"))

    def test(self):
        x_test = self.data[1][0]
        y_test = self.data[1][1]

        print("Testing ==========~~~~~~~~~~~~======")
        loss, accuracy = self.model.evaluate(x_test, y_test)
        print("loss:",loss)
        print("accuracy:",accuracy)
