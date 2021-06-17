from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.python.keras.applications.resnet import ResNet50, preprocess_input
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import glob
from tensorflow.python.keras.preprocessing.image import load_img
from tensorflow.python.keras.preprocessing.image import img_to_array
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from imutils import paths
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
from imutils import paths
import seaborn as sns
from tensorflow.keras.models import load_model


class TestModel():
    def __init__(self):
        self.test_path = str(os.getcwd()) + '/' + 'testdata'
        self.model = load_model(str(os.getcwd()) + '/' + 'model_binary_right_left_retina_resnet50_53k.h5')

    def single_image(self):
        img_data__path = list(paths.list_images(self.test_path))
        pred_list = []
        label_list = []
        img_count=0
        for path in img_data__path:
            label = path.split(os.path.sep)[-2]
            if 'left' in label:
                label_list.append(0)
            elif 'right' in label:
                label_list.append(1)
            original = load_img(path, target_size=(224, 224))
            numpy_image = img_to_array(original)
            image_batch = np.expand_dims(numpy_image, axis=0)
            processed_image = preprocess_input(image_batch.copy())
            prediction_prob = self.model.predict(processed_image)
            prediction = prediction_prob.argmax(axis=1)
            pred_list.append(prediction)

            if prediction == 1:
            	lbl_text = 'right'
            else:
            	lbl_text='left'

            if lbl_text != label:
            	print(prediction_prob)
            	# print("detected")
            	# cv2.imshow('raw img',np.array(original))
            	# cv2.waitKey(0)
            	cv2.imwrite(str(os.getcwd())+'/missclass/' + 'Image ' + str(img_count) +' predicted ' + 
	            	str(lbl_text) + 'Actual ' + str(label) + '.png', np.array(original))
            	img_count += 1

            

        pred_array = np.array(pred_list)
        predict = pred_array.reshape(-1, 1)
        img_label = np.array(label_list)

        print(".........Using single_image.........")
        print(classification_report(img_label, predict))

        cm = confusion_matrix(img_label, predict, )
        print(cm)
        labels = ['left','right']
        ax = plt.subplot()
        sns.heatmap(cm, annot=True, ax=ax, fmt='d')

        # labels, title and ticks
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(labels)
        ax.yaxis.set_ticklabels(labels)
        ax.figure.savefig('confusion matrix.png')
        plt.close()


if __name__ == '__main__':
    obj = TestModel()
    obj.single_image()
