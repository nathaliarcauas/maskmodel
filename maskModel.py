import os
import tensorflow.keras
import numpy as np
import cv2
import json
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import model_from_json
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
np.set_printoptions(suppress=True)

def prep_img(path):
	x = cv2.imread(path)
	im_rgb = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
	dsize = (200, 200)
	img = cv2.resize(im_rgb, dsize)
	img = img.reshape(1,200, 200,3)/255.
	return img

path_weights = '/home/nathalia/test_internship/Model/weights/new_weights_epoch_046_loss_0.05.hdf5'

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(path_weights)
print("Loaded model from disk")
 