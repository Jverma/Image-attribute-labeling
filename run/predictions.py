import numpy as np 
import sys

sys.path.append('../src/')



import cv2

import data_building as DB
from TrainTaskModels import TrainModels as TM

from keras.models import load_model, Model

import data_building as DB

import pickle
import os
import json

from collections import defaultdict

#==============================================================================

global base_model, finetuned_model, layer_name, categories_dict

layer_name = 'block5_pool'
base_model = load_model('../models/finetuned_vgg16.hdf5')
finetuned_model = Model(inputs=base_model.input, \
			outputs=base_model.get_layer(layer_name).output)


task_filename = '../data/task.json'
categories_dict = DB.get_categories(task_filename)
categories = list(categories_dict.keys())

#============================================================================

def get_image_features(img_path):
	img = cv2.resize(cv2.imread(img_path), (224,224))
	img = np.expand_dims(img, axis=0)
	img = img/255.0
	category = base_model.predict(img)
	category = np.argmax(category)
	features = finetuned_model.predict(img)
	return category, np.ndarray.flatten(features)


#===========================================================================



def predict_class(img_path):
	category_id, features = get_image_features(img_path)
	category= categories[category_id]
	print(category)
	tasks = categories_dict[category]
	predictions = {}
	for task in tasks:
		model_path = '../models/' + task.replace(':', '_') + '.sav'
		model_path = model_path.replace(' ', '_')
		saved_model = pickle.load(open(model_path, 'rb'))
		predictions[task] = saved_model.predict(features)
	return predictions


#==========================================================================

task_filename = '../data/task.json'
categories_dict = DB.get_categories(task_filename)
categories = list(categories_dict.keys())


predict_dict = defaultdict(list)
for i in range(len(categories)):
	foldername = '../data/testset/'+categories[i]+'/'
	directory = os.listdir(foldername)
	for file in directory:
		f = os.path.join(foldername, file)
		try:
			pred = predict_class(f)
			predict_dict[categories[i]].append(pred)
		except:
			pass

res = json.dumps(predict_dict)
f = open('validation_pred.json')
f.write(res)
f.close()



