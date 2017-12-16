import numpy as np 
import matplotlib.pyplot as plt 

from sys import exit
import sys
sys.path.append('../src/')

import data_building as DB
from TrainTaskModels import TrainModels as TM

#==============================================================================

base_model = '../models/finetuned_vgg16.hdf5'
layer_name = 'block5_pool'
obj = TM(base_model, layer_name)

# exit()

filters=[64, 64]
loss="categorical_crossentropy"
optimizer="adam"
metrics=["accuracy"]
batch_size=10
epochs=100
validation_split=0.2

#==============================================================================

task_filename = '../data/task.json'
categories_dict = DB.get_categories(task_filename)
categories = list(categories_dict.keys())

print (categories)
for i in range(len(categories)):
	foldername = '../data/train/'+categories[i]+'/'
	for j in range(len(categories_dict[categories[i]])):
		taskname = categories_dict[categories[i]][j]
		savemodelname = taskname.replace(':','_')+'.hdf5'
		savemodelname = '../models/'+savemodelname.replace(' ', '_')
		print (foldername, taskname, savemodelname)
		obj.train_model(foldername, taskname, filters=filters, \
				loss=loss, optimizer=optimizer, metrics=metrics, \
				batch_size=batch_size, epochs=epochs, \
				validation_split=validation_split, \
				savemodel=True, savemodelname=savemodelname)
		
#==============================================================================