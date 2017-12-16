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




task_filename = '../data/task.json'
categories_dict = DB.get_categories(task_filename)
categories = list(categories_dict.keys())


print (categories)
for i in range(len(categories)):
	foldername = '../data/validation/'+categories[i]+'/'
	for j in range(len(categories_dict[categories[i]])):
		taskname = categories_dict[categories[i]][j]
		savemodelname = taskname.replace(':','_')+'.sav'
		savemodelname = '../models/'+ savemodelname.replace(' ', '_')
		print (foldername, taskname, savemodelname)
		obj.train_simple_model(foldername, taskname, savemodelname)
	
#==============================================================================