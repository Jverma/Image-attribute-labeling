import numpy as np 
import matplotlib.pyplot as plt 
import cv2 

from keras.models import load_model, Model
from keras.layers import Input, Dense, Flatten

import data_building as DB
from keras.utils import to_categorical

from sys import exit

from sklearn.neighbors import KNeighborsClassifier

import pickle
#==============================================================================

class TrainModels(object):
	def __init__(self, base_model, layer_name):
		self.model_finetuned_vgg16 = load_model(base_model)
		# print (model_finetuned_vgg16.summary())

		self.model_convpart = Model(inputs=self.model_finetuned_vgg16.input, \
			outputs=self.model_finetuned_vgg16.get_layer(layer_name).output)

		self.clf = KNeighborsClassifier(n_neighbors=10)
		print (self.model_convpart.summary())

#------------------------------------------------------------------------------

	def get_data(self, foldername, taskname):
		[filenames, labels] = DB.get_images_by_task(foldername, taskname)
		# print (len(filenames), len(labels))
		x_train = []
		for i in range(len(filenames)):
			try:
				img = cv2.resize(cv2.imread(foldername+filenames[i]), (224,224))
				x_train.append(img)
			except:
				del labels[i]
		return np.array(x_train), np.array(labels)

#------------------------------------------------------------------------------

	def get_features(self, foldername, taskname):
		x_train, labels = self.get_data(foldername, taskname)
		features = self.model_convpart.predict(x_train/255., batch_size=10, verbose=1)
		return features, labels

#------------------------------------------------------------------------------

	def get_image_features(self, img_path):
		img = cv2.resize(cv2.imread(img_path), (224,224))
		img = np.expand_dims(img, axis=0)
		category = self.model_finetuned_vgg16.predict(img)
		category = np.argmax(category)
		features = self.model_convpart.predict(img)
		return category, np.ndarray.flatten(features)

#------------------------------------------------------------------------------

	def get_model(self, nclass=2, filters=[32,32,32]):
		input_img = Input(shape=self.model_convpart.output_shape[1:])
		x = Flatten()(input_img)
		x = Dense(filters[0], activation='relu')(x)
		for i in range(1, len(filters)):
			x = Dense(filters[i], activation='relu')(x)
		x = Dense(nclass, activation='sigmoid')(x)
		mymodel = Model(input_img, x)
		return mymodel

#------------------------------------------------------------------------------

	def train_model(self, foldername, taskname, filters=[64, 64], \
						loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"], \
						batch_size=10, epochs=10, validation_split=0.2, \
						savemodel=False, savemodelname='model.hdf5'):
		features, labels = self.get_features(foldername, taskname)
		print ("Feature shape: ", features.shape, labels.shape)

		unique_ids = list(set(list(labels)))
		print ("Unique IDs: ", unique_ids)

		labels_dict = {}
		for i, j in enumerate(unique_ids):
			labels_dict[j] = i

		true_labels = np.array([labels_dict[x] for x in labels])

		nclasses = len(set(list(labels)))
		print ("Number of classes: ", nclasses)

		if (nclasses > 1):
			y_train = to_categorical(true_labels, num_classes=nclasses)
			print ("Shape of categorial labels", y_train.shape)

			mymodel = self.get_model(nclass=nclasses, filters=filters)
			mymodel.compile(loss=loss, optimizer=optimizer, metrics=metrics)
			print (mymodel.summary())
			hist = mymodel.fit(features, y_train, batch_size=batch_size, \
				epochs=epochs, validation_split=validation_split)
			if savemodel:
				mymodel.save(savemodelname)
			return hist
		else:
			pass

#==============================================================================


	def train_simple_model(self, foldername, taskname, filename):
		features, labels = self.get_features(foldername, taskname)
		print ("Feature shape: ", features.shape, labels.shape)

		features = np.array([np.ndarray.flatten(x) for x in features])

		self.clf.fit(features, labels)
		print('knn model trained....')

		pickle.dump(self.clf, open(filename, 'wb'))
		print('knn model saved')

 

#============================================================================================#
if __name__=="__main__":
	base_model = '../models/finetuned_vgg16.hdf5'
	layer_name = 'block5_pool'
	obj = TrainModels(base_model, layer_name)

	foldername = '../data/train/dress/'
	taskname = ['dress:decoration', 'dress:color']
	for i in range(len(taskname)):
		obj.train_model(foldername, taskname[i])

