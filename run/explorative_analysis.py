import numpy as np
import matplotlib.pyplot as plt

from glob import glob
import cv2

from sklearn.manifold import TSNE

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model, load_model

import sys
sys.path.append('../src/')

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

global base_model, model

#==============================================================================

def plot_samples(filenames, nrow=5, ncol=10):
    f, axarr = plt.subplots(nrow, ncol, sharex=True, sharey=True, figsize=(2*ncol,2*nrow))
    f.subplots_adjust(wspace=0.02, hspace=0.02)
    for i in range(nrow):
        for j in range(ncol):
            axarr[i,j].imshow(cv2.resize(cv2.imread(filenames[i*ncol+j]), (224,224)))


def dimension_reduction(features_arr):
	dim_obj = TSNE(n_components = 2)
	tsne_features = dim_obj.fit_transform(features_arr)
	return tsne_features

#============================================================================

def extract_pixel_features(categories):
	"""
	Extracts the RGB features from the data. 

	categories = ['shoe', 'dress', 'outerwear', 'pants']
	"""
	features_pixels = []
	labels = []
	for category in categories:
		dir_name = '../data/train/' + category + '/*'
		filenames = glob(dir_name) 
		for f in filenames:
			print(f)
			img = cv2.resize(cv2.imread(f), (150,150))
			features_pixels.append(np.ndarray.flatten(img))
			labels.append(category)
	return np.array(features_pixels), np.array(labels)




#==============================================================================

base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer("fc1").output)


def get_vgg_features(img_path):
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	features = model.predict(x)
	return np.ndarray.flatten(features)

def extract_vgg_features(categories):
	"""
	"""
	print (categories)
	vgg19_features = []
	labels = []
	for category in categories:
		dir_name = '../data/train/' + category + '/*'
		filenames = glob(dir_name) 
		for f in filenames:
			print(f)
			feats = get_vgg_features(f)
			vgg19_features.append(feats)
			labels.append(category)
	return np.array(vgg19_features), np.array(labels)


#==============================================================================

# finetuned model initialize
layer_name = 'block5_pool'
base_model1 = load_model('../models/finetuned_vgg16.hdf5')
finetuned_model = Model(inputs=base_model1.input, \
			outputs=base_model1.get_layer(layer_name).output)


def get_finetuned_features(img_path):
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	features = finetuned_model.predict(x)
	return np.ndarray.flatten(features)

def extract_finetuned_features(categories):
	finetuned_features = []
	labels = []
	for category in categories:
		dir_name = '../data/train/' + category + '/*'
		filenames = glob(dir_name) 
		for f in filenames:
			print(f)
			feats = get_finetuned_features(f)
			finetuned_features.append(feats)
			labels.append(category)
	return np.array(finetuned_features), np.array(labels)


#===============================================================================

categories = ['shoe', 'dress', 'outerwear', 'pants']

rgb_features, rgb_labels = extract_pixel_features(categories)
np.save('../models/rgb_features.npy', rgb_features)
np.save('../models/rgb_labels.npy', rgb_labels)


rgb_features = np.load('../models/rgb_features.npy')
print ("RGB features shape", rgb_features.shape)
reduced_rgb = dimension_reduction(rgb_features)
np.save('../models/rgb_tsne.npy', reduced_rgb
print('rgb tsne done...')



vgg16_features, vgg16_labels = extract_vgg_features(categories)
np.save('../models/vgg16_features.npy', vgg16_features)
np.save('../models/vgg16_labels.npy', vgg16_labels)


vgg16_features = np.load('../models/vgg16_features.npy')
print ("VGG16 features shape", vgg16_features.shape)
reduced_vgg = dimension_reduction(vgg16_features)
np.save('../models/vgg16_tsne.npy', reduced_vgg)
print('vgg tsne done...')



finetuned_features, finetuned_labels = extract_finetuned_features(categories)
np.save('../models/finetuned_features.npy', finetuned_features)
np.save('../models/finetuned_labels.npy', finetuned_labels)


finetuned_features = np.load('../models/finetuned_features.npy')
print ("Finetuned features shape", finetuned_features.shape)
reduced_finedtuned = dimension_reduction(finetuned_features)
np.save('../models/finedtuned_tsne.npy', reduced_finedtuned)
print('finetuned tsne done...')


#==============================================================================
