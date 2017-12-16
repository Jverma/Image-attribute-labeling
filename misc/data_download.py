import numpy as np 
import matplotlib.pyplot as plt 

import json
import urllib

from data_building import *
from subprocess import call

#==============================================================================

def read_image(url, timeout=1):
	return urllib.request.urlopen(url, timeout=timeout).read()

#------------------------------------------------------------------------------

def write_image(url, filename, timeout=1):
	pic = read_image(url, timeout)
	f = open(filename, 'w')
	f.write(pic)
	f.close()
	return pic

#------------------------------------------------------------------------------

def get_image_info(imageID):
	global records
	images =records[0]
	annotations = records[1]
	tasks = records[2]

	indices = build_indices(images, annotations, tasks)
	index_images = indices[0]
	index_annotations = indices[1]
	index_tasks = indices[2]

	image_url = index_images[imageID]
	ann = index_annotations[imageID]['taskId']
	taskName = filter(lambda x: x['taskId'] == ann, tasks)
	category = list(taskName)[0]['taskName'].split(':')[0]
	# url_file.close()
	# task_file.close()
	return image_url,category

#==============================================================================


url_file = open('../data/train.json')
task_file = open('../data/task.json')
records = build_records(url_file, task_file)

num_images = 1000
for i in range(1000, 2*num_images):
	image_url, category = get_image_info('%i'%i)
	filename = '../data/testset/'+category+'/%i.jpg'%i
	try:
		command = 'wget %s -O %s --tries=1 --timeout=1|| rm -f %s'%(image_url[1], filename, filename)
		print (command)
		call(command, shell=True)
	except:
		pass

url_file.close()
task_file.close()




