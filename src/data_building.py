import json
import numpy as np
from collections import defaultdict
import os





def build_records(url_file, task_file):
	"""
	"""
	for line in url_file:
		record = json.loads(line)
		images = record['images']
		annotations = record['annotations']
	for t in task_file:
		tasks = json.loads(t)['taskInfo']
	return [images, annotations, tasks]





def build_indices(images, annotations, tasks):
	"""
	"""
	index_images = {}
	for x in images:
		index_images[x['imageId']] = x['url']
	index_annotations = {}
	for y in annotations:
		index_annotations[y['imageId']] = {'taskId': y['taskId'], 'labelId': y['labelId']}
	index_tasks = {}
	for z in tasks:
		index_tasks[z['taskName']] = z['taskId']
	return [index_images, index_annotations, index_tasks]





def get_categories(task_filename):
	task_file =  open(task_filename)
	unique_tasks = defaultdict(list)
	for line in task_file:
		tasks_all = json.loads(line)['taskInfo']

	for task in tasks_all:
		taskName = task['taskName']
		unique_tasks[taskName.split(':')[0]].append(taskName)
	task_file.close()
	return unique_tasks





def task_label_dict(annotations):
	task_label_dict = defaultdict(list)
	for ann in annotations:
		task_label_dict[ann['taskId']].append((ann['imageId'], ann['labelId']))
	return task_label_dict





def get_images_by_task(folder_name, task):
	"""
	"""
	url_file = open('../data/train.json')
	task_file = open('../data/task.json')
	records = build_records(url_file, task_file)
	images =records[0]
	annotations = records[1]
	tasks = records[2]

	indices = build_indices(images, annotations, tasks)
	index_images = indices[0]
	index_annotations = indices[1]
	index_tasks = indices[2]

	task_label_rel = task_label_dict(annotations)

	taskId = index_tasks[task]
	imageIds = []
	labelIds = []
	directory = os.listdir(folder_name)
	images_and_labels = task_label_rel[taskId]
	for img_lbl in images_and_labels:
		file_init = folder_name + img_lbl[0]
		for file in directory:
			if (os.path.join(folder_name, file).startswith(file_init)):
				imageIds.append(file)
				labelIds.append(img_lbl[1])
			else:
				pass
	url_file.close()
	task_file.close()				
	return [imageIds, labelIds]


# print(len(get_images_by_task(url_file, task_file, '../data/train/dress/', 'dress:color')[0]))







