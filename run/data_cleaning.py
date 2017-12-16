import cv2
import os

from sys import argv


def remove_bad_files(foldername):
	directory = os.listdir(foldername)
	print("Total number of files: ", len(directory))
	for file in directory:
		f = os.path.join(foldername, file)
		try:
			img = cv2.resize(cv2.imread(f), (224,224))
		except:
			os.remove(f)
	return None



foldername = argv[1]
remove_bad_files(foldername)
