import json
from subprocess import call



test_file = open('../data/test.json')
for line in test_file:
	test_data = json.loads(line)['images']


for i,x in enumerate(test_data):
	img_url = x['url'][0]
	print(img_url)
	filename = '../data/test_images/%i.jpg'%i
	command = 'wget %s -O %s --tries=1 --timeout=1|| rm -f %s'%(img_url, filename, filename)
	# print (command)
	call(command, shell=True)
