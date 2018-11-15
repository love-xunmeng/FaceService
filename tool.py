# -*- coding: utf-8 -*-

import sys
import json
import urllib2
import time
import base64
import os
import json
import cv2
import v4l2capture
import select
from PIL import Image
import datetime
import numpy as np
import requests

def count_names(root_dir):
	file_name_set = os.listdir(root_dir)
	file_name_set.sort()
	name_set = []
	for file_name in file_name_set:
		name = file_name.split('_')[0]
		if name not in name_set:
			name_set.append(name)
	print(len("\n"))
	print("len(names): " + str(len(name_set)))
	print(name_set)

def register_one_face():
	register_image_name = 'caiwenjing_10002.jpg'
	image_path = os.path.join('./test_images/recognization-set2/register/' + register_image_name)
	url = "http://192.168.0.27:5000/face/register"
	with open(image_path, 'rb') as f:
		base64_data = base64.b64encode(f.read())
	base64_data_string = base64_data.decode()
	json_data = {}
	json_data['image_name'] = register_image_name
	json_data['image_data'] = base64_data_string
	headers = {'Content-Type':'application/json'}
	data = json.dumps(json_data)
	req = urllib2.Request(url, headers=headers, data=data)
	resp = urllib2.urlopen(req).read().decode()
	resp_data = json.loads(resp)
	print(resp_data)


#register_one_face()


print("\n")
count_names('./test_images/recognization-set2/register')
print("\n")
count_names('./test_images/recognization-set2/test')