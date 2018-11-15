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
import shutil

def validate_detection():
	url = "http://192.168.0.27:5000/face/detect"
	root_dir = os.path.join(os.path.dirname(__file__), 'test_images', 'MingXing')
	dir_set = os.listdir(root_dir)
	dir_set.sort()

	for dir_name in dir_set:
		dir_path = os.path.join(root_dir, dir_name)
		file_name_set = os.listdir(dir_path)
		file_name_set.sort()
		for file_name in file_name_set:
			print("\n")
			file_path = os.path.join(dir_path, file_name)
			print(file_path)
			img = cv2.imread(file_path)
			img_encode = cv2.imencode('.jpg', img)[1]
			base64_data = base64.b64encode(img_encode)
			base64_data_string = base64_data.decode()
			json_data = {}
			json_data['image_name'] = dir_name + "_" + file_name
			json_data['image_data'] = base64_data_string
			headers = {'Content-Type':'application/json'}
			data = json.dumps(json_data)
			                                             
			req = urllib2.Request(url, headers=headers, data=data)
			resp = urllib2.urlopen(req).read().decode()
			resp_data = json.loads(resp)
			#print(resp_data)

			if resp_data['face_count'] == 1:
				result_image_save_path = "./result/1-face/" + dir_name + "_" + file_name
				cv2.imwrite(result_image_save_path, img)
			
			'''
			result_image_save_path = ""
			if resp_data['face_count'] != 1:
				result_image_save_path = "./result/others/" + dir_name + "_" + file_name
			else:
				result_image_save_path = "./result/1-face/" + dir_name + "_" + file_name
			for i in range(resp_data['face_count']):
				face_node = resp_data['face_list'][i]	
				left_up_x = face_node['x']
				left_up_y = face_node['y']
				right_down_x = face_node['x'] + face_node['width']
				right_down_y = face_node['y'] + face_node['height']
				cv2.rectangle(img, (left_up_x, left_up_y), (right_down_x, right_down_y), (255,0,0), 3)
			
			'''

def register_faces():
	register_image_dir_root = "faces/uwonders"

	dir_set = os.listdir(register_image_dir_root)
	dir_set.sort()

	for dir_name in dir_set:
		dir_path = os.path.join(register_image_dir_root, dir_name)
		register_image_name_set = os.listdir(dir_path)
		register_image_name_set.sort()
		for register_image_name in register_image_name_set:
			print(register_image_name)
			image_path = os.path.join(os.path.join(dir_path, register_image_name))
			url = "http://192.168.0.27:5000/face/register"
			with open(image_path, 'rb') as f:
				base64_data = base64.b64encode(f.read())
			base64_data_string = base64_data.decode()
			json_data = {}
			json_data['image_name'] = dir_name + "_" + register_image_name
			json_data['image_data'] = base64_data_string
			headers = {'Content-Type':'application/json'}
			data = json.dumps(json_data)
			req = urllib2.Request(url, headers=headers, data=data)
			resp = urllib2.urlopen(req).read().decode()
			resp_data = json.loads(resp)

def validate_recognition():
	url = "http://192.168.0.27:5000/face/recognize"
	root_dir = './test_images/recognization-set2/test'
	image_name_set = os.listdir(root_dir)
	image_name_set.sort()
	correct_prediction_count = 0
	total = 0
	for image_name in image_name_set:
		image_path = os.path.join(root_dir, image_name)
		files = {'file':open(image_path, 'rb')}
		response = requests.post(url, files=files)
		resp_data = json.loads(response.text)
		if 1 != resp_data['face_count']:
			print("\n@@@@ Error: " + image_name + " @@@@\n")
			continue
		predict_id = resp_data['face_list'][0]['id']
		target_id = image_name.split('_')[0]
		result_path = ""
		if predict_id == target_id:
			correct_prediction_count += 1
			result_path = os.path.join('result/recognition/correct', image_name)
		else:
			print("Mismatch: " + image_name + " " + predict_id + " score=" + str(resp_data['face_list'][0]['score']))
			result_path = os.path.join('result/recognition/wrong', image_name)
		#shutil.copyfile(image_path, result_path)
		total += 1

		if(total % 10 == 0):
			print("total: " + str(total) + ", correct: " +str(correct_prediction_count))

		if total % 500 == 0:
			break

	print("\n")
	print("total images: " + str(len(image_name_set)))
	print("recognize images: " + str(total))
	print("correct recognize images: " + str(correct_prediction_count))
	print("correct ratio: " + str(correct_prediction_count / total))

if __name__ == '__main__':
	#validate_detection()
	register_faces()
	#validate_recognition()
