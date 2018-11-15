import json
from urllib import request
import time
import base64
import os
import json
import cv2
import redis
import requests

redis_cli = redis.Redis(host='localhost', port=6379, decode_responses=True)

def test_face_register_no_face():
	resp_data = request_face_register('0-face.jpg')
	face_count = resp_data['face_count']
	assert(0 == face_count)

def test_face_register_only_one_face():
	image_name = '1-face.jpg'
	resp_data = request_face_register(image_name)
	face_count = resp_data['face_count']
	assert(1 == face_count)

	test_origial_image_is_saved(image_name, resp_data)
	test_face_image_is_saved(image_name, resp_data)
	test_face_feature_is_saved(image_name, resp_data)
	test_is_registered_to_redis(image_name, resp_data)
	draw_bounding_box(image_name, resp_data)

def test_face_register_multi_faces():
	image_name = '3-faces.jpg'
	resp_data = request_face_register(image_name)
	face_count = resp_data['face_count']
	assert(0 == face_count)

def test_origial_image_is_saved(image_name, resp_data):
	face_id = resp_data['face_id']
	server_save_image_path = os.path.join('images', image_name)
	assert(os.path.exists(server_save_image_path))

def test_face_image_is_saved(image_name, resp_data):
	face_id = resp_data['face_id']
	server_save_face_image_path = os.path.join('register_faces', image_name.split('.')[0] + '_' + face_id + '.png')
	assert(os.path.exists(server_save_face_image_path))

def test_face_feature_is_saved(image_name, resp_data):
	face_id = resp_data['face_id']
	server_save_face_feature_path = os.path.join('register_faces', image_name.split('.')[0] + '_' + face_id + '.fea')
	assert(os.path.exists(server_save_face_feature_path))

def test_is_registered_to_redis(image_name, resp_data):
	global redis_cli
	feature_key = image_name.split('.')[0] + '_' + resp_data['face_id']
	assert(redis_cli.exists(feature_key))

def draw_bounding_box(image_path, resp_data):
	img = cv2.imread(os.path.join('test_images', image_name))
	left_up_x = resp_data['x']
	left_up_y = resp_data['y']
	right_down_x = resp_data['x'] + resp_data['width']
	right_down_y = resp_data['y'] + resp_data['height']
	cv2.rectangle(img, (left_up_x, left_up_y), (right_down_x, right_down_y), (255,0,0), 6)
	cv2.imwrite(os.path.join('tmp', image_name), img)

def request_face_register(image_name):
	url = "http://192.168.0.27:5000/face/register"
	image_path = os.path.join("test_images", image_name)
	with open(image_path, 'rb') as f:
		base64_data = base64.b64encode(f.read())
	base64_data_string = base64_data.decode()
	json_data = {}
	json_data['image_name'] = image_name
	json_data['image_data'] = base64_data_string
	headers = {'Content-Type':'application/json'}
	data = bytes(json.dumps(json_data), 'utf8')
	start_time = time.time()
	req = request.Request(url, headers=headers, data=data, method='POST')
	try:
		resp = request.urlopen(req)
	except urllib.error.URLError as e:
		print("exception: ")
		print(e.code)
		print(e.reason)
	end_time = time.time()
	elapsed_time = end_time - start_time
	print("elapsed time: %ds", elapsed_time)
	resp_data = json.loads(resp.read().decode())
	return resp_data

def test_face_register():
	test_face_register_no_face()
	test_face_register_only_one_face()
	test_face_register_multi_faces()

def test_detect():
	url = "http://192.168.0.30:5000/face/detect_by_image_path"
	src_dir = os.path.join('test_images', 'detect')
	image_name_set = os.listdir(src_dir)
	for image_name in image_name_set:
		image_path = os.path.join(src_dir, image_name)
		with open(image_path, 'rb') as f:
			base64_data = base64.b64encode(f.read())
		base64_data_string = base64_data.decode()
		json_data = {}
		json_data['image_name'] = image_name
		json_data['image_data'] = base64_data_string
		headers = {'Content-Type':'application/json'}
		data = bytes(json.dumps(json_data), 'utf8')

		start_time = time.time()

		req = request.Request(url, headers=headers, data=data, method='POST')
		resp = request.urlopen(req)
		resp_data = json.loads(resp.read().decode())

		elapsed_time = time.time() - start_time
		print("\n")
		print("elapsed_time: %dms" %(int(round(elapsed_time * 1000))))

		img = cv2.imread(os.path.join(src_dir, image_name))
		for i in range(resp_data['face_count']):
			face_node = resp_data['face_list'][i]	
			left_up_x = face_node['x']
			left_up_y = face_node['y']
			right_down_x = face_node['x'] + face_node['width']
			right_down_y = face_node['y'] + face_node['height']
			cv2.rectangle(img, (left_up_x, left_up_y), (right_down_x, right_down_y), (255,0,0), 3)
		cv2.imwrite(os.path.join('tmp', image_name), img)

def test_recognizae():
	url = "http://192.168.0.27:5000/face/recognize"
	file_name = '1-face.jpg'
	files = {
		'file':(file_name, open(file_name, 'rb'))
	}
	response = requests.request('POST', url, files=files)
	#resp_data = json.loads(response.read().decode())
	print(response.text)
	print()
	print(response.json)

def test_face_detect_and_recognize():
	register_image_dir_root = os.path.join('test_images', 'recognization', 'register')
	test_image_dir_root = os.path.join('test_images', 'recognization', 'test')

	'''
	register_image_dir_set = os.listdir(register_image_dir_root)
	register_image_dir_set.sort()
	for register_image_dir in register_image_dir_set:
		register_image_name_set = os.listdir(os.path.join(register_image_dir_root, register_image_dir))
		register_image_name_set.sort()
		for register_image_name in register_image_name_set:
			image_path = os.path.join(os.path.join(register_image_dir_root, register_image_dir, register_image_name))
			url = "http://192.168.0.30:5000/face/register"
			with open(image_path, 'rb') as f:
				base64_data = base64.b64encode(f.read())
			base64_data_string = base64_data.decode()
			json_data = {}
			json_data['image_name'] = register_image_name
			json_data['image_data'] = base64_data_string
			headers = {'Content-Type':'application/json'}
			data = bytes(json.dumps(json_data), 'utf8')
			start_time = time.time()
			req = request.Request(url, headers=headers, data=data, method='POST')
			resp = request.urlopen(req)
			end_time = time.time()
			elapsed_time = end_time - start_time
			print("elapsed time: %ds", elapsed_time)
			resp_data = json.loads(resp.read().decode())
	'''

	image_count = 0
	correct_prediction_count = 0
	test_image_dir_set = os.listdir(test_image_dir_root)
	test_image_dir_set.sort()
	for test_image_dir in test_image_dir_set:
		test_image_name_set = os.listdir(os.path.join(test_image_dir_root, test_image_dir))
		test_image_name_set.sort()
		for test_image_name in test_image_name_set:
			image_path = os.path.join(os.path.join(test_image_dir_root, test_image_dir, test_image_name))
			url = "http://192.168.0.30:5000/face/detect_and_recognize"
			with open(image_path, 'rb') as f:
				base64_data = base64.b64encode(f.read())
			base64_data_string = base64_data.decode()
			json_data = {}
			json_data['image_name'] = test_image_name
			json_data['image_data'] = base64_data_string
			headers = {'Content-Type':'application/json'}
			data = bytes(json.dumps(json_data), 'utf8')
			req = request.Request(url, headers=headers, data=data, method='POST')
			resp = request.urlopen(req)
			resp_data = json.loads(resp.read().decode())
			if resp_data['face_count'] > 1:
				continue
			if 0 != resp_data['face_count']:
				img = cv2.imread(image_path)
				face_list = resp_data['face_list']
				for face_node in face_list:
					'''
					left_up_x = face_node['x']
					left_up_y = face_node['y']
					right_down_x = face_node['x'] + face_node['width']
					right_down_y = face_node['y'] + face_node['height']
					cv2.rectangle(img, (left_up_x, left_up_y), (right_down_x, right_down_y), (255,0,0), 3)
					text = face_node['face_id'] + ":" + str(face_node['score'])
					img_text = cv2.putText(img, text, (left_up_x, left_up_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 2)
					'''
					prediction = face_node['face_id'].split('-')[0]
					if prediction == test_image_dir:
						correct_prediction_count += 1
					image_count += 1
				cv2.imwrite(os.path.join('tmp', test_image_name), img)
	accuracy = correct_prediction_count / image_count
	print()
	print("Accuracy:", accuracy)
	print()


	'''
	image_name_list = ['3.jpg', '10.jpg', '11.jpg']
	for image_name in image_name_list:
		print()
		print(image_name + ":")
		url = "http://192.168.0.30:5000/face/detect_and_recognize"
		image_path = os.path.join("test_images", image_name)
		with open(image_path, 'rb') as f:
			base64_data = base64.b64encode(f.read())
		base64_data_string = base64_data.decode()
		json_data = {}
		json_data['image_name'] = image_name
		json_data['image_data'] = base64_data_string
		headers = {'Content-Type':'application/json'}
		data = bytes(json.dumps(json_data), 'utf8')
		start_time = time.time()
		req = request.Request(url, headers=headers, data=data, method='POST')
		resp = request.urlopen(req)
		end_time = time.time()
		elapsed_time = end_time - start_time
		print("elapsed time: %ds", elapsed_time)
		print(resp.read())
	print("\n")
	'''
				
if __name__ == '__main__':
	test_recognizae()
	#request_face_register("1-face.jpg")

	#test_face_detect_and_recognize()
	#test_face_register()
	#while True:
	#	test_detect()
	#test_face_detect_and_recognize()