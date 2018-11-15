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

def detect():
	url = "http://192.168.0.27:5000/face/detect"
	capture_name = 'video1'
	try:
		camera = v4l2capture.Video_device('/dev/' + capture_name)
		camera.set_format(1280, 720)
		camera.create_buffers(10)				
		camera.start()
	except IOError:
		print("No such file or directory: /dev/" + capture_name)
	time.sleep(2)
	camera.queue_all_buffers()

	top = 270
	left = 350
	down = 719
	right = 780

	while True:
		start_time = time.time()
		readable, _, _ = select.select([camera], (), ())
		for camera in readable:
			image_data = camera.read_and_queue()
			image = Image.frombytes("RGB", (1280, 720), image_data)
			img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
			img_crop = img[top:down, left:right]
			img_encode = cv2.imencode('.jpg', img_crop)[1]

			b64encode_start_time = time.time()
			base64_data = base64.b64encode(img_encode)
			b64encode_elapsed_time = time.time() - b64encode_start_time

			base64_data_string = base64_data.decode()
			timestamp = int(round(time.time() * 1000))
			image_name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '_' + str(timestamp) + ".jpg"
			json_data = {}
			json_data['image_name'] = image_name
			json_data['image_data'] = base64_data_string
			headers = {'Content-Type':'application/json'}
			data = json.dumps(json_data)                                               

			request_start_time = time.time()
			req = urllib2.Request(url, headers=headers, data=data)
			resp = urllib2.urlopen(req).read().decode()
			request_elapsed_time = time.time() - request_start_time

			resp_data = json.loads(resp)
			if resp_data['face_count'] > 0:
				image.save('./Images/' + str(timestamp) + ".jpg")

			'''
			img = cv2.imread(image_path)
			for i in range(resp_data['face_count']):
				face_node = resp_data['face_list'][i]	
				left_up_x = face_node['x']
				left_up_y = face_node['y']
				right_down_x = face_node['x'] + face_node['width']
				right_down_y = face_node['y'] + face_node['height']
				cv2.rectangle(img, (left_up_x, left_up_y), (right_down_x, right_down_y), (255,0,0), 3)
			cv2.imwrite(os.path.join('Images', image_name), img)
			'''
		elapsed_time = time.time() - start_time
		print("\n")
		print("elapsed_time: %dms" %(int(round(elapsed_time * 1000))))

def recognize():
	url = "http://192.168.0.27:5000/face/recognize"
	capture_name = 'video1'
	try:
		camera = v4l2capture.Video_device('/dev/' + capture_name)
		camera.set_format(1280, 720)
		camera.create_buffers(10)				
		camera.start()
	except IOError:
		print("No such file or directory: /dev/" + capture_name)
	time.sleep(2)
	camera.queue_all_buffers()

	tmp_image_path = "./tmp/recognization.jpg"

	top = 270
	left = 350
	down = 719
	right = 780

	while True:
		start_time = time.time()
		readable, _, _ = select.select([camera], (), ())
		for camera in readable:
			image_data = camera.read_and_queue()
			image = Image.frombytes("RGB", (1280, 720), image_data)
			#image.save(tmp_image_path)

			img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
			img_crop = img[top:down, left:right]
			cv2.imwrite(tmp_image_path, img_crop)

			files = {'file':open(tmp_image_path, 'rb')}
			response = requests.post(url, files=files)
			resp_data = json.loads(response.text)
			if 0 == resp_data['face_count']:
				continue

			for i in range(resp_data['face_count']):
				face_item = resp_data['face_list'][i]
				print("\n")
				print("name=" + face_item['id'] + ", score=" + str(face_item['score']))
				dst_dir = os.path.join('result', 'recognization', face_item['id'])
				if not os.path.exists(dst_dir):
					os.makedirs(dst_dir)
				timestamp = int(round(time.time() * 1000))
				image_name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '_' + str(timestamp) + ".jpg"
				image_path = os.path.join(dst_dir, image_name)
				image.save(image_path)

if __name__ == '__main__':
	#detect()
	recognize()