from flask import Flask, jsonify, request
import base64
import os
import json
import uuid
from scipy import misc
from PIL import Image
import numpy as np
import zlib
import cv2
import time
import face_recognition
from face_dlib import Face

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
	return '.' in filename and \
		filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class FaceService:
	def __init__(self, app):
		self.app_ = app
		self.face_ = Face()

	def start_service(self):
		self.app_.add_url_rule('/face', view_func=self.face, methods=['GET','POST'])
		self.app_.add_url_rule('/face/register', view_func=self.face_register, methods=['POST'])
		self.app_.add_url_rule('/face/detect', view_func=self.face_detect, methods=['POST'])
		self.app_.add_url_rule('/face/detect_by_image_path', view_func=self.face_detect_by_image_path, methods=['POST'])
		self.app_.add_url_rule('/face/recognize', view_func=self.face_recognize, methods=['POST'])
		#self.app_.add_url_rule('/face/compare', view_func=self.face_compare, methods=['GET', 'POST'])
		self.app_.add_url_rule('/face/detect_and_recognize', view_func=self.face_detect_and_recognize, methods=['POST'])

	def face(self):
		return "Welcome to FaceWorld!"

	def face_compare(self):
		return "face_compare"

	def face_detect(self):
		print("\n")
		
		json_data = request.get_json()
		image_name = json_data['image_name']
		image_data_base64 = json_data['image_data'].encode()
		str_encode = base64.b64decode(image_data_base64)
		nparr = np.fromstring(str_encode, np.uint8)
		img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
		image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
		image = np.asarray(image)

		face_detect_start_time = time.time()
		face_locations = face_recognition.face_locations(image, model='cnn')
		face_detect_elapsed_time = time.time() - face_detect_start_time
		print("face_detect_elapsed_time: %dms" %(int(round(face_detect_elapsed_time * 1000))))

		resp_json_data = {}
		if None == face_locations or 0 == len(face_locations):
			resp_json_data['face_count'] = 0
			resp_json_data['error_msg'] = "Can not detect face."
		else:
			resp_json_data['face_count'] = len(face_locations)
			face_list = []
			for i in range(len(face_locations)):
				face_node = {}
				face_node['x'] = int(face_locations[i][3])
				face_node['y'] = int(face_locations[i][0])
				face_node['width'] = int(face_locations[i][1] - face_locations[i][3])
				face_node['height'] = int(face_locations[i][2] - face_locations[i][0])
				face_list.append(face_node)
			resp_json_data['face_list'] = face_list

		return jsonify(resp_json_data)

	def face_detect_by_image_path(self):
		json_data = request.get_json()
		image_name = json_data['image_name']
		image_data_base64 = json_data['image_data'].encode()
		image_data = base64.b64decode(image_data_base64)
		image_path = os.path.join('images', image_name)

		with open(image_path, 'wb') as f:
			f.write(image_data)
		image = face_recognition.load_image_file(image_path)

		start_time = time.time()
		face_locations = face_recognition.face_locations(image, model='cnn')
		elpased_time = time.time() - start_time
		print("detect time: %dms" %(int(round(elpased_time * 1000))))

		resp_json_data = {}
		if None == face_locations or 0 == len(face_locations):
			resp_json_data['face_count'] = 0
			resp_json_data['error_msg'] = "Can not detect face."
		else:
			resp_json_data['face_count'] = len(face_locations)
			face_list = []
			for i in range(len(face_locations)):
				face_node = {}
				fface_node['x'] = int(face_locations[i][3])
				face_node['y'] = int(face_locations[i][0])
				face_node['width'] = int(face_locations[i][1] - face_locations[i][3])
				face_node['height'] = int(face_locations[i][2] - face_locations[i][0])
				face_list.append(face_node)
			resp_json_data['face_list'] = face_list
		
		return jsonify(resp_json_data)

	def face_recognize(self):
		if 'file' not in request.files:
			return jsonify({})

		file = request.files['file']
		if file.filename == '':
			return jsonify({})

		if not allowed_file(file.filename):
			return jsonify({})

		if file == None:
			return jsonify({})

		image_bytes = file.read()
		#pil_image = Image.frombytes(image_bytes)
		pil_image = Image.open(file)
		np_image = np.asarray(pil_image)

		min_id_set, face_locations, min_score_set = self.face_.recognize_by_image(np_image)

		resp_json_data = {}
		resp_json_data['face_count'] = len(min_id_set)
		face_list = []
		for i in range(len(min_id_set)):
			face_item = {}
			face_item['id'] = min_id_set[i]
			face_item['x'] = int(face_locations[i][0])
			face_item['y'] = int(face_locations[i][3])
			face_item['width'] = int(face_locations[i][1] - face_locations[i][3])
			face_item['height'] = int(face_locations[i][2] - face_locations[i][0])
			face_item['score'] = min_score_set[i][0]
			face_list.append(face_item)
		resp_json_data['face_list'] = face_list

		return jsonify(resp_json_data)

	def face_register(self):
		json_data = request.get_json()
		image_name = json_data['image_name']
		image_data_base64 = json_data['image_data'].encode()
		image_data = base64.b64decode(image_data_base64)
		image_path = os.path.join('images', image_name)

		name = image_name.split('.')[0]
		image_type = image_name.split('.')[1]
		str_uuid = str(uuid.uuid1())

		with open(image_path, 'wb') as f:
			f.write(image_data)
		image = face_recognition.load_image_file(image_path)
		face_locations = face_recognition.face_locations(image, model='cnn')

		if None != face_locations and 1 == len(face_locations):
			face_image_save_path = os.path.join('register_faces', name+'_'+str_uuid+'.png')
			face_feature_save_path = os.path.join('register_faces', name+'_'+str_uuid+'.fea')
			redis_key = name + '_' + str_uuid
			feature = face_recognition.face_encodings(image, face_locations)[0]

			top, right, bottom, left = face_locations[0]
			face_image = image[top:bottom, left:right]
			pil_image = Image.fromarray(face_image)
			pil_image.save(face_image_save_path)

			feature.tofile(face_feature_save_path)

			self.face_.register_by_feature_id_and_feature(redis_key, feature)

		resp_json_data = {}
		if None == face_locations or 1 != len(face_locations):
			resp_json_data['face_count'] = 0
			resp_json_data['error_msg'] = "Register faces should be 1"
		else:
			resp_json_data['face_count'] = len(face_locations)
			resp_json_data['x'] = int(face_locations[0][3])
			resp_json_data['y'] = int(face_locations[0][0])
			resp_json_data['width'] = int(face_locations[0][1] - face_locations[0][3])
			resp_json_data['height'] = int(face_locations[0][2] - face_locations[0][0])
			resp_json_data['face_id'] = str_uuid

		return jsonify(resp_json_data)

	def remove_face(self):
		return "remove face"

	def face_detect_and_recognize(self):
		json_data = request.get_json()
		image_name = json_data['image_name']
		image_data_base64 = json_data['image_data'].encode()
		image_data = base64.b64decode(image_data_base64)
		image_path = os.path.join('images', image_name)
		with open(image_path, 'wb') as f:
			f.write(image_data)

		id_set, face_locations, max_score_set = self.face_.detect_and_recognize(image_path)
		resp_json_data = {}
		if None == id_set:
			resp_json_data['face_count'] = 0
		else:
			resp_json_data['face_count'] = len(id_set)
			face_list = []
			for i in range(len(id_set)):
				face_node = {}
				face_node['face_id'] = id_set[i]
				face_node['x'] = int(face_locations[i][3])
				face_node['y'] = int(face_locations[i][0])
				face_node['width'] = int(face_locations[i][1] - face_locations[i][3])
				face_node['height'] = int(face_locations[i][2] - face_locations[i][0])
				face_node['score'] = float(max_score_set[i])
				face_list.append(face_node)
			resp_json_data['face_list'] = face_list

		return jsonify(resp_json_data)