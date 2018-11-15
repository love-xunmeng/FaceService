from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
from scipy.spatial.distance import pdist
import time
import redis
import threading
import face_recognition

class FeatureNode:
    def __init__(self, id, feature):
        self.id_ = id
        self.feature_ = feature

def consine_distance(vec1, vec2):
    dist = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return dist

class Face:
	def __init__(self):
		self.feature_node_list_ = []
		self.lock_ = threading.Lock()
		self.redis_cli_ = redis.Redis(host='localhost', port=6379, decode_responses=True)
		self.load_feature_from_redis()

	def detect(self, img):
		face_locations = face_recognition.face_locations(img, model='cnn')
		return face_locations

	def detect_by_image_path(self, image_path):
		img = face_recognition.load_image_file(image_path)
		face_locations = face_recognition.face_locations(img, model='cnn')
		return face_locations

	def extrace_feature(self, image, face_locations):
		feature = face_recognition.face_encodings(image, face_locations)[0]
		return feature

	def recognize_by_image(self, image):
		face_locations = self.detect(image)
		if 0 == len(face_locations):
			return [], [], []
		feature = self.extrace_feature(image, face_locations)
		min_id_set, min_score_set = self.recognize([feature])
		'''
		print("\n")
		print(type(feature))
		print(type(min_id_set))
		print(type(min_score_set))
		print()
		print(min_id_set)
		print(min_score_set)
		print("\n")
		'''
		return min_id_set, face_locations, min_score_set

	def recognize(self, feature_set):
		min_id_set = []
		min_score_set = []
		print("len(feature_set): ", len(feature_set))
		for i in range(len(feature_set)):
			min_score = 10000.0
			min_score_index = -10000
			min_id = ""
			print("len(self.feature_node_list_): %d" %(len(self.feature_node_list_)))
			for j in range(len(self.feature_node_list_)):
				score = face_recognition.face_distance([self.feature_node_list_[j].feature_], feature_set[i])
				if score < min_score:
					min_score = score
					min_score_index = j
					min_id = self.feature_node_list_[j].id_.split('_')[0]
			min_id_set.append(min_id)
			min_score_set.append(min_score)
		return min_id_set, min_score_set

	def detect_and_recognize(self, image_path):
		print()
		
		bounding_box_list, faces = self.detect(image_path)
		
		if None == bounding_box_list:
			return None, None, None
		start_time = time.time()
		feature_set = self.extrace_feature(faces)
		end_time = time.time()
		elapsed_time = end_time - start_time
		print("extract time: %ds", elapsed_time)

		start_time = time.time()
		id_set, max_score_set = self.recognize(feature_set)
		end_time = time.time()
		elapsed_time = end_time - start_time
		print("recognize time: %ds", elapsed_time)

		return id_set, bounding_box_list, max_score_set

	def load_feature_from_redis(self):
		print()
		keys = self.redis_cli_.keys()
		for feature_id in keys:
			feature = self.redis_cli_.lrange(feature_id, 0, -1)
			print(type(feature))
			print(len(feature))
			print(feature)
			for i in range(len(feature)):
				#print(feature[i])
				#print(type(feature[i]))
				#print()
				feature[i] = float(feature[i])
			feature_node = FeatureNode(feature_id, np.array(feature, dtype=np.float32))
			self.feature_node_list_.append(feature_node)
		print()
		print(len(self.feature_node_list_))

	def register_by_feature_id_and_feature(self, feature_id, feature):
		for i in range(len(feature)):
			self.redis_cli_.lpush(feature_id, feature[i])
		feature_node = FeatureNode(feature_id, feature)
		self.feature_node_list_.append(feature_node)