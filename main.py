from face_service_dlib import FaceService
from flask import Flask
import sys

if __name__ == '__main__':
	app = Flask(__name__)
	face_service_instance = FaceService(app)
	face_service_instance.start_service()	
	app.run(host='0.0.0.0', port=int(sys.argv[1]), threaded=True)