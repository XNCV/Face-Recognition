import cv2
import os
import numpy as np
from PIL import Image

path = 'dataset'
training = cv2.face.LBPHFaceRecognizer_create()

# Tao ham lay du lieu anh trong dataset vaf training dung LBPH
def FacesAndID(path):
	# Dia chi cua nhung khuon mat
	face_paths = [os.path.join(path, file) for file in os.listdir(path)]
	faces = []
	IDs = []
	for face_path in face_paths:
		imagepil = Image.open(face_path).convert('L')
		imagenp = np.array(imagepil, 'uint8')
		Id=int(os.path.split(face_path)[1].split("_")[1])
		faces.append(imagenp)
		IDs.append(Id)
	return faces, np.array(IDs)

faces,IDs = FacesAndID(path)
# Thuc hien training
training.train(faces, IDs)
# Luu ket qua vao file .yml
training.save('trained_file/trained_file.yml')

