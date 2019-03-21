import sys
import os
from PIL import Image
import cv2 as cv
import numpy as np
import sqlite3
import time
from PyQt5.QtCore import pyqtSlot, QTimer
from PyQt5.QtWidgets import QApplication, QDialog, QLineEdit, QMessageBox
from PyQt5.uic import loadUi
from PyQt5.QtGui import QImage, QPixmap


faces_detect = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
recognize = cv.face.LBPHFaceRecognizer_create()
recognize.read('trained_file/trained_file.yml')
cap = cv.VideoCapture(0)

# Give name correspond to ID
def getProfile(id):
	conn=sqlite3.connect("dataface.db")
	cmd="SELECT * FROM facedata WHERE ID="+str(id)
	cursor=conn.execute(cmd)
	profile=None
	for row in cursor:
		profile=row[1]
	conn.close()
	return profile
# Create of update new person
def insertOrUpdate(Id,Name):
    conn=sqlite3.connect("dataface.db")
    cmd="SELECT * FROM facedata WHERE ID="+str(Id)
    cursor=conn.execute(cmd)
    isRecordExist=0
	# If the ID is exists, we update name
    for row in cursor:
        isRecordExist=1
    if(isRecordExist==1):
        cmd="UPDATE facedata SET Name='"+str(Name)+"'WHERE ID="+str(Id)
    else:
        cmd="INSERT INTO facedata(ID,Name) Values("+str(Id)+",'"+str(Name)+"')"
    conn.execute(cmd)
    conn.commit()
    conn.close()
# Training using LBPH
def FacesAndID(path):
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
# GUI
class facerecog_gui(QDialog):
	def __init__(self):
		super(facerecog_gui,self).__init__()
		loadUi('face_recog_GUI.ui',self)
		self.setWindowTitle('Face Recognition Nhom 11')
		self.registerButton.clicked.connect(self.registerClicked)
		self.verifyButton.clicked.connect(self.controlTimer)
		self.quitButton.clicked.connect(self.quitClicked)
		# create a timer to verify
		self.timerVerify = QTimer()
		self.timerVerify.timeout.connect(self.viewCamVerify)
		# create a timer to register
		self.timerRegister = QTimer()
		self.timerRegister.timeout.connect(self.viewCamRegister)
	@pyqtSlot()
	def registerClicked(self):
		self.timerVerify.stop()
		global ident, name, number_sample
		ident = self.id_lineEdit.text()
		name = self.name_lineEdit.text()
		self.id_lineEdit.clear()
		self.name_lineEdit.clear()
		if ident == '' or name == '':
			QMessageBox.critical(recogUI, 'Error', 'ID or Name is illegal!!!')
			self.timerVerify.start(20)
		else:
			insertOrUpdate(ident,name)
			number_sample = 0
			self.timerRegister.start(20)
	def quitClicked(self):
		exitmessage = 'Do you want to exit?'
		resp = QMessageBox.question(self, 'Exit', exitmessage, QMessageBox.Yes, QMessageBox.No)
		if resp == QMessageBox.Yes: 
			self.timerVerify.stop()
			cap.release()
			self.close()
		else: pass
	def viewCamRegister(self):
		global ident, name, number_sample
		ret, image = cap.read()
		# convert image to RGB format
		image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
		img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
		# phat hien khuon mat
		faces = faces_detect.detectMultiScale(img_gray, 1.3, 5)
		for (x, y, w, h) in faces:
			number_sample = number_sample + 1
			# Save faces to dataset
			cv.imwrite('dataset/person' + '_' + str(ident) + '_' + str(number_sample) + '.jpg', img_gray[y:y+h, x:x+w])
			cv.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
			cv.waitKey(200)
		# get image infos
		height, width, channel = image.shape
		step = channel * width
		# create QImage from image
		qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
		# show image in img_label
		self.ShowImageLabel.setPixmap(QPixmap.fromImage(qImg))
		if (number_sample > 19) or (cv.waitKey(100) & 0xff == ord('q')):
			self.timerRegister.stop()
			QMessageBox.critical(recogUI, 'Information', 'Wait until OK!!!')
			faces,IDs = FacesAndID('dataset')
			# training
			recognize.train(faces, IDs)
			# save the result to file .yml
			recognize.save('trained_file/trained_file.yml')
			QMessageBox.critical(recogUI, 'Information', 'OK, recognize face now!!!')

	def viewCamVerify(self):
		# read image in BGR format
		ret, image = cap.read()
		# convert image to RGB format
		image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
		image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
		# Detect the faces
		faces = faces_detect.detectMultiScale(image_gray, 1.3, 5)
		for (x, y, w, h) in faces:
			cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
			ID, conf = recognize.predict(image_gray[y:y+h, x:x+w])
			profile = getProfile(ID)
			if(conf < 60):
				if(profile!=None):
					cv.putText(image, str(profile), (x, y+h), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
			else:
				cv.putText(image, str('Unknown'), (x, y+h), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
		# get image infos
		height, width, channel = image.shape
		step = channel * width
		# create QImage from image
		qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
		# show image in img_label
		self.ShowImageLabel.setPixmap(QPixmap.fromImage(qImg))

	# start/stop timer
	def controlTimer(self):
		# if timer is stopped
		if not self.timerVerify.isActive():
			self.timerVerify.start(20)
		# if timer is started
		else:
			# stop timer
			self.timerVerify.stop()


if __name__ == '__main__':
	app = QApplication(sys.argv)
	recogUI = facerecog_gui()
	recogUI.show()
	sys.exit(app.exec_())