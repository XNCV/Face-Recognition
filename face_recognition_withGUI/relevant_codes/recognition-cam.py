import cv2 as cv
import numpy as np
import sqlite3

faces_detect = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv.VideoCapture(0)
recognize = cv.face.LBPHFaceRecognizer_create()
recognize.read('trained_file/trained_file.yml')

# Ham tra ve ten tuong ung voi ID
def getProfile(id):
    conn=sqlite3.connect("dataface.db")
    cmd="SELECT * FROM facedata WHERE ID="+str(id)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row[1]
    conn.close()
    return profile

while True:
	ret, image = cam.read()
	image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	# phat hien khuon mat
	faces = faces_detect.detectMultiScale(image_gray, 1.3, 5)
	for (x, y, w, h) in faces:
		cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
		# Matching khuon mat moi voi nhung khuon mat trong dataset
		ID, conf = recognize.predict(image_gray[y:y+h, x:x+w])
		#print(str(conf))
		profile = getProfile(ID)
		# Neu khoang cach > 60 thi 2 khuon mat khong cung mot nguoi
		# Nguoc lai, hien ten nguoi tuong ung
		if(conf < 60):
		    if(profile!=None):
			    cv.putText(image, str(profile), (x, y+h), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
		else:
			cv.putText(image, str('Unknown'), (x, y+h), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

	cv.imshow('Face', image)
	if (cv.waitKey(1) & 0xff == ord('q')):
		break

cam.release()
cv.destroyAllWindows()
