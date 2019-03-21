import cv2 as cv
import numpy as np
import sqlite3

# Tao ham cho biet them hay update khuon mat
# Va chinh sua data cho phu hop
def insertOrUpdate(Id,Name):
    conn=sqlite3.connect("dataface.db")
    cmd="SELECT * FROM facedata WHERE ID="+str(Id)
    cursor=conn.execute(cmd)
    isRecordExist=0
	# Neu gia tri ID da co thi update nguoc lai insert
    for row in cursor:
        isRecordExist=1
    if(isRecordExist==1):
        cmd="UPDATE facedata SET Name='"+str(Name)+"'WHERE ID="+str(Id)
    else:
        cmd="INSERT INTO facedata(ID,Name) Values("+str(Id)+",'"+str(Name)+"')"
    conn.execute(cmd)
    conn.commit()
    conn.close()
	
faces_cascade  = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv.VideoCapture(0)
ident = input('Enter your ID: ')
name=input('Enter your name: ')
insertOrUpdate(ident,name)
number_sample = 0

while True:
	ret, img = cam.read()
	img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	# phat hien khuon mat
	faces = faces_cascade.detectMultiScale(img_gray, 1.3, 5)
	for (x, y, w, h) in faces:
		number_sample = number_sample + 1
		# luu khuon mat dung dia chi file dataset
		cv.imwrite('dataset/person' + '_' + str(ident) + '_' + str(number_sample) + '.jpg', img_gray[y:y+h, x:x+w])
		cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
		cv.waitKey(200)
	cv.imshow('image', img)
	# Lay 20 anh khuon mat sau moi 0.2 giay
	if (number_sample > 19):
		break
	if (cv.waitKey(100) & 0xff == ord('q')):
		break

cam.release()
cv.destroyAllWindows()
