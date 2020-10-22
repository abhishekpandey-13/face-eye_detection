import cv2
import numpy as np
face = cv2.CascadeClassifier('haarcascade.xml')
eye = cv2.CascadeClassifier('haarcascade_eye.xml')
img = cv2.VideoCapture(0)
while True:
	_,cap = img.read()
	gray = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
	faces = face.detectMultiScale(gray, 1.3, 5)
	for(x,y,w,h) in faces:
		cv2.rectangle(cap,(x,y),(x+w,y+h), (255,0,0),2)
		roi = gray[y:y+h,x:x+w]
		roi_color = cap[y:y+h,x:x+w]
		eyes = eye.detectMultiScale(roi, 1.3, 5)
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh),(0,255,0),2)
	cv2.imshow('cap',cap)
	k = cv2.waitKey(30) & 0xff
	if k==27:
		break
img.release()
cv2.destroyAllWindows()