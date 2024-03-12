import os
import cv2
i=0
j=0
img_folder = "/home/sid/Documents/new_pro/City_walk/"
cap = cv2.VideoCapture("/home/sid/Documents/new_pro/city.mp4")
while cap.isOpened(): 
	ret,img=cap.read()
	i= i+1
	if ret is True:
		if (i % 60==0):
			j=j+1
			frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			h,w,_ =frame_rgb.shape
			#img=cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
			cv2.imwrite(img_folder+"city_"+str(j)+".jpg",img)
	else:
		break
cap.release()
print(i,h,w)