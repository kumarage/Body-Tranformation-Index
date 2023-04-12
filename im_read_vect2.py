#https://learnopencv.com/contour-detection-using-opencv-python-c/
import numpy as np
import cv2
 
# read the image
image1 = cv2.imread('image5x1V.png')
image2 = cv2.imread('image5x1F.png')

# convert the image to grayscale format
img_gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
print(img_gray1.shape)
#cv2.imshow('None approximation', img_gray1)
#cv2.waitKey(0)
#cv2.imwrite('contours_none_image1.jpg', image_copy)

img_gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
print(img_gray2.shape)
#cv2.imshow('None approximation', img_gray2)
#cv2.waitKey(0)
#cv2.imwrite('contours_none_image1.jpg', image_copy)


nx1 = img_gray1.shape[1]
ny1 = img_gray1.shape[0]
nx2 = img_gray2.shape[1]
ny2 = img_gray2.shape[0]

print(nx1,ny1)
print(nx2,ny2)
#profile = np.zeros(ny1)
frontal = np.zeros(ny1)

#for i in range(ny1):
#	for j in range(nx1):
#		if(img_gray1[i][j]!=255):
#			profile[i] += 1


for i in range(ny2):
	for j in range(nx2):
		if(img_gray2[i][j]!=255):
			frontal[i] += 1


#for i in range(ny1):
#	print(frontal[i],'	',profile[i])


body_weight = np.array([96.815,98.26,99.705])
n_w = 3

height = 1.7
#BMI = body_weight/(height*height)

tot_cubes = 0

for i in range(ny1):
	for j in range(nx1):
		if(img_gray1[i][j]!=255):
			nD = abs(1 - abs(j-nx1/2)/(nx1/2))
			n = int(nD*frontal[i-50])
			tot_cubes += n

print(tot_cubes)
dens = np.zeros(n_w)
#density in kilograms per pix cube

for k in range(n_w):
	dens[k] = body_weight[k]/tot_cubes
print(dens)

baseline = 0
#calculate the second moment
IYY = 0
IXX = 0
im_width = 0.75
im_height = ny1*0.75/nx1
wd = im_width/nx1

for k in range(n_w):
	IYY = 0
	IXX = 0
	for i in range(ny1):
		for j in range(nx1):
			if(img_gray1[i][j]!=255):
				nD = abs(1 - abs(j-nx1/2)/(nx1/2))
				n = int(nD*frontal[i-50])
				m = dens[k]*n
				#IYY +=(m*(wd**2+(n*wd)**2)/12)+m*abs((j-baseline)*im_width/nx1)**2
				IXX +=(m*(wd**2+(n*wd)**2)/12)+m*abs((i-baseline)*im_height/ny1)**2
	#print('IYY = ',IYY)
	print('IXX = ',IXX)

cv2.destroyAllWindows()