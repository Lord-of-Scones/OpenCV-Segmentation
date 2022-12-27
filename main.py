import cv2
import matplotlib.pyplot as plt
import numpy as np

inputImage = cv2.imread("sample.jpg", cv2.IMREAD_COLOR)
inputImageG = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

histogram = cv2.calcHist([inputImageG], [0], None, [256], [0, 256])

# Binary threshold one value
ret, threshedImage3 = cv2.threshold(inputImageG, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Multi value threshold of the image done by thresholding the orignial image section by section and then recombining them

ret, threshedImageL = cv2.threshold(inputImageG, 75, 255, cv2.THRESH_TOZERO)
ret, threshedImageL = cv2.threshold(threshedImageL, 142, 255, cv2.THRESH_TOZERO_INV)
ret, threshedImageL = cv2.threshold(threshedImageL, 75, 255, cv2.THRESH_TRUNC)

ret, threshedImageM = cv2.threshold(inputImageG, 142, 255, cv2.THRESH_TOZERO)
ret, threshedImageM = cv2.threshold(threshedImageM, 190, 255, cv2.THRESH_TOZERO_INV)
ret, threshedImageM = cv2.threshold(threshedImageM, 142, 255, cv2.THRESH_TRUNC)

ret, threshedImageH = cv2.threshold(inputImageG, 190, 255, cv2.THRESH_BINARY)

threshedImage2 = threshedImageL + threshedImageM + threshedImageH

# Convert orignal image in BGR format into RGB format
inputImageRGB = cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
# This is done because for the open CV k means formula it needs to be in RGB format

# the k means function takes in a 2D array but since this colored image is a 3D array (Hieght, Width, and channels) you need to reshape it
imageValues = inputImageRGB.reshape((-1, 3))

# It then needs to be converted to float form for the calculations (required by K means function)
imageValues = np.float32(imageValues)

# One of the parameters for the K means function is criteria, this is found by using the equation below
Criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, .85)

# K the number of clusters also needs to be designated and inputed into the function
k = 6

# Plug all of the parameters into the kmeans function, it then returns 3 values.
# Lables which is an array that shows which pixel belongs to which cluster, centers which show the center of each cluster, and retVal which we wont use
retVal, pixelLabels, (clusterCenters) = cv2.kmeans(imageValues, k, None, Criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# We most then convert the data from float back into int values and then map each value to the corisponding cluster center point
clusterCenters = np.uint8(clusterCenters)
pixelLabels = pixelLabels.flatten()
imageValuesS = clusterCenters[pixelLabels.flatten()]

# Then we need to reshape the newly mapped data into the image format of the original image
segmentedImageKMeans = imageValuesS.reshape(inputImageRGB.shape)
segmentedImageKMeans = cv2.cvtColor(segmentedImageKMeans, cv2.COLOR_RGB2BGR)

#First threshold the image, and then apply canny edge to help with finding contours
#Dialating it helps the contors stick out more for the algorithm
ret, tempImage = cv2.threshold(inputImageG, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cannyEdge = cv2.dilate(cv2.Canny(tempImage, 0, 255), None)

#Using the find contour algorithm to find contours and then using the sorted function to make the larger contours more accessible
contour = sorted(cv2.findContours(cannyEdge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2], key=cv2.contourArea)[-1]

#create an empty zero image equal to the size of the original and then use that as a mask to draw the found contours on
Height, Width = inputImageG.shape
mask = np.zeros((Height,Width), np.uint8)
cv2.drawContours(mask, [contour], -1, (255,255,255), -1)

#Then I use bitwise and to combine the original image and the mask to show only the segmented area and then
#convert iot to the proper RGB format
contouredImage = cv2.bitwise_and(inputImageRGB, inputImageRGB, mask=mask)
contouredImage = cv2.cvtColor(contouredImage, cv2.COLOR_RGB2BGR)

cv2.imshow('Normal Image', inputImage)
cv2.imshow('Greyscale Image', inputImageG)
cv2.imshow('Multi-value Thresh', threshedImage2)
cv2.imshow('Otsu Binarization image', threshedImage3)
cv2.imshow('K-means segemented image', segmentedImageKMeans)
cv2.imshow('Image with contours', contouredImage)
cv2.waitKey(0)

plt.plot(histogram)
plt.savefig("Greyscale Histogram.jpg")
plt.show()

cv2.imwrite("sample(Otsu Thresh).jpg", threshedImage3)
cv2.imwrite("sample(Multi-value Thresh).jpg", threshedImage2)
cv2.imwrite("sample(K means Segementation).jpg", segmentedImageKMeans)
cv2.imwrite("sample(Contoured).jpg", contouredImage)
