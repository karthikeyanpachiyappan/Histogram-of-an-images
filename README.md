# Histogram-of-an-images

## Name : KARTHIKEYAN P
## Reg.No: 212223230102

## Aim
To obtain a histogram for finding the frequency of pixels in an Image with pixel values ranging from 0 to 255. Also write the code using OpenCV to perform histogram equalization.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:
Read the gray and color image using imread()

### Step2:
Print the image using imshow().



### Step3:
Use calcHist() function to mark the image in graph frequency for gray and color image.

### step4:
Use calcHist() function to mark the image in graph frequency for gray and color image.

### Step5:
The Histogram of gray scale image and color image is shown.


## Program:
```
# Name : KARTHIKEYAN P
# Reg.No.: 212223230102
# In[1]:Write your code to find the histogram of gray scale image and color image channels

import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('parrot.jpg', cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.show()




# In[2]:Display the histogram of gray scale image and any one channel histogram from color image

import numpy as np
# Read the color image.
img = cv2.imread('parrot.jpg', cv2.IMREAD_COLOR)
# Convert to HSV.
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# Perform histogram equalization only on the V channel, for value intensity.
img_hsv[:,:,2] = cv2.equalizeHist(img_hsv[:, :, 2])
# Convert back to BGR format.
img_eq = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
 plt.imshow(img_eq[:,:,::-1]); plt.title('Equalized Image');plt.show()
# Display the images.
plt.imshow(img_eq, cmap='gray')
plt.title('Original Image')
plt.show()

# In[3]:Write the code to perform histogram equalization of the image. 

# Display the images.
#plt.figure(figsize = (20,10))
plt.subplot(221); plt.imshow(img[:, :, ::-1]); plt.title('Original Color Image')
plt.subplot(222); plt.imshow(img_eq[:, :, ::-1]); plt.title('Equalized Image')
plt.subplot(223); plt.hist(img.ravel(),256,range = [0, 256]); plt.title('Original Image')
plt.subplot(224); plt.hist(img_eq.ravel(),256,range = [0, 256]); plt.title('Histogram Equalized');plt.show()
# Display the histograms.
plt.figure(figsize = [15,4])
plt.subplot(121); plt.hist(img.ravel(),256,range = [0, 256]); plt.title('Original Image')
plt.subplot(122); plt.hist(img_eq.ravel(),256,range = [0, 256]); plt.title('Histogram Equalized')
```
## Output:

## Input Grayscale Image and Color Image
<img width="757" height="521" alt="Screenshot 2025-10-16 153833" src="https://github.com/user-attachments/assets/2ea52211-9bfa-4ae7-a53d-f4aab0a03568" />

<img width="742" height="493" alt="Screenshot 2025-10-16 153825" src="https://github.com/user-attachments/assets/2760daae-1dc9-4c49-81df-fc58fa520c22" />

## Histogram of Grayscale Image and any channel of Color Image

<img width="800" height="544" alt="Screenshot 2025-10-16 153919" src="https://github.com/user-attachments/assets/e81bd0cb-bad4-4786-9001-e2fd9a12aea9" />
<img width="751" height="548" alt="Screenshot 2025-10-16 153927" src="https://github.com/user-attachments/assets/53d623a4-886f-41d7-ad61-32453d64d36c" />

## Histogram Equalization of Grayscale Image.


<img width="877" height="541" alt="Screenshot 2025-10-16 153818" src="https://github.com/user-attachments/assets/1bab59dd-b4a5-4696-b9b3-484057cf4d91" />





## Result: 
Thus the histogram for finding the frequency of pixels in an image with pixel values ranging from 0 to 255 is obtained. Also,histogram equalization is done for the gray scale image using OpenCV.
