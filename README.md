# IMAGE-TRANSFORMATIONS


## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:
Import the necessary libraries and read the original image and save it as a image variable.

### Step2:
Translate the image using a function warpPerpective()

### Step3:
Scale the image by multiplying the rows and columns with a float value.

### Step4:
Shear the image in both the rows and columns.

### Step5:
Find the reflection of the image.

### step 6:
Rotate the image using angle function.

## Program:
## Developed By: THARIKA S
## Register Number: 212222230159
## i)Original Image:
```
import numpy as np
import cv2
import matplotlib.pyplot as plt
input_img = cv2.imread("rapunzel.jpg")
# cv2.imshow("image webp",input_img)
input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(input_img)
plt.show()
```
## ii)Image Translation
```
rows,cols,dim=input_img.shape
M=np.float32([[1,0,50],  [0,1,100],  [0,0,1]])
translated_image=cv2.warpPerspective(input_img,M,(cols,rows))
plt.axis('off')
plt.imshow(translated_image)
plt.show()
```
## iii) Image Scaling
```
scale_factor = 1.5
M_scale = np.float32([[scale_factor, 0, 0],
                      [0, scale_factor, 0],
                      [0, 0, 1]])

scaled_img = cv2.warpAffine(input_img, M[:2], (int(cols*scale_factor), int(rows*scale_factor)))
plt.axis('off')
plt.imshow(scaled_img)
plt.show()
```
## iv)Image shearing
```
M_x = np.float32([[1, 0.2, 0],
                  [0, 1, 0],
                  [0, 0, 1]])

sheared_img_xaxis = cv2.warpAffine(input_img, M_x[:2], (cols, rows))
plt.axis('off')
plt.imshow(sheared_img_xaxis)
plt.show()
M_y = np.float32([[1, 0, 0],
                  [0.2, 1, 0],
                  [0, 0, 1]])

sheared_img_yaxis = cv2.warpAffine(input_img, M_y[:2], (cols, rows))
plt.axis('off')
plt.imshow(sheared_img_yaxis)
plt.show()
```
## v)Image Reflection
```
M_x = np.float32([[-1, 0, cols],
                  [0, 1, 0],
                  [0, 0, 1]])

reflected_img_xaxis = cv2.warpAffine(input_img, M_x[:2], (cols, rows))

plt.axis("off")
plt.imshow(reflected_img_xaxis)
plt.show()

M_y = np.float32([[1, 0, 0],
                  [0, -1, rows],
                  [0, 0, 1]])

reflected_img_yaxis = cv2.warpAffine(input_img, M_y[:2], (cols, rows))

plt.axis("off")
plt.imshow(reflected_img_yaxis)
plt.show()
```
## vi)Image Rotation
```
angle = np.radians(-60)
M = np.float32([[np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0]])
center = (cols // 2, rows // 2)
M = cv2.getRotationMatrix2D(center, -60, 1.0)
rotated_img = cv2.warpAffine(input_img, M, (cols, rows))

plt.axis('off')
plt.imshow(rotated_img)
plt.show()
```
## vii)Image Cropping
```
cropped_img = input_img[50:200, 200:400]
plt.axis('off')
plt.imshow(cropped_img)
plt.show()
```

## Output:
### i)Original image:
![image](https://github.com/user-attachments/assets/6fc65b72-8665-4da6-8cb8-5b06c796fe7e)

### ii)Image Translation:
![image](https://github.com/user-attachments/assets/a305aec4-7922-490b-abbc-42cbba7b7518)

### iii) Image Scaling:
![image](https://github.com/user-attachments/assets/69ee6090-8e0c-4db3-91a9-26e3557a3765)

### iv)Image shearing:
![image](https://github.com/user-attachments/assets/88fe0aa1-10c7-48ea-af16-22f38691bb63)

### v)Image Reflection:
![image](https://github.com/user-attachments/assets/dbe71684-3f38-4caa-b7d3-4ad0c8c91339)

### vi)Image Rotation:
![image](https://github.com/user-attachments/assets/a9d9ebbf-a23a-44a1-8d2b-e1500f8efc82)

### vii)Image Cropping:
![image](https://github.com/user-attachments/assets/32e7042e-3b93-4ae9-8d9d-39014c031290)


## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.

