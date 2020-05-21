import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sys

def SegmentImage(im,k,name):
    #im=cv2.imread('Beach.jpg') #reads image in BGR format
    im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    original_shape=im.shape
    plt.imshow(im)  #Image in real colors
    plt.show()

    shape1=im.shape[0]
    shape2=im.shape[1]
    shape3=im.shape[2]

    #Flatten Each Channel of the Image
    all_pixels=im.reshape((shape1*shape2,3))
    print(all_pixels.shape)

    dominant_colors= k  #Giving the value of K
    km=KMeans(n_clusters=dominant_colors)
    km.fit(all_pixels)

    centers=np.array(km.cluster_centers_,dtype='uint8')
    new_img=np.zeros((shape1*shape2,3),dtype='uint8')

    for i in range(new_img.shape[0]):
        new_img[i]=colors[km.labels_[i]]
        new_img=new_img.reshape((original_shape))
    
    new_img_converted=cv2.cvtColor(new_img,cv2.COLOR_RGB2BGR)
    cv2.imwrite(name,new_img_converted)


def getSegmentedImageDocument(inputImage,k,outputImage):
    im=cv2.imread('Beach.jpg') #reads image in BGR format
    SegmentImage(im,k,outputImage)



# Read command line arguments
inputImage = sys.argv[1]
k =sys.argv[2]
outputImage = sys.argv[3]
  
getSegmentedImageDocument(inputImage,k,outputImage)












