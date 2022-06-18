import numpy as np
import base64
import cv2

def processBitmap(bitmap):
    decoded_image= base64.b64decode(bitmap)
    image= np.fromstring(decoded_image,np.uint8)
    image_as_ndarray= cv2.imdecode(image,cv2.IMREAD_UNCHANGED)
    image_without_alpha_channel= image_as_ndarray[:,:,:3]
    # passare a pyhton image_without_alpha_channel
