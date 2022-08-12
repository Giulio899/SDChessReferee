import numpy as np
import base64
import cv2
import io
from PIL import Image

class TestBitmap:

    def __init__(self):
        self.count = 0

    def processBitmap(self, bitmap):
        decoded_image= base64.b64decode(bitmap)
        image= np.fromstring(decoded_image,np.uint8)
        image_as_ndarray= cv2.imdecode(image,cv2.IMREAD_UNCHANGED)
        image_without_alpha_channel= image_as_ndarray[:,:,:3]
        # passare a pyhton image_without_alpha_channel

        #encode
        pil_im = Image.fromarray(image_without_alpha_channel)

        buff = io.BytesIO()
        pil_im.save(buff,format="PNG")
        img_str = base64.b64encode(buff.getvalue())

        return "" + str(img_str, 'utf-8')
