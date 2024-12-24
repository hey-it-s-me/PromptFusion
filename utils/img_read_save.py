import numpy as np
import cv2
import os
from skimage.io import imsave
from PIL import Image

# def image_read_cv2(path, mode='RGB'):
#     img_BGR = cv2.imread(path).astype('float32')
#     assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
#     if mode == 'RGB':
#         img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
#     elif mode == 'GRAY':  
#         img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
#     elif mode == 'YCrCb':
#         img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
#     return img

# def img_save(image,imagename,savepath):
#     if not os.path.exists(savepath):
#         os.makedirs(savepath)
#     # Gray_pic
#     imsave(os.path.join(savepath, "{}.png".format(imagename)),image)

def image_read_cv2(path, mode='RGB'):
    # 'float32'
    img = Image.open(path)
    # img_BGR = cv2.imread(path)
    img = cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)
    # .astype(np.uint8)
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':  
        img = np.round(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    return img

def img_save(image,imagename,savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    # Gray_pic
    imsave(os.path.join(savepath, "{}.png".format(imagename)),image.astype(np.uint8))