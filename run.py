import sys
import os

import cv2
import json
import retinex

with open('config.json', 'r') as f:
    config = json.load(f)

img = cv2.imread('data/raw/image19.jpg')

cv2.imshow('img', img)
# print('MSRCR processing...')
# img_msrsr = retinex.MSRCR(
#     img,
#     config['sigma_list'],
#     config['G'],
#     config['b'],
#     config['alpha'],
#     config['beta'],
#     config['low_clip'],
#     config['high_clip']
# )
# cv2.imshow('MSRCR retinex', img_msrsr)
# cv2.imwrite("MSRCR_retinex.tif",img_msrsr)

print('amsrcr processing......')
img_amsrcr = retinex.automatedMSRCR(
    img,
    config['sigma_list']
)
# img_amsrcr_denoise = cv2.fastNlMeansDenoisingColored(img_amsrcr, img_amsrcr, 1, 100, 1, 10)
cv2.imshow('autoMSRCR retinex', img_amsrcr)
cv2.imwrite('AutomatedMSRCR_retinex.jpg', img_amsrcr)

# print('msrcp processing......')
# img_msrcp = retinex.MSRCP(
#     img,
#     config['sigma_list'],
#     config['low_clip'],
#     config['high_clip']
# )
#
# shape = img.shape
# cv2.imshow('Image', img)
#
# cv2.imshow('MSRCP', img_msrcp)
# cv2.imwrite('MSRCP.tif', img_msrcp)

cv2.waitKey()