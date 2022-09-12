# -*- coding: utf-8 -*-
"""
@author: JiXuan Xu, Jun Wang
@date: 20201015
@contact: jun21wangustc@gmail.com 
"""
import sys
sys.path.append('.')
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
import logging.config
logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('api')
import cv2

from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper

if __name__ == '__main__':
    num1 = 26
    for i in range(0,num1,1):
        image_path = 'api_usage/17821/'
        image_info_file = 'api_usage/landmark_17821/'
        line = open(image_info_file+'17821_mask_landmark_res{}.txt'.format(i)).readline().strip()
        landmarks_str = line.split(' ')
        landmarks = [float(num) for num in landmarks_str]

        face_cropper = FaceRecImageCropper()
        image = cv2.imread(image_path+'LINE_ALBUM_Crop_icecream_220809_{}.jpg'.format(i))
        cropped_image = face_cropper.crop_image_by_mat(image, landmarks)
        cropped_image = cv2.resize(cropped_image, (112, 112))  # 將大小修改成128*128
        print(cropped_image.shape)
        cv2.imwrite('api_usage/crop_17821/{}.jpg'.format(i), cropped_image)
        logger.info('Crop image successful!')
        
