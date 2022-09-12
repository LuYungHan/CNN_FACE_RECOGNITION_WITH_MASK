# -*- coding: utf-8 -*-
# + {}
######################## face detect ###############################
# -

"""
@author: JiXuan Xu, Jun Wang
@date: 20201019
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

import yaml
import cv2
import numpy as np
#detect
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
#landmark
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
#crop
from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper

with open('config/model_conf.yaml') as f:
#     model_conf = yaml.load(f)
    model_conf = yaml.full_load(f)

# +
################## inference (face feature) lib ###############################
# -

import argparse
import torch
from backbones import get_model
from _eval.verification import calculate_roc ,evaluate1,calculate_val
from sklearn.metrics.pairwise import cosine_similarity


# +
################# inference sub-function ######################

# +
@torch.no_grad()
def inference(weight, name, img, img1):
    # img 1st
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        print(np.shape(img))
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight))
    net.eval()
    feat = net(img).numpy()
    print('------------ img feat-----------------')
#     print(feat)
    # img 2nd
    if img1 is None:
        img1 = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img1 = cv2.imread(img1)
        img1 = cv2.resize(img1, (112, 112))

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1 = np.transpose(img1, (2, 0, 1))
    img1 = torch.from_numpy(img1).unsqueeze(0).float()
    img1.div_(255).sub_(0.5).div_(0.5)
    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight))
    net.eval()
    feat1 = net(img1).numpy()
    print('------------ img1 feat-----------------')
#     print(feat1)
    
    diff = np.subtract(feat, feat1)#做減法
    dist = np.sum(np.square(diff), 1)#計算各元素的平方後加和
    
    print(dist)
    regmat = np.array(feat) #regfeat特徵
    capmat = np.array(feat1) #capfeat特徵

    regmat_T = regmat.T
    SimilarityMatrix = np.dot(capmat, regmat_T)#dot()返回的是兩個矩陣的乘積
#     print(SimilarityMatrix)
    
    cos = (float(dist)/float(SimilarityMatrix))
    
    print(cosine_similarity(feat, feat1))
    print(cosine_similarity(regmat, capmat))
    
    return cosine_similarity(regmat, capmat)
    
#     tpr, fpr, accuracy, val, val_std, far = calculate_roc(feat,feat1,'1',1,0)
#     print('tpr: ',tpr)
#     print('fpr: ',fpr)
#     print('accuracy: ',accuracy)


# +
###################### main function ######################################
# -

if __name__ == '__main__':
    # common setting for all model, need not modify.
    model_path = 'models'

    # model setting, modified along with model
    scene = 'non-mask'
    model_category = 'face_detection'
    model_name =  model_conf[scene][model_category]

    logger.info('Start to load the face detection model...')
    # load model
    try:
        faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name)
    except Exception as e:
        logger.error('Failed to parse model configuration file!')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Successfully parsed the model configuration file model_meta.json!')

    try:
        model, cfg = faceDetModelLoader.load_model()
    except Exception as e:
        logger.error('Model loading failed!')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Successfully loaded the face detection model!')
        
    num = 18
    for i in range(0,num,1):    
        # read image
        image_path = 'api_usage/test/'
        image = cv2.imread(image_path +'{}.jpg'.format(i), cv2.IMREAD_COLOR)
        faceDetModelHandler = FaceDetModelHandler(model, 'cuda:0', cfg)

        try:
            dets = faceDetModelHandler.inference_on_image(image)
        except Exception as e:
           logger.error('Face detection failed!')
           logger.error(e)
           sys.exit(-1)
        else:
           logger.info('Successful face detection!')

        # gen result
        save_path_img = ('api_usage/test/detect_17821_{}.jpg'.format(i))
        save_path_txt = ('api_usage/test/detect_17821_res_{}.txt'.format(i))

        bboxs = dets
        with open(save_path_txt, "w") as fd:
            for box in bboxs:
                line = str(int(box[0])) + " " + str(int(box[1])) + " " + \
                       str(int(box[2])) + " " + str(int(box[3])) + " " + \
                       str(box[4]) + " \n"
                fd.write(line)

        for box in bboxs:
            box = list(map(int, box))
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        cv2.imwrite(save_path_img, image)
        logger.info('Successfully generate face detection results!')
        
        
    ######### face_alignment
    # model setting, modified along with model
    scene = 'non-mask'
    model_category = 'face_alignment'
    model_name =  model_conf[scene][model_category]

    logger.info('Start to load the face landmark model...')
    # load model
    try:
        faceAlignModelLoader = FaceAlignModelLoader(model_path, model_category, model_name)
    except Exception as e:
        logger.error('Failed to parse model configuration file!')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Successfully parsed the model configuration file model_meta.json!')

    try:
        model, cfg = faceAlignModelLoader.load_model()
    except Exception as e:
        logger.error('Model loading failed!')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Successfully loaded the face landmark model!')

    faceAlignModelHandler = FaceAlignModelHandler(model, 'cuda:0', cfg)
    num = 18
    for j in range(0,num,1):
        # read image
        image_path = 'api_usage/test/'
        image_det_txt_path = 'api_usage/test/'
        image = cv2.imread(image_path + '{}.jpg'.format(j), cv2.IMREAD_COLOR)
        with open(image_det_txt_path+'detect_17821_res_{}.txt'.format(j), 'r') as f:
            lines = f.readlines()
        try:
            for i, line in enumerate(lines):
                line = line.strip().split()
                det = np.asarray(list(map(int, line[0:4])), dtype=np.int32)
                landmarks = faceAlignModelHandler.inference_on_image(image, det)

                save_path_img = 'api_usage/test/17821_' + 'landmark_res' + str(j) + '.jpg'
                save_path_txt = 'api_usage/test/17821_mask_' + 'landmark_res' + str(j) + '.txt'
                image_show = image.copy()
                with open(save_path_txt, "w") as fd:
                    for (x, y) in landmarks.astype(np.int32):
                        cv2.circle(image_show, (x, y), 2, (255, 0, 0),-1)
                        line = str(x) + ' ' + str(y) + ' '
                        fd.write(line)
                cv2.imwrite(save_path_img, image_show)
        except Exception as e:
            logger.error('Face landmark failed!')
            logger.error(e)
            sys.exit(-1)
        else:
            logger.info('Successful face landmark!')
        
    #######face_crop 
    
    num1 = 18
    for i in range(0,num1,1):
        image_path = 'api_usage/test/'
        image_info_file = 'api_usage/test/'
        line = open(image_info_file+'17821_mask_landmark_res{}.txt'.format(i)).readline().strip()
        landmarks_str = line.split(' ')
        landmarks = [float(num) for num in landmarks_str]

        face_cropper = FaceRecImageCropper()
        image = cv2.imread(image_path+'{}.jpg'.format(i))
        cropped_image = face_cropper.crop_image_by_mat(image, landmarks)
        cropped_image = cv2.resize(cropped_image, (112, 112))  # 將大小修改成128*128
        print(cropped_image.shape)
        cv2.imwrite('api_usage/test/crop_{}.jpg'.format(i), cropped_image)
        logger.info('Crop image successful!')
        
        
    ######## get feature ###########################
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='api_usage/work_dirs/ms1mv2_r50/model.pt')
    parser.add_argument('--img', type=str, default='api_usage/test/crop_0.jpg')
    parser.add_argument('--img1', type=str, default='api_usage/test/crop_18.jpg')
    args = parser.parse_args()
    predict_is_the_same = inference(args.weight, args.network, args.img, args.img1)
        
    if abs(predict_is_the_same) > 0.45:
        print('They are the same person')
    else:
        print('They are different person')




