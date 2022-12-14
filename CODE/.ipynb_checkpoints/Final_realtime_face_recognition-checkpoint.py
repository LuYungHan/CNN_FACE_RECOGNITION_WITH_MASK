# %matplotlib inline
# import matplotlib.pyplot as plt

# import sys
# sys.path.append('.')
# import logging
# mpl_logger = logging.getLogger('matplotlib')
# mpl_logger.setLevel(logging.WARNING)
# import logging.config
# logging.config.fileConfig("config/logging.conf")
# logger = logging.getLogger('api')
from datetime import datetime,timedelta 
# Ttime = str(datetime.now())
# print(Ttime)
# import yaml
import cv2
import numpy as np
#detect
# from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
# from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
# #landmark
# from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
# from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
# #crop
# from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper

# with open('config/model_conf.yaml') as f:
# #     model_conf = yaml.load(f)
#     model_conf = yaml.full_load(f)

# +
################## inference (face feature) lib ###############################
# -

import argparse
import torch
from backbones import get_model
# from _eval.verification import calculate_roc ,evaluate1,calculate_val
# from sklearn.metrics.pairwise import cosine_similarity


# +
################# all folder lib ##############################
# +

import os
from pathlib import Path

# +
################# feature2csv sub-function ######################

# +
import pandas as pd




# +
################# inference sub-function ######################

# +
@torch.no_grad()
def inference(net, img, file):
    # img 1st
#     print()
#     Ttime = str(datetime.now())
#     print('1:',Ttime)
#     if img is None:
#         img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
#     else:
#         img = cv2.imread(img)
# #         print(np.shape(img))
#         img = cv2.resize(img, (112, 112))
#     print()
#     Ttime = str(datetime.now())
#     print(Ttime)
#     img = cv2.imread(img)
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
#     print()
#     Ttime = str(datetime.now())
#     print('2:',Ttime)
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
#     print()
#     Ttime = str(datetime.now())
#     print('3:',Ttime)


    feat = net(img).numpy()
#     print()
#     Ttime = str(datetime.now())
#     print('6:',Ttime)
#     print('------------ img feat-----------------')
#     print(feat)
    # read_csv
#     if img1 is None:
#         img1 = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
#     else:
#         img1 = cv2.imread(img1)
#         img1 = cv2.resize(img1, (112, 112))

#     img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
#     img1 = np.transpose(img1, (2, 0, 1))
#     img1 = torch.from_numpy(img1).unsqueeze(0).float()
#     img1.div_(255).sub_(0.5).div_(0.5)

    data = np.array(pd.read_csv(file))
#     print(data)
#     print(data[0,:])
    compare_feat = data[:,3:]
    
#     net = get_model(name, fp16=False)
#     net.load_state_dict(torch.load(weight))
#     net.eval()
#     feat1 = net(img1).numpy()
#     print('------------ img1 feat-----------------')
#     print(feat1)
    
#     diff = np.subtract(feat, [data[0,3:]])#?????????
#     dist = np.sum(np.square(diff), 1)#?????????????????????????????????
    
#     print(dist)
#     regmat = np.array(feat) #regfeat??????
#     capmat = np.array(feat1) #capfeat??????

#     regmat_T = regmat.T
#     SimilarityMatrix = np.dot(capmat, regmat_T)#dot()?????????????????????????????????
# #     print(SimilarityMatrix)
    
#     cos = (float(dist)/float(SimilarityMatrix))
    
#     print(cosine_similarity(feat, feat1))
#     print(cosine_similarity(regmat, capmat))


    distances = (np.sum(np.square(feat - compare_feat),axis= 1))**0.5
#     print(distances**0.5)
    min_distance = np.argmin(distances)
    dist_idx = distances[min_distance]
    print(distances[min_distance])
#     distances_df = pd.DataFrame(distances)
#     print(distances_df)
#     distances_df = pd.DataFrame(distances)
#     distances_df.to_csv('10384_feat.csv')
#     print(distances_df)
    idx = np.argmin(distances)
    threshold = 1
#     print('This is :',data[idx,2])
    if dist_idx <= threshold : 
        print('threshold,This is :',data[idx,2])
        return data[idx,2]
    else :  
        print('unknown !!!')
        return 'unknown'
#     Ttime = str(datetime.now())
#     print(Ttime)
#     return cosine_similarity(regmat, capmat)
    
#     tpr, fpr, accuracy, val, val_std, far = calculate_roc(feat,feat1,'1',1,0)
#     print('tpr: ',tpr)
#     print('fpr: ',fpr)
#     print('accuracy: ',accuracy)




# +
############################## main function #################################################
# -

if __name__ == '__main__':


################################################################################## face detect
    # common setting for all model, need not modify.
#     model_path = 'models'

#     # model setting, modified along with model
#     scene = 'non-mask'
#     model_category = 'face_detection'
#     model_name =  model_conf[scene][model_category]

#     logger.info('Start to load the face detection model...')
#     # load model
#     try:
#         faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name)
#     except Exception as e:
#         logger.error('Failed to parse model configuration file!')
#         logger.error(e)
#         sys.exit(-1)
#     else:
#         logger.info('Successfully parsed the model configuration file model_meta.json!')

#     try:
#         model, cfg = faceDetModelLoader.load_model()
#     except Exception as e:
#         logger.error('Model loading failed!')
#         logger.error(e)
#         sys.exit(-1)
#     else:
#         logger.info('Successfully loaded the face detection model!')


#     path = 'api_usage/CCP_MI'
#     files = os.listdir(path)

#     for index, file in enumerate(files): 
#         staff_id = file
#         staff_id_path = path +'/'+ file
#         staff_id_photo_path = staff_id_path +'/photo'
#         staff_id_detect_path = staff_id_path +'/detect'

# #         if not os.path.exists(staff_id_photo_path):
# #                 os.makedirs(staff_id_photo_path)

#         if not os.path.exists(staff_id_detect_path):
#                 os.makedirs(staff_id_detect_path)
# #             path = os.path.join(path,'0_' + str(i))


#         print(staff_id)
#         print(index)
#         if not staff_id.startswith('.'):
#             img_ids = os.listdir(staff_id_photo_path)
# #             error_file_path = 'api_usage/CCP_MI/.ipynb_checkpoints/photo'

# #             if os.path.isfile(img_ids == error_file_path):
# #                     os.remove(error_file_path)
# #                     print("File has been deleted")
# #                     continue
# #             else:
# #                     print("File does not exist")
#             for index1, img_id in enumerate(img_ids):


#                 image_path = staff_id_photo_path + '/' + img_id
#         #         print(index1)
#         #         print(img_id)
#                 print(image_path)

#     #     num = 18
#     #     for i in range(0,num,1):    
#             # read image
#     #             image_path = 'api_usage/CCP_MI/'
#     #             image = cv2.imread(image_path +'{}.jpg'.format(i), cv2.IMREAD_COLOR)

#                 if not img_id.startswith('.'):
#                     image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#                     faceDetModelHandler = FaceDetModelHandler(model, 'cuda:0', cfg)

#                     try:
#                         dets = faceDetModelHandler.inference_on_image(image)
#                     except Exception as e:
#                        logger.error('Face detection failed!')
#                        logger.error(e)
#                        sys.exit(-1)
#                     else:
#                        logger.info('Successful face detection!')

#                     # gen result
#                     save_path_img = ('api_usage/CCP_MI/{}'.format(staff_id) +'/detect/detect_{}.jpg'.format(index1))
#                     save_path_txt = ('api_usage/CCP_MI/{}'.format(staff_id) +'/detect/detect_res_{}.txt'.format(index1))

#                     bboxs = dets
#                     with open(save_path_txt, "w") as fd:
#                         for box in bboxs:
#                             line = str(int(box[0])) + " " + str(int(box[1])) + " " + \
#                                    str(int(box[2])) + " " + str(int(box[3])) + " " + \
#                                    str(box[4]) + " \n"
#                             fd.write(line)

#                     for box in bboxs:
#                         box = list(map(int, box))
#                         cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
#                     cv2.imwrite(save_path_img, image)
#                     logger.info('Successfully generate face detection results!')


# ###################################################################### face_alignment(landmark)
#     # model setting, modified along with model
#     scene = 'non-mask'
#     model_category = 'face_alignment'
#     model_name =  model_conf[scene][model_category]

#     logger.info('Start to load the face landmark model...')
#     # load model
#     try:
#         faceAlignModelLoader = FaceAlignModelLoader(model_path, model_category, model_name)
#     except Exception as e:
#         logger.error('Failed to parse model configuration file!')
#         logger.error(e)
#         sys.exit(-1)
#     else:
#         logger.info('Successfully parsed the model configuration file model_meta.json!')

#     try:
#         model, cfg = faceAlignModelLoader.load_model()
#     except Exception as e:
#         logger.error('Model loading failed!')
#         logger.error(e)
#         sys.exit(-1)
#     else:
#         logger.info('Successfully loaded the face landmark model!')

#     faceAlignModelHandler = FaceAlignModelHandler(model, 'cuda:0', cfg)


#     path = 'api_usage/CCP_MI'
#     files = os.listdir(path)

#     for index, file in enumerate(files): 
#         staff_id = file
#         staff_id_path = path +'/'+file
#         staff_id_photo_path = staff_id_path +'/photo'
#         staff_id_landmark_path = staff_id_path +'/landmark'

# #         error_file_path = 'api_usage/CCP_MI/.ipynb_checkpoints/photo'

# #         if os.path.isfile(error_file_path):
# #                 os.remove(error_file_path)
# #                 print("File has been deleted")
# #                 continue
# #         else:
# #                 print("File does not exist")
# #                 continue
#         if not os.path.exists(staff_id_landmark_path):
#                 os.makedirs(staff_id_landmark_path)

#         print(staff_id)
#         print(index)
#         if not staff_id.startswith('.'):
#             img_ids = os.listdir(staff_id_photo_path)

#             for index1, img_id in enumerate(img_ids):
#                 image_path = staff_id_photo_path + '/' + img_id
#         #         print(index1)
#         #         print(img_id)
#                 print(image_path)

#     #     num = 18
#     #     for j in range(0,num,1):
#             # read image
#     #             image_path = 'api_usage/test/'
#                 if not img_id.startswith('.'):
#                     image_det_txt_path = 'api_usage/CCP_MI/{}/'.format(staff_id)
#                     image = cv2.imread(image_path , cv2.IMREAD_COLOR)
#                     with open(image_det_txt_path+'detect/detect_res_{}.txt'.format(index1), 'r') as f:
#                         lines = f.readlines()
#                     try:
#                         for i, line in enumerate(lines):
#                             line = line.strip().split()
#                             det = np.asarray(list(map(int, line[0:4])), dtype=np.int32)
#                             landmarks = faceAlignModelHandler.inference_on_image(image, det)

#                             save_path_img = 'api_usage/CCP_MI/{}'.format(staff_id) + '/landmark/landmark_res' + str(index1) + '.jpg'
#                             save_path_txt = 'api_usage/CCP_MI/{}'.format(staff_id) + '/landmark/landmark_res' + str(index1) + '.txt'
#                             image_show = image.copy()
#                             with open(save_path_txt, "w") as fd:
#                                 for (x, y) in landmarks.astype(np.int32):
#                                     cv2.circle(image_show, (x, y), 2, (255, 0, 0),-1)
#                                     line = str(x) + ' ' + str(y) + ' '
#                                     fd.write(line)
#                             cv2.imwrite(save_path_img, image_show)
#                     except Exception as e:
#                         logger.error('Face landmark failed!')
#                         logger.error(e)
#                         sys.exit(-1)
#                     else:
#                         logger.info('Successful face landmark!')

# ##################################################################################### face_crop 

#     path = 'api_usage/CCP_MI'
#     files = os.listdir(path)

#     for index, file in enumerate(files): 
#         staff_id = file
#         staff_id_path = path +'/'+file

#         staff_id_photo_path = staff_id_path +'/photo'
#         staff_id_face_path = staff_id_path +'/crop_face'
# #         error_file_path = 'api_usage/CCP_MI/.ipynb_checkpoints/photo'

# #         if os.path.isfile(error_file_path):
# #                 os.remove(error_file_path)
# #                 print("File has been deleted")
# #                 continue
# #         else:
# #                 print("File does not exist")
# #                 continue
#         if not os.path.exists(staff_id_face_path):
#                 os.makedirs(staff_id_face_path)
#         print(staff_id)
#         print(index)
#         if not staff_id.startswith('.'):
#             image_num_label = 0
#             img_ids = os.listdir(staff_id_photo_path)

#             for index1, img_id in enumerate(img_ids):
#                 image_path = staff_id_photo_path + '/' + img_id
#         #         print(index1)
#         #         print(img_id)
#                 print(image_path)
#                 image_label_name = image_num_label
#     #     num1 = 18
#     #     for i in range(0,num1,1):
#     #             image_path = 'api_usage/test/'
#                 if not img_id.startswith('.'):
#                     image_info_file = 'api_usage/CCP_MI/{}/'.format(staff_id)
#                     line = open(image_info_file+'landmark/landmark_res{}.txt'.format(index1)).readline().strip()
#                     landmarks_str = line.split(' ')
#                     landmarks = [float(num) for num in landmarks_str]

#                     face_cropper = FaceRecImageCropper()
#                     image = cv2.imread(image_path)
#                     cropped_image = face_cropper.crop_image_by_mat(image, landmarks)
#                     cropped_image = cv2.resize(cropped_image, (112, 112))  # ??????????????????128*128
#                     print(cropped_image.shape)
#                     cv2.imwrite('api_usage/CCP_MI/{}'.format(staff_id) + '/crop_face/{}.jpg'.format(image_label_name), cropped_image)
#                     logger.info('Crop image successful!')
#                     image_num_label = image_num_label + 1

# ################################################################################# get feature 

    net = get_model('r50', fp16=False)
    net.load_state_dict(torch.load('api_usage/work_dirs/ms1mv2_r50/model.pt'))
    net.eval()

    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
#     parser.add_argument('--network', type=str, default='r50', help='backbone network')
#     parser.add_argument('--weight', type=str, default='api_usage/work_dirs/ms1mv2_r50/model.pt')
    parser.add_argument('--img', type=str,default='api_usage/test_test_test/man1.jpg')
    
    parser.add_argument('--file', type=str, default='api_usage/all_face_feat_0824.csv')
    parser.add_argument('--net', type=str, default=net)
    args = parser.parse_args()
    args = parser.parse_args(args=[])
    predict_is_the_same = inference(args.net, args.img, args.file)

#     if abs(predict_is_the_same) > 0.45:
#         print('They are the same person')
#     else:
#         print('They are different person')

# ################################################################################# feature2csv

#     path = 'api_usage/CCP_MI'
#     files = os.listdir(path)

#     for index, file in enumerate(files): 
#         staff_id = file
#         staff_id_path = path +'/'+file

#         staff_id_photo_path = staff_id_path +'/photo'
#         staff_id_face_path = staff_id_path +'/face'
#         error_file_path = 'api_usage/CCP_MI/.ipynb_checkpoints/face'
#         print(staff_id_face_path)
#         if os.path.isfile(error_file_path):
#                 os.remove(error_file_path)
#                 print("File has been deleted")

#         else:
#                 print("File does not exist")
#         continue
#                 continue
#         if not os.path.exists(staff_id_face_path):
#                 os.makedirs(staff_id_face_path)
#         print(staff_id)
#         print(index)

#         if not staff_id.startswith('.'):
#             img_ids = os.listdir(staff_id_face_path)
#             print(img_ids)
    #         if os.path.isfile(error_file_path):
    #                 os.remove(error_file_path)
    #                 print("File has been deleted")

    #         else:
    #                 print("File does not exist")

#             for index1, img_id in enumerate(img_ids):
#                 image_path = staff_id_face_path + '/' + img_id
#         #         print(index1)
#         #         print(img_id)
#                 print(image_path)



############################
#     parser.add_argument('--file', type=str, default='api_usage/all_face_feat.csv')

#     args = parser.parse_args(args=[])
#     feat2csv(args.weight, args.network, args.file)
