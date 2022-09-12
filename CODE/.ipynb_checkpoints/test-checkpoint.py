from api_usage.for_ipad_realtime_rec_IMPORTANT import inference
from PIL import Image
from backbones import get_model
import torch
import numpy as np
import json
import traceback
import time
import cv2
import os
from sklearn.preprocessing import normalize


# @torch.no_grad()
# def inference(net, img):
#     if img is None:
#         img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
#     else:
#         img = cv2.imread(img)
#         img = cv2.resize(img, (112, 112))

#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = np.transpose(img, (2, 0, 1))
#     img = torch.from_numpy(img).unsqueeze(0).float()
#     img.div_(255).sub_(0.5).div_(0.5)
#     feat = net(img).numpy()
#     return feat

if __name__ == '__main__':
    net = get_model('r50', fp16=False)
    net.load_state_dict(torch.load('./work_dirs/ms1mv2_r50/model.pt'))
    net.eval()
    inference(net,'api_usage/test_test_test/2.jpg','api_usage/all_face_feat_0824.csv',1)
#     feat1 = inference(net, './test_images/80108_2_transform.jpg')
#     feat2 = inference(net, './test_images/80108_transform.jpg')
#     feat1 = normalize(feat1)
#     feat2 = normalize(feat2)
#     print(np.sum(np.square(feat1 - feat2)))
