# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import os
path = './api_usage/CCP_MI_test_original'
files = os.listdir(path)
print(files)

i = 1
for index, file in enumerate(files): 
#     os.rename(os.path.join(path, file),os.path.join(path, ''.join('0_0_{}'.format(("%07d" % i)))))
#     print(index)
#     print(file)
    i = i+1
    staff_id = file
# print('total files number : ',i)
#     rename_img_path = path + '/' + file + '/' + 'face'
    rename_img_path = path + '/' + file + '/'+ 'crop_face'
#     print(rename_img_path)
    images = os.listdir(rename_img_path)
    img_num = 0
    for index1, image_name in enumerate(images): 
        if not image_name.startswith('.'):
            
            staff_id_img_total_num = index1 
            os.rename(rename_img_path + '/' + image_name , rename_img_path + '/' + staff_id +  '_{}'.format(("%07d" % img_num))+'.jpg')
            img_num = img_num + 1
        
#     staff_id_img_total_num = staff_id_img_total_num + 1 
#     for num in range(0,index+1,1):
#     os.rename(path + '/' + staff_id + '/' +  ,path+'/0_0_{}'.format(("%07d" % num)))
#     print(num)


