# 執行順序:
##### 1.face_detect(detect the person face)
##### 2.face_alignmnet(landmark_face)
##### 3.face_crop(transform the right position of the person)
##### 4.fece_feature(calculate the person featrue and store into .npy)
###### 5.face_pipeline(calculate the similarity of the two people)

# instruction:
##### problem 1:
##### no module:utils.XXX
##### reason:
##### 因為 python 自己有utils 的module ，這樣的話他取不到user自己寫的utils 所以要加下面的solution
##### 然後還要在XXX同層加上一個__init__.py才不會出錯
##### solution:
##### 加上這段code before import utils.XXX
##### import sys
##### sys.path = ['.'] + sys.path
##### print(sys.path)

```python

```
