import os
import numpy as np
from torchvision import transforms as tfs
from PIL import Image
import random
from sklearn.model_selection import train_test_split

##--------------------------define augmentation function--------------------------
def aug(x):
    im_aug = tfs.Compose([
        tfs.RandomRotation(30),
        tfs.RandomGrayscale(),
        tfs.RandomHorizontalFlip(p=0.5),
        tfs.RandomVerticalFlip(p=0.5)
    ])
    x = np.array(im_aug(x))
    return x

x1, y1 = np.load('x.npy'), np.load('y.npy')

cls = list(y1)
cls_dic = dict((i, cls.count(i)) for i in cls)
print(cls_dic)
print(len(cls_dic))

x2, y2 =[], []
for i in range(len(x1)):
    label = y1[i]
    label_num = cls_dic[y1[i]]

    for j in range(5000// label_num):
        x2.append(aug(tfs.ToPILImage()(x1[i])))
        y2.append(y1[i])

x2, y2= np.array(x2), np.array(y2)

x, y= np.vstack((x1,x2)), np.hstack((y1,y2))

#split
x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.2,random_state=31)

np.save("x_train.npy", x_train); np.save("y_train.npy", y_train)
np.save("x_val.npy", x_val); np.save("y_val.npy", y_val)

cls = list(y_train)
cls_dic = dict((i, cls.count(i)) for i in cls)
print(cls_dic)
print(len(cls_dic))

cls = list(y)
cls_dic = dict((i, cls.count(i)) for i in cls)
print(cls_dic)
print(len(cls_dic))

print(x_train.shape)
print(x_val.shape)
print(y_train.shape)
print(y_val.shape)