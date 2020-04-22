import os
import numpy as np
import csv
import cv2
##-----------------------------------------load data-----------------------------------------##

# train data
IMG_DIR = "/home/ubuntu/Deep_Learning/final/plant/images/"
LABEL_DIR = "/home/ubuntu/Deep_Learning/final/plant/"

x, y = [], []
for path in [f for f in os.listdir(IMG_DIR) if f[:3] == "Tra"]:
    x.append(cv2.resize(cv2.imread(IMG_DIR + path),(50,50)))
    with open(LABEL_DIR + "train.csv", "r") as i:
        reader = csv.reader(i)
        for row in reader:
            if row[0] == path[:-4]:
                y.append(row[1:].index('1'))

x, y = np.array(x), np.array(y)

# test data

x_test = []
ord_test = []
for path in [f for f in os.listdir(IMG_DIR) if f[:3] == "Tes"]:
    ord_test.append(path[:-4])
    x_test.append(cv2.resize(cv2.imread(IMG_DIR + path),(50,50)))

x_test = np.array(x_test)
ord_test = np.array(ord_test)

np.save('x.npy',x)
np.save('y.npy',y)
np.save('x_test.npy',x_test)
np.save('ord_test.npy',ord_test)
print(x.shape,y.shape)
print(x_test.shape,ord_test.shape)




