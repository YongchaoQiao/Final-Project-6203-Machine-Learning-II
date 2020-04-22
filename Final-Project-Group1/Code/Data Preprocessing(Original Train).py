import os
import re
import cv2
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from keras.preprocessing.image import ImageDataGenerator

z = zipfile.ZipFile(r'plant-pathology-2020-fgvc7.zip', 'r')

for f in z.namelist():
    item = z.open(f, 'r')
    z.extract(f, r'data')
z.close()

total_img_dir = "data/images"
train_target_dir = "data/train.csv"
test_ID_dir = "data/test.csv"

total_name = os.listdir(total_img_dir)


def read_data_directory(directory_name):
    train_tuple = []

    def number(x):
        return list(map(int, list(re.findall(r"\d+", x))))

    for filename in sorted(os.listdir(directory_name), key=number):
        if re.findall(r"Train", filename):
            img_train = cv2.imread(directory_name + "/" + filename)
            if img_train.shape != (1365, 2048, 3):
                print(img_train.shape, filename)
                img_train = img_train.swapaxes(1, 0)
                print(img_train.shape, filename)
            img_train_resized = cv2.resize(img_train, (164, 110), interpolation=cv2.INTER_LANCZOS4)
            train_tuple += [(img_train_resized, filename.strip(r"\.jpg"))]
    return train_tuple


train_image = read_data_directory(total_img_dir)
print(len(train_image), train_image[0][0].shape)
for i in range(20):
    imshow(train_image[i][0] / 255)
    # plt.show()
    # print(train_image[i][1])

for i in range(len(train_image)):
    counter = 0
    if train_image[i][0].shape != (110, 164, 3):
        print(train_image[i][1], train_image[i][0].shape)
        counter += 1
print(counter)

train_target = pd.read_csv(train_target_dir)
train_target_np = train_target.values

'''category_name = []
target_total = np.zeros([len(train_target_np), 1])
for i in range(len(train_target_np)):
    if train_target_np[i][1] == 1:
        target_total[i] = 0
    elif train_target_np[i][2] == 1:
        target_total[i] = 1
    elif train_target_np[i][3] == 1:
        target_total[i] = 2
    elif train_target_np[i][4] == 1:
        target_total[i] = 3'''
# Augmentation
new_total_list = []
for j in range(len(train_image)):
    if train_image[j][1] == train_target_np[j][0]:
        new_total_list += [(train_image[j][0], train_target_np[j][1:5])]
        print(j)
        for i in range(5):
            image_generator = ImageDataGenerator(rotation_range=30, horizontal_flip=True, vertical_flip=True,
                                                 brightness_range=[0.8, 1.2],
                                                 width_shift_range=0.2, height_shift_range=0.2, featurewise_center=True,
                                                 featurewise_std_normalization=True, fill_mode='reflect')
            new_total_list += [(image_generator.random_transform(train_image[j][0], seed=None), train_target_np[j][1:5])]

# Resize the image array to (120, 180, 3) / 255
res = np.zeros([len(new_total_list), 82, 123, 3])

for i in range(len(new_total_list)):
    res[i] = np.clip(new_total_list[i][0] / 255, 0, 1)

# Get the one-hot coding target
target_total = np.zeros([len(new_total_list), 4])
for i in range(len(new_total_list)):
    target_total[i] = new_total_list[i][1]

print(new_total_list[0][1])
print(new_total_list[0][1][[0]])
print(new_total_list[0][1][[1]])
print(new_total_list[0][1][[2]])
print(new_total_list[0][1][[3]])

# Get the target_category_coding
target_category_coding = np.zeros([len(new_total_list), ])
for i in range(len(new_total_list)):
    if new_total_list[i][1][[0]] == 1:
        target_category_coding[i] = 0
    elif new_total_list[i][1][[1]] == 1:
        target_category_coding[i] = 1
    elif new_total_list[i][1][[2]] == 1:
        target_category_coding[i] = 2
    elif new_total_list[i][1][[3]] == 1:
        target_category_coding[i] = 3
print(len(target_category_coding))

print(res[0].shape, res[0], type(res[0]), target_total[0], target_total[0].shape)
# Save the x_total and target_total as np arrays
np.save("target_category_coding_82_123_9_6.npy", target_category_coding)
np.save("x_total_82_123_9_6.npy", res)
np.save("target_total_82_123_9_6.npy", target_total)

