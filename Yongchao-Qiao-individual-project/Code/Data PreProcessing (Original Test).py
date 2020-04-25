import os
import re
import cv2
import torch
import numpy as np

device = torch.device("cpu")

total_img_dir = "data/images"
train_target_dir = "data/train.csv"
test_ID_dir = "data/test.csv"

total_name = os.listdir(total_img_dir)


def read_test_data_directory(directory_name):
    test_tuple = []

    def number(x):
        return list(map(int, list(re.findall(r"\d+", x))))
    counter = 1
    for filename in sorted(os.listdir(directory_name), key=number):
        if re.findall(r"Test", filename):
            img_test = cv2.imread(directory_name + "/" + filename)
            if img_test.shape != (1365, 2048, 3):
                print(counter)
                counter +=1
                print(img_test.shape, filename)
                img_test = img_test.swapaxes(1, 0)
                print(img_test.shape, filename)
            img_test_resized = cv2.resize(img_test, (164, 110), interpolation=cv2.INTER_LANCZOS4)
            test_tuple += [(img_test_resized, filename.strip(r"\.jpg"))]
    return test_tuple


test_image = read_test_data_directory(total_img_dir)
print(len(test_image), test_image[0][0].shape)
for i in range(20):
    # imshow(train_image[i][0] / 255)
    # plt.show()
    print(test_image[i][1])

res = np.zeros([len(test_image), 110, 164, 3])

for i in range(len(test_image)):
    res[i] = np.clip(test_image[i][0] / 255, 0, 1)

np.save("Test_110_164_9_6.npy", res)