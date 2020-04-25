import os
import re
import cv2
import zipfile
import numpy as np
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

device = torch.device("cpu")
DROPOUT1 = 0.2
DROPOUT2 = 0.6

total_img_dir = "data/images"
train_target_dir = "data/train.csv"
test_ID_dir = "data/test.csv"

total_name = os.listdir(total_img_dir)


def read_data_directory(directory_name):
    test_tuple = []

    def number(x):
        return list(map(int, list(re.findall(r"\d+", x))))

    for filename in sorted(os.listdir(directory_name), key=number):
        if re.findall(r"Test", filename):
            img_test = cv2.imread(directory_name + "/" + filename)
            if img_test.shape != (1365, 2048, 3):
                print(img_test.shape, filename)
                img_test = img_test.swapaxes(1, 0)
                print(img_test.shape, filename)
            img_test_resized = cv2.resize(img_test, (164, 110), interpolation=cv2.INTER_LANCZOS4)
            test_tuple += [(img_test_resized, filename.strip(r"\.jpg"))]
    return test_tuple


test_image = read_data_directory(total_img_dir)
print(len(test_image), test_image[0][0].shape)
for i in range(20):
    # imshow(train_image[i][0] / 255)
    # plt.show()
    print(test_image[i][1])

res = np.zeros([len(test_image), 110, 164, 3])

for i in range(len(test_image)):
    res[i] = np.clip(test_image[i][0] / 255, 0, 1)

np.save("x_test_110_164_9_6.npy", res)

x_test = res
# %% -------------------------------------- CNN Class ------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, (1, 1))  # output (n_examples, 8, 110, 164)
        self.convnorm1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d((2, 2))  # output (n_examples, 8, 55, 82)
        self.conv2 = nn.Conv2d(8, 16, (1, 1))  # output (n_examples, 8, 110, 164)
        self.convnorm2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, (1, 3), groups=1)  # output (n_examples, 16, 55, 80)
        self.convnorm3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d((2, 2), padding=(1, 0))  # output (n_examples, 16, 28, 40)
        self.conv4 = nn.Conv2d(32, 48, (1, 1))  # output (n_examples, 8, 110, 164)
        self.convnorm4 = nn.BatchNorm2d(48)
        self.conv5 = nn.Conv2d(48, 64, (3, 1))  # output (n_examples, 32, 26, 40)
        self.convnorm5 = nn.BatchNorm2d(64)
        self.pool5 = nn.MaxPool2d((2, 2))  # output (n_examples, 32, 13, 20)
        self.conv6 = nn.Conv2d(64, 72, (1, 1))  # output (n_examples, 8, 110, 164)
        self.convnorm6 = nn.BatchNorm2d(72)
        self.conv7 = nn.Conv2d(72, 64, (3, 3), groups=1)  # output (n_examples, 48, 11, 18
        self.convnorm7 = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(64, 72, (1, 1))  # output (n_examples, 8, 110, 164)
        self.convnorm8 = nn.BatchNorm2d(72)
        self.conv9 = nn.Conv2d(72, 64, (3, 3), groups=1)  # output (n_examples, 48, 9, 16
        self.convnorm9 = nn.BatchNorm2d(64)
        self.conv10 = nn.Conv2d(64, 72, (1, 1))  # output (n_examples, 8, 110, 164)
        self.convnorm10 = nn.BatchNorm2d(72)
        self.conv11 = nn.Conv2d(72, 64, (3, 3), groups=1)  # output (n_examples, 48, 7, 14
        self.convnorm11 = nn.BatchNorm2d(64)
        self.linear1 = nn.Linear(64 * 7 * 14, 64)  # input will be flattened to (n_examples, 48*6*10)
        self.linear1_bn = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(DROPOUT1)
        self.linear2 = nn.Linear(64, 64)
        self.linear2_bn = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(DROPOUT2)
        self.linear3 = nn.Linear(64, 4)
        self.act = nn.ReLU(inplace=True)
        self.act2 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool1(self.convnorm1(self.act(self.conv1(x.permute(0, 3, 1, 2)))))
        x = self.convnorm2(self.act(self.conv2(x)))
        x = self.pool3(self.convnorm3(self.act(self.conv3(x))))
        x = self.convnorm4(self.act(self.conv4(x)))
        x = self.pool5(self.convnorm5(self.act(self.conv5(x))))
        x = self.convnorm6(self.act(self.conv6(x)))
        x = self.convnorm7(self.act(self.conv7(x)))
        x = self.convnorm8(self.act(self.conv8(x)))
        x = self.convnorm9(self.act(self.conv9(x)))
        x = self.convnorm10(self.act(self.conv10(x)))
        x = self.convnorm11(self.act(self.conv11(x)))
        x = self.drop1(self.linear1_bn(self.act(self.linear1(x.view(len(x), -1)))))
        x = self.drop2(self.linear2_bn(self.act(self.linear2(x))))
        return self.act2(self.linear3(x))

# Reshape the data
x_test = torch.tensor(x_test).float().to(device)
# Load the model
model = CNN().to(device)
PATH1 = 'model_yongchaoqiaofinal.pt'
# PATH2 = 'model_yongchaoqiao9day5_2.pt'
model.load_state_dict(torch.load(PATH1, map_location=device))
model.eval()
y_pred1 = model(x_test).detach().numpy()
# model.load_state_dict(torch.load(PATH2, map_location=device))
# model.eval()
# y_pred2 = model(x_test).detach().numpy()
# y_pred = (y_pred1 + y_pred2) / 2

y_pred = pd.DataFrame(y_pred1)
y_pred.columns = ["healthy", "multiple_diseases", "rust", "scab"]
print(y_pred, type(y_pred))

test_ID = pd.read_csv(test_ID_dir)
submission = pd.concat([test_ID, y_pred], axis=1)
print(submission)
submission.to_csv("submission_final.csv", index=False)
