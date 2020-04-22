import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
from sklearn.metrics import confusion_matrix,accuracy_score
import matplotlib.pyplot as plt




def predict(x,y):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x = torch.from_numpy(x).view(len(x), 3, 50, 50).float().to(device)
    y = torch.from_numpy(y).long().to(device)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    DROPOUT = 0.5
    softmax = nn.Softmax(dim=1)

    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, (3, 3), padding=1)  # output (n_examples, 16, 50, 50)
            self.convnorm1 = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16, 64, (3, 3), padding=1)  # output (n_examples, 64, 50, 50)
            self.convnorm2 = nn.BatchNorm2d(64)
            self.pool1 = nn.MaxPool2d((2, 2))  # output (n_examples, 64, 25, 25)
            self.conv3 = nn.Conv2d(64, 128, (3, 3), padding=1)  # output (n_examples, 128, 25, 25)
            self.convnorm3 = nn.BatchNorm2d(128)
            self.conv4 = nn.Conv2d(128, 256, (3, 3), padding=1)  # output (n_examples, 256, 25, 25)
            self.convnorm4 = nn.BatchNorm2d(256)
            self.pool2 = nn.MaxPool2d((2, 2))  # output (n_examples, 256, 12, 12)
            self.conv5 = nn.Conv2d(256, 256, (3, 3), padding=1)  # output (n_examples, 256, 12, 12)
            self.convnorm5 = nn.BatchNorm2d(256)
            self.conv6 = nn.Conv2d(256, 512, (3, 3), padding=1)  # output (n_examples, 512, 12, 12)
            self.convnorm6 = nn.BatchNorm2d(512)
            self.pool3 = nn.MaxPool2d((2, 2))  # output (n_examples, 512, 6, 6)
            self.linear1 = nn.Linear(512 * 6 * 6, 400)
            self.linear1_bn = nn.BatchNorm1d(400)
            self.drop = nn.Dropout(DROPOUT)
            self.linear2 = nn.Linear(400, 4)
            self.act = torch.relu

        def forward(self, x):
            x = self.convnorm1(self.act(self.conv1(x)))
            x = self.pool1(self.convnorm2(self.act(self.conv2(x))))
            x = self.convnorm3(self.act(self.conv3(x)))
            x = self.pool2(self.convnorm4(self.act(self.conv4(x))))
            x = self.convnorm5(self.act(self.conv5(x)))
            x = self.pool3(self.convnorm6(self.act(self.conv6(x))))
            x = self.drop(self.linear1_bn(self.act(self.linear1(x.view(len(x), -1)))))
            return self.linear2(x)

    model = CNN().to(device)
    model.load_state_dict(torch.load("model1.pt"))
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        pred_labels = np.argmax(y_pred.cpu().numpy(), axis=1)
        acc=100 * accuracy_score(y_val, pred_labels)
    return y_pred,acc


# %% -------------------------------------------------------------------------------------------------------------------
x,y=np.load('/home/ubuntu/Deep_Learning/final/data/x_train.npy'),np.load(
    '/home/ubuntu/Deep_Learning/final/data/y_train.npy')


print(predict(x,y))





