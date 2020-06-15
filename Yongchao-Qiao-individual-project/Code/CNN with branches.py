# %% --------------------------------------- Imports -------------------------------------------------------------------
import torch
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
# %% --------------------------------------- Set-Up --------------------------------------------------------------------
device = torch.device("cpu")
SEED = 512
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 0.01  # 0.15 0.1 0.01
N_EPOCHS = 3000
BATCH_SIZE = 512  # 200, 412, 430(Max memory)
DROPOUT1 = 0.5
DROPOUT2 = 0.5
patience = 200
PATH = 'model_yongchaoqiaocombinationday14.pt'

# %% -------------------------------------- CNN Class ------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 3, (3, 3))  # output (n_examples, 8, 108, 162)
        self.convnorm1_1 = nn.BatchNorm2d(3)
        self.conv1_2 = nn.Conv2d(3, 8, (1, 1))  # output (n_examples, 8, 108, 162)
        self.convnorm1_2 = nn.BatchNorm2d(8)
        self.pool1_2 = nn.MaxPool2d((2, 2))  # output (n_examples, 8, 54, 81)

        self.conv1_3 = nn.Conv2d(8, 8, (3, 3))  # output (n_examples, 8, 52, 79)
        self.convnorm1_3 = nn.BatchNorm2d(8)
        self.conv1_4 = nn.Conv2d(8, 16, (1, 1))  # output (n_examples, 16, 52, 79)
        self.convnorm1_4 = nn.BatchNorm2d(16)

        self.conv1_5 = nn.Conv2d(16, 16, (3, 3))  # output (n_examples, 16, 50, 77)
        self.convnorm1_5 = nn.BatchNorm2d(16)
        self.conv1_6 = nn.Conv2d(16, 32, (1, 1))  # output (n_examples, 16, 50, 77)
        self.convnorm1_6 = nn.BatchNorm2d(32)

        self.conv1_7 = nn.Conv2d(32, 32, (3, 3))  # output (n_examples, 16, 48, 75)
        self.convnorm1_7 = nn.BatchNorm2d(32)
        self.conv1_8 = nn.Conv2d(32, 64, (1, 1))  # output (n_examples, 16, 48, 75)
        self.convnorm1_8 = nn.BatchNorm2d(64)
        self.pool1_8 = nn.MaxPool2d((2, 2), padding=(0, 1))  # output (n_examples, 32, 24, 38)

        self.conv1_9 = nn.Conv2d(64, 64, (3, 3))  # output (n_examples, 64, 22, 36)
        self.convnorm1_9 = nn.BatchNorm2d(64)
        self.conv1_10 = nn.Conv2d(64, 128, (1, 1))  # output (n_examples, 128, 22, 36)
        self.convnorm1_10 = nn.BatchNorm2d(128)

        self.conv1_11 = nn.Conv2d(128, 64, (1, 1))  # output (n_examples, 16, 22, 36)
        self.convnorm1_11 = nn.BatchNorm2d(64)
        self.conv1_12 = nn.Conv2d(64, 64, (3, 3))  # output (n_examples, 16, 20, 34)
        self.convnorm1_12 = nn.BatchNorm2d(64)
        self.conv1_13 = nn.Conv2d(64, 256, (1, 1))  # output (n_examples, 128, 20, 34)
        self.convnorm1_13 = nn.BatchNorm2d(256)

        self.conv1_14 = nn.Conv2d(256, 64, (1, 1))  # output (n_examples, 16, 20, 34)
        self.convnorm1_14 = nn.BatchNorm2d(64)
        self.conv1_15 = nn.Conv2d(64, 64, (3, 3))  # output (n_examples, 16, 18, 32)
        self.convnorm1_15 = nn.BatchNorm2d(64)
        self.conv1_16 = nn.Conv2d(64, 256, (1, 1))  # output (n_examples, 16, 18, 32)
        self.convnorm1_16 = nn.BatchNorm2d(256)

        self.conv1_17 = nn.Conv2d(256, 64, (1, 1))  # output (n_examples, 16, 18, 32)
        self.convnorm1_17 = nn.BatchNorm2d(64)
        self.conv1_18 = nn.Conv2d(64, 64, (3, 3))  # output (n_examples, 16, 16, 30)
        self.convnorm1_18 = nn.BatchNorm2d(64)
        self.conv1_19 = nn.Conv2d(64, 256, (1, 1))  # output (n_examples, 16, 16, 30)
        self.convnorm1_19 = nn.BatchNorm2d(256)

        self.conv1_20 = nn.Conv2d(256, 64, (1, 1))  # output (n_examples, 16, 16, 30)
        self.convnorm1_20 = nn.BatchNorm2d(64)
        self.conv1_21 = nn.Conv2d(64, 64, (3, 3))  # output (n_examples, 16, 14, 28)
        self.convnorm1_21 = nn.BatchNorm2d(64)
        self.conv1_22 = nn.Conv2d(64, 256, (1, 1))  # output (n_examples, 16, 14, 28)
        self.convnorm1_22 = nn.BatchNorm2d(256)

        self.conv1_23 = nn.Conv2d(256, 64, (1, 1))  # output (n_examples, 16, 14, 28)
        self.convnorm1_23 = nn.BatchNorm2d(64)
        self.conv1_24 = nn.Conv2d(64, 64, (3, 3))  # output (n_examples, 16, 12, 26)
        self.convnorm1_24 = nn.BatchNorm2d(64)
        self.conv1_25 = nn.Conv2d(64, 256, (1, 1))  # output (n_examples, 16, 12, 26)
        self.convnorm1_25 = nn.BatchNorm2d(256)
        self.pool1_25 = nn.MaxPool2d((2, 2), padding=(1, 0))  # output (n_examples, 64, 7, 13)


        self.conv2_1 = nn.Conv2d(3, 3, (3, 3))  # output (n_examples, 8, 108, 162)
        self.convnorm2_1 = nn.BatchNorm2d(3)
        self.conv2_2 = nn.Conv2d(3, 8, (1, 1))  # output (n_examples, 8, 108, 162)
        self.convnorm2_2 = nn.BatchNorm2d(8)
        self.pool2_2 = nn.MaxPool2d((2, 2))  # output (n_examples, 8, 54, 81)

        self.conv2_3 = nn.Conv2d(8, 8, (3, 3))  # output (n_examples, 16, 52, 79)
        self.convnorm2_3 = nn.BatchNorm2d(8)
        self.conv2_4 = nn.Conv2d(8, 16, (1, 1))  # output (n_examples, 8, 52, 79)
        self.convnorm2_4 = nn.BatchNorm2d(16)

        self.conv2_5 = nn.Conv2d(16, 16, (3, 3))  # output (n_examples, 8, 50, 77)
        self.convnorm2_5 = nn.BatchNorm2d(16)
        self.conv2_6 = nn.Conv2d(16, 32, (1, 1))  # output (n_examples, 8, 50, 77)
        self.convnorm2_6 = nn.BatchNorm2d(32)

        self.conv2_7 = nn.Conv2d(32, 32, (3, 3))  # output (n_examples, 8, 48, 75)
        self.convnorm2_7 = nn.BatchNorm2d(32)
        self.conv2_8 = nn.Conv2d(32, 64, (1, 1))  # output (n_examples, 8, 48, 75)
        self.convnorm2_8 = nn.BatchNorm2d(64)
        self.pool2_8 = nn.MaxPool2d((2, 2), padding=(0, 1))  # output (n_examples, 64, 24, 38)

        self.conv2_9 = nn.Conv2d(64, 64, (3, 3))  # output (n_examples, 64, 22, 36)
        self.convnorm2_9 = nn.BatchNorm2d(64)
        self.conv2_10 = nn.Conv2d(64, 128, (1, 1))  # output (n_examples, 128, 22, 36)
        self.convnorm2_10 = nn.BatchNorm2d(128)

        self.conv2_11 = nn.Conv2d(128, 64, (1, 1))  # output (n_examples, 16, 22, 36)
        self.convnorm2_11 = nn.BatchNorm2d(64)
        self.conv2_12 = nn.Conv2d(64, 64, (3, 3))  # output (n_examples, 16, 20, 34)
        self.convnorm2_12 = nn.BatchNorm2d(64)
        self.conv2_13 = nn.Conv2d(64, 256, (1, 1))  # output (n_examples, 128, 20, 34)
        self.convnorm2_13 = nn.BatchNorm2d(256)

        self.conv2_14 = nn.Conv2d(256, 64, (1, 1))  # output (n_examples, 16, 20, 34)
        self.convnorm2_14 = nn.BatchNorm2d(64)
        self.conv2_15 = nn.Conv2d(64, 64, (3, 3))  # output (n_examples, 16, 18, 32)
        self.convnorm2_15 = nn.BatchNorm2d(64)
        self.conv2_16 = nn.Conv2d(64, 256, (1, 1))  # output (n_examples, 128, 18, 32)
        self.convnorm2_16 = nn.BatchNorm2d(256)

        self.conv2_17 = nn.Conv2d(256, 64, (1, 1))  # output (n_examples, 16, 18, 32)
        self.convnorm2_17 = nn.BatchNorm2d(64)
        self.conv2_18 = nn.Conv2d(64, 64, (3, 3))  # output (n_examples, 16, 16, 30)
        self.convnorm2_18 = nn.BatchNorm2d(64)
        self.conv2_19 = nn.Conv2d(64, 256, (1, 1))  # output (n_examples, 128, 16, 30)
        self.convnorm2_19 = nn.BatchNorm2d(256)

        self.conv2_20 = nn.Conv2d(256, 64, (1, 1))  # output (n_examples, 16, 16, 30)
        self.convnorm2_20 = nn.BatchNorm2d(64)
        self.conv2_21 = nn.Conv2d(64, 64, (3, 3))  # output (n_examples, 16, 14, 28)
        self.convnorm2_21 = nn.BatchNorm2d(64)
        self.conv2_22 = nn.Conv2d(64, 256, (1, 1))  # output (n_examples, 128, 14, 28)
        self.convnorm2_22 = nn.BatchNorm2d(256)

        self.conv2_23 = nn.Conv2d(256, 64, (1, 1))  # output (n_examples, 16, 14, 28)
        self.convnorm2_23 = nn.BatchNorm2d(64)
        self.conv2_24 = nn.Conv2d(64, 64, (3, 3))  # output (n_examples, 16, 12, 26)
        self.convnorm2_24 = nn.BatchNorm2d(64)
        self.conv2_25 = nn.Conv2d(64, 256, (1, 1))  # output (n_examples, 128, 12, 26)
        self.convnorm2_25 = nn.BatchNorm2d(256)
        self.pool2_25 = nn.MaxPool2d((2, 2), padding=(1, 0))  # output (n_examples, 64, 7, 13)

        self.conv3_1 = nn.Conv2d(3, 3, (3, 3))  # output (n_examples, 8, 108, 162)
        self.convnorm3_1 = nn.BatchNorm2d(3)
        self.conv3_2 = nn.Conv2d(3, 8, (1, 1))  # output (n_examples, 8, 102, 162)
        self.convnorm3_2 = nn.BatchNorm2d(8)

        self.conv3_3 = nn.Conv2d(8, 4, (1, 1))  # output (n_examples, 8, 102, 162)
        self.convnorm3_3 = nn.BatchNorm2d(4)
        self.conv3_4 = nn.Conv2d(4, 4, (3, 3))  # output (n_examples, 8, 100, 160)
        self.convnorm3_4 = nn.BatchNorm2d(4)
        self.conv3_5 = nn.Conv2d(4, 8, (1, 1))  # output (n_examples, 8, 100, 160)
        self.convnorm3_5 = nn.BatchNorm2d(8)
        self.pool3_5 = nn.MaxPool2d((2, 2))  # output (n_examples, 8, 53, 80)

        self.conv3_6 = nn.Conv2d(8, 8, (3, 3))  # output (n_examples, 8, 51, 78)
        self.convnorm3_6 = nn.BatchNorm2d(8)
        self.conv3_7 = nn.Conv2d(8, 16, (1, 1))  # output (n_examples, 8, 51, 78)
        self.convnorm3_7 = nn.BatchNorm2d(16)

        self.conv3_8 = nn.Conv2d(16, 16, (3, 3))  # output (n_examples, 8, 49, 76）
        self.convnorm3_8 = nn.BatchNorm2d(16)
        self.conv3_9 = nn.Conv2d(16, 32, (1, 1))  # output (n_examples, 8, 49, 76)
        self.convnorm3_9 = nn.BatchNorm2d(32)

        self.conv3_10 = nn.Conv2d(32, 32, (3, 3))  # output (n_examples, 8, 47, 74)
        self.convnorm3_10 = nn.BatchNorm2d(32)
        self.conv3_11 = nn.Conv2d(32, 64, (1, 1))  # output (n_examples, 8, 47, 74)
        self.convnorm3_11 = nn.BatchNorm2d(64)

        self.conv3_12 = nn.Conv2d(64, 32, (1, 1))  # output (n_examples, 8, 47, 74)
        self.convnorm3_12 = nn.BatchNorm2d(32)
        self.conv3_13 = nn.Conv2d(32, 32, (3, 3))  # output (n_examples, 8, 45, 72)
        self.convnorm3_13 = nn.BatchNorm2d(32)
        self.conv3_14 = nn.Conv2d(32, 64, (1, 1))  # output (n_examples, 8, 45, 72)
        self.convnorm3_14 = nn.BatchNorm2d(64)

        self.conv3_15 = nn.Conv2d(64, 32, (1, 1))  # output (n_examples, 8, 45, 72)
        self.convnorm3_15 = nn.BatchNorm2d(32)
        self.conv3_16 = nn.Conv2d(32, 32, (3, 3))  # output (n_examples, 8, 43, 70)
        self.convnorm3_16 = nn.BatchNorm2d(32)
        self.conv3_17 = nn.Conv2d(32, 64, (1, 1))  # output (n_examples, 8, 43, 70)
        self.convnorm3_17 = nn.BatchNorm2d(64)

        self.conv3_18 = nn.Conv2d(64, 32, (1, 1))  # output (n_examples, 8, 43, 70)
        self.convnorm3_18 = nn.BatchNorm2d(32)
        self.conv3_19 = nn.Conv2d(32, 32, (3, 3))  # output (n_examples, 8, 41, 68)
        self.convnorm3_19 = nn.BatchNorm2d(32)
        self.conv3_20 = nn.Conv2d(32, 128, (1, 1))  # output (n_examples, 8, 41, 68)
        self.convnorm3_20 = nn.BatchNorm2d(128)
        self.pool3_20 = nn.MaxPool2d((2, 2), padding=(1, 0))  # output (n_examples, 32, 21, 34)


        self.conv3_21 = nn.Conv2d(128, 64, (1, 1))  # output (n_examples, 48, 21, 34)
        self.convnorm3_21 = nn.BatchNorm2d(64)
        self.conv3_22 = nn.Conv2d(64, 64, (3, 3))  # output (n_examples, 48, 19, 32)
        self.convnorm3_22 = nn.BatchNorm2d(64)
        self.conv3_23 = nn.Conv2d(64, 128, (1, 1))  # output (n_examples, 48, 19, 32)
        self.convnorm3_23 = nn.BatchNorm2d(128)

        self.conv3_24 = nn.Conv2d(128, 64, (1, 1))  # output (n_examples, 48, 19, 32)
        self.convnorm3_24 = nn.BatchNorm2d(64)
        self.conv3_25 = nn.Conv2d(64, 64, (3, 3))  # output (n_examples, 48, 17, 30)
        self.convnorm3_25 = nn.BatchNorm2d(64)
        self.conv3_26 = nn.Conv2d(64, 128, (1, 1))  # output (n_examples, 48, 17, 30)
        self.convnorm3_26 = nn.BatchNorm2d(128)

        self.conv3_27 = nn.Conv2d(128, 64, (1, 1))  # output (n_examples, 64, 17, 30)
        self.convnorm3_27 = nn.BatchNorm2d(64)
        self.conv3_28 = nn.Conv2d(64, 64, (3, 3))  # output (n_examples, 48, 15, 28)
        self.convnorm3_28 = nn.BatchNorm2d(64)
        self.conv3_29 = nn.Conv2d(64, 128, (1, 1))  # output (n_examples, 48, 17, 30)
        self.convnorm3_29 = nn.BatchNorm2d(128)

        self.conv3_30 = nn.Conv2d(128, 64, (1, 1))  # output (n_examples, 64, 17, 30)
        self.convnorm3_30 = nn.BatchNorm2d(64)
        self.conv3_31 = nn.Conv2d(64, 64, (1, 3))  # output (n_examples, 48, 15, 26)
        self.convnorm3_31 = nn.BatchNorm2d(64)
        self.conv3_32 = nn.Conv2d(64, 256, (1, 1))  # output (n_examples, 48, 17, 30)
        self.convnorm3_32 = nn.BatchNorm2d(256)

        self.conv3_33 = nn.Conv2d(256, 64, (1, 1))  # output (n_examples, 64, 17, 30)
        self.convnorm3_33 = nn.BatchNorm2d(64)
        self.conv3_34 = nn.Conv2d(64, 64, (3, 1))  # output (n_examples, 48, 13, 26)
        self.convnorm3_34 = nn.BatchNorm2d(64)
        self.conv3_35 = nn.Conv2d(64, 256, (1, 1))  # output (n_examples, 48, 17, 30)
        self.convnorm3_35 = nn.BatchNorm2d(256)
        self.pool3_35 = nn.MaxPool2d((2, 2), padding=(1, 0))  # output (n_examples, 64, 7, 13)

        self.conv7 = nn.Conv2d(768, 256, (1, 1))  # output (n_examples, 72, 7, 13)
        self.convnorm7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, (3, 3), groups=1)  # output (n_examples, 64, 5, 11)
        self.convnorm8 = nn.BatchNorm2d(256)
        self.conv9 = nn.Conv2d(256, 512, (1, 1))  # output (n_examples, 72, 9, 16)
        self.convnorm9 = nn.BatchNorm2d(512)
        self.linear1 = nn.Linear(512 * 5 * 11, 128)  # input will be flattened to (n_examples, 64*7*14)
        self.linear1_bn = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(DROPOUT1)
        self.linear2 = nn.Linear(128, 128)
        self.linear2_bn = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(DROPOUT2)
        self.linear3 = nn.Linear(128, 4)
        self.act = nn.ReLU(inplace=True)
        self.act2 = nn.Softmax(dim=1)

    def forward(self, x):
        x_1 = self.convnorm1_1(self.act(self.conv1_1(x.permute(0, 3, 1, 2))))
        x_1 = self.pool1_2(self.convnorm1_2(self.act(self.conv1_2(x_1))))
        x_1 = self.convnorm1_3(self.act(self.conv1_3(x_1)))
        x_1 = self.convnorm1_4(self.act(self.conv1_4(x_1)))
        x_1 = self.convnorm1_5(self.act(self.conv1_5(x_1)))
        x_1 = self.convnorm1_6(self.act(self.conv1_6(x_1)))
        x_1 = self.convnorm1_7(self.act(self.conv1_7(x_1)))
        x_1 = self.pool1_8(self.convnorm1_8(self.act(self.conv1_8(x_1))))
        x_1 = self.convnorm1_9(self.act(self.conv1_9(x_1)))
        x_1 = self.convnorm1_10(self.act(self.conv1_10(x_1)))
        x_1 = self.convnorm1_11(self.act(self.conv1_11(x_1)))
        x_1 = self.convnorm1_12(self.act(self.conv1_12(x_1)))
        x_1 = self.convnorm1_13(self.act(self.conv1_13(x_1)))
        x_1 = self.convnorm1_14(self.act(self.conv1_14(x_1)))
        x_1 = self.convnorm1_15(self.act(self.conv1_15(x_1)))
        x_1 = self.convnorm1_16(self.act(self.conv1_16(x_1)))
        x_1 = self.convnorm1_17(self.act(self.conv1_17(x_1)))
        x_1 = self.convnorm1_18(self.act(self.conv1_18(x_1)))
        x_1 = self.convnorm1_19(self.act(self.conv1_19(x_1)))
        x_1 = self.convnorm1_20(self.act(self.conv1_20(x_1)))
        x_1 = self.convnorm1_21(self.act(self.conv1_21(x_1)))
        x_1 = self.convnorm1_22(self.act(self.conv1_22(x_1)))
        x_1 = self.convnorm1_23(self.act(self.conv1_23(x_1)))
        x_1 = self.convnorm1_24(self.act(self.conv1_24(x_1)))
        x_1 = self.pool1_25(self.convnorm1_25(self.act(self.conv1_25(x_1))))


        x_2 = self.convnorm2_1(self.act(self.conv2_1(x.permute(0, 3, 1, 2))))
        x_2 = self.pool2_2(self.convnorm2_2(self.act(self.conv2_2(x_2))))
        x_2 = self.convnorm2_3(self.act(self.conv2_3(x_2)))
        x_2 = self.convnorm2_4(self.act(self.conv2_4(x_2)))
        x_2 = self.convnorm2_5(self.act(self.conv2_5(x_2)))
        x_2 = self.convnorm2_6(self.act(self.conv2_6(x_2)))
        x_2 = self.convnorm2_7(self.act(self.conv2_7(x_2)))
        x_2 = self.pool2_8(self.convnorm2_8(self.act(self.conv2_8(x_2))))
        x_2 = self.convnorm2_9(self.act(self.conv2_9(x_2)))
        x_2 = self.convnorm2_10(self.act(self.conv2_10(x_2)))
        x_2 = self.convnorm2_11(self.act(self.conv2_11(x_2)))
        x_2 = self.convnorm2_12(self.act(self.conv2_12(x_2)))
        x_2 = self.convnorm2_13(self.act(self.conv2_13(x_2)))
        x_2 = self.convnorm2_14(self.act(self.conv2_14(x_2)))
        x_2 = self.convnorm2_15(self.act(self.conv2_15(x_2)))
        x_2 = self.convnorm2_16(self.act(self.conv2_16(x_2)))
        x_2 = self.convnorm2_17(self.act(self.conv2_17(x_2)))
        x_2 = self.convnorm2_18(self.act(self.conv2_18(x_2)))
        x_2 = self.convnorm2_19(self.act(self.conv2_19(x_2)))
        x_2 = self.convnorm2_20(self.act(self.conv2_20(x_2)))
        x_2 = self.convnorm2_21(self.act(self.conv2_21(x_2)))
        x_2 = self.convnorm2_22(self.act(self.conv2_22(x_2)))
        x_2 = self.convnorm2_23(self.act(self.conv2_23(x_2)))
        x_2 = self.convnorm2_24(self.act(self.conv2_24(x_2)))
        x_2 = self.pool2_25(self.convnorm2_25(self.act(self.conv2_25(x_2))))

        x_3 = self.convnorm3_1(self.act(self.conv3_1(x.permute(0, 3, 1, 2))))
        x_3 = self.convnorm3_2(self.act(self.conv3_2(x_3)))
        x_3 = self.convnorm3_3(self.act(self.conv3_3(x_3)))
        x_3 = self.convnorm3_4(self.act(self.conv3_4(x_3)))
        x_3 = self.pool3_5(self.convnorm3_5(self.act(self.conv3_5(x_3))))
        x_3 = self.convnorm3_6(self.act(self.conv3_6(x_3)))
        x_3 = self.convnorm3_7(self.act(self.conv3_7(x_3)))
        x_3 = self.convnorm3_8(self.act(self.conv3_8(x_3)))
        x_3 = self.convnorm3_9(self.act(self.conv3_9(x_3)))
        x_3 = self.convnorm3_10(self.act(self.conv3_10(x_3)))
        x_3 = self.convnorm3_11(self.act(self.conv3_11(x_3)))
        x_3 = self.convnorm3_12(self.act(self.conv3_12(x_3)))
        x_3 = self.convnorm3_13(self.act(self.conv3_13(x_3)))
        x_3 = self.convnorm3_14(self.act(self.conv3_14(x_3)))
        x_3 = self.convnorm3_15(self.act(self.conv3_15(x_3)))
        x_3 = self.convnorm3_16(self.act(self.conv3_16(x_3)))
        x_3 = self.convnorm3_17(self.act(self.conv3_17(x_3)))
        x_3 = self.convnorm3_18(self.act(self.conv3_18(x_3)))
        x_3 = self.convnorm3_19(self.act(self.conv3_19(x_3)))
        x_3 = self.pool3_20(self.convnorm3_20(self.act(self.conv3_20(x_3))))
        x_3 = self.convnorm3_21(self.act(self.conv3_21(x_3)))
        x_3 = self.convnorm3_22(self.act(self.conv3_22(x_3)))
        x_3 = self.convnorm3_23(self.act(self.conv3_23(x_3)))
        x_3 = self.convnorm3_24(self.act(self.conv3_24(x_3)))
        x_3 = self.convnorm3_25(self.act(self.conv3_25(x_3)))
        x_3 = self.convnorm3_26(self.act(self.conv3_26(x_3)))
        x_3 = self.convnorm3_27(self.act(self.conv3_27(x_3)))
        x_3 = self.convnorm3_28(self.act(self.conv3_28(x_3)))
        x_3 = self.convnorm3_29(self.act(self.conv3_29(x_3)))
        x_3 = self.convnorm3_30(self.act(self.conv3_30(x_3)))
        x_3 = self.convnorm3_31(self.act(self.conv3_31(x_3)))
        x_3 = self.convnorm3_32(self.act(self.conv3_32(x_3)))
        x_3 = self.convnorm3_33(self.act(self.conv3_33(x_3)))
        x_3 = self.convnorm3_34(self.act(self.conv3_34(x_3)))
        x_3 = self.pool3_35(self.convnorm3_35(self.act(self.conv3_35(x_3))))
        # print(x_1.shape, x_2.shape, x_3.shape)
        x = torch.cat([x_1, x_2, x_3], dim=1)

        x = self.convnorm7(self.act(self.conv7(x)))
        x = self.convnorm8(self.act(self.conv8(x)))
        x = self.convnorm9(self.act(self.conv9(x)))
        x = self.drop1(self.linear1_bn(self.act(self.linear1(x.view(len(x), -1)))))
        x = self.drop2(self.linear2_bn(self.act(self.linear2(x))))
        return self.linear3(x)


def weights_init_1(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        torch.nn.init.normal_(m.bias.data, 0, 0.02)
    elif classname.find('linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        torch.nn.init.normal_(m.bias.data, 0, 0.02)


def weights_init_2(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        torch.nn.init.normal_(m.bias.data, 0, 0.02)
    elif classname.find('linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        torch.nn.init.normal_(m.bias.data, 0, 0.02)


def weights_init_3(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data)
        torch.nn.init.normal_(m.bias.data, 0, 0.02)
    elif classname.find('linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data)
        torch.nn.init.normal_(m.bias.data, 0, 0.02)


def weights_init_4(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.kaiming_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data, 0, 0.02)
    elif classname.find('linear') != -1:
        torch.nn.init.kaiming_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data, 0, 0.02)


# %% -------------------------------------- Data Prep ------------------------------------------------------------------
x_total = np.load("x_total_110_164_9_6.npy")
target_total = np.load("target_total_110_164_9_6.npy")
target_category_coding = np.load("target_category_coding_110_164_9_6.npy")
print(x_total.shape, target_total.shape)

x_train, x_test, y_train, y_validation = train_test_split(x_total, target_total, random_state=SEED, test_size=0.2,
                                                          stratify=target_category_coding)
x_train, x_test, y_train, y_test = train_test_split(x_total, target_category_coding, random_state=SEED, test_size=0.2,
                                                    stratify=target_category_coding)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# np.save("x_validation_120_180_9_9.npy", x_test)
# np.save("y_validation_120_180_9_9.npy", y_test)
x_train, y_train = torch.tensor(x_train).float().to(torch.device("cpu")), torch.tensor(y_train).to(torch.device("cpu"))
y_train = y_train.long()
print(y_train.dtype)
x_train.requires_grad = True
y_train.requires_grad = False
x_test, y_test, y_validation = torch.tensor(x_test).float().to(torch.device("cpu")), torch.tensor(y_test).to(
    torch.device("cpu")), torch.tensor(y_validation).to(torch.device("cpu"))
y_test = y_test.long()
x_test.requires_grad = False
y_test.requires_grad = False
y_validation.requires_grad = False
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_train.dtype, y_train.dtype)

# %% -------------------------------------- Training Prep ----------------------------------------------------------
# Load the model to GPU
model = CNN().to(device)

# Initialize the weights
model.apply(weights_init_1)  # weights_ini_1 weights_ini_2 weights_ini_3 weights_ini_4

# Specify optimizers
# optimizer = torch.optim.SGD(model.parameters(), lr=LR)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Specify learning rate schedulers
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=30, verbose=True, min_lr=8e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=130)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1, eta_min=5e-7)
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.01, step_size_up=50, mode='triangular')

# Specify different criterion
criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.CrossEntropyLoss(weight=torch.tensor(np.array([3.5293, 20.0209,  2.9280,  3.0752])).float().to(device))
best_model_AUC = None  # [ 3.5293, 20.0209,  2.9280,  3.0752][9, 50, 2.93, 5]
counter = 0

# %% -------------------------------------- Training Loop ----------------------------------------------------------
print("Starting training loop...")
Train_loss = torch.zeros(N_EPOCHS, device=torch.device("cpu"))
Test_loss = torch.zeros(N_EPOCHS, device=torch.device("cpu"))
for epoch in range(N_EPOCHS):

    loss_train = 0
    model = model.to(torch.device("cpu"))
    model.train()
    for batch in range(len(x_train) // BATCH_SIZE + 1):
        # index = [i for i in range(batch * BATCH_SIZE, np.min(((batch + 1) * BATCH_SIZE, len(x_train))))]
        index = [i for i in range(0, len(x_train))]
        np.random.shuffle(index)
        inds = slice(batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE)
        optimizer.zero_grad()
        logits = model(x_train[index][inds].to(device))
        loss = criterion2(logits, y_train[index][inds].to(device))
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_train += loss.item()
    # Calculate test loss
    model.eval()
    with torch.no_grad():
        model = model.to(torch.device("cpu"))
        y_test_pred = model(x_test)
        criterion_c1 = nn.CrossEntropyLoss()
        criterion_c2 = nn.CrossEntropyLoss(weight=torch.tensor(np.array([3.53, 40, 2.93, 3.08])).float().to(torch.device("cpu")))
        loss = criterion_c1(y_test_pred, y_test)
        loss_test = loss.item()
        test_AUC = roc_auc_score(y_validation.cpu().numpy(), y_test_pred.cpu().numpy())
        Train_loss[epoch] = loss_train / (len(x_train) // BATCH_SIZE + 1)
        Test_loss[epoch] = loss_test
        # scheduler.step(loss_test)
    print("Epoch {} | Train Loss {:.5f} - - Test Loss {:.5f} - - Test AUC {:.5f}".format(epoch, loss_train / (
                len(x_train) // BATCH_SIZE + 1),
                                                                                         loss_test, test_AUC))

    # Early stopping
    if best_model_AUC is None:
        best_model_AUC = test_AUC
        print(f'Validation AUC increased ({0:.6f} --> {best_model_AUC:.6f}).  Saving model as ' + '\"' + PATH +
              '\"')
        torch.save(model.state_dict(), PATH)
    elif test_AUC > best_model_AUC:
        print(f'Validation AUC increased ({best_model_AUC:.6f} --> {test_AUC:.6f}).  Saving model as ' + '\"' + PATH
              + '\"')
        best_model_AUC = test_AUC
        torch.save(model.state_dict(), PATH)
        counter = 0
    else:
        counter += 1
        print(f'EarlyStopping counter: {counter} out of {patience}')
        if counter >= patience:
            print("Early stopping")
            break

fig = plt.figure()
left, bottom, width, height = 0.11, 0.1, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height])
ax.plot(np.mat(range(N_EPOCHS)).T, Train_loss.cpu().numpy(), label="Train loss")
ax.plot(np.mat(range(N_EPOCHS)).T, Test_loss.cpu().numpy(), label="Test loss")
ax.set_title('Loss vs Epoch')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
plt.legend()
plt.show()

model = CNN().to(torch.device("cpu"))
# PATH = 'model_yongchaoqiaocombination.pt'
# PATH2 = 'model_yongchaoqiao9day5_2.pt'
model.load_state_dict(torch.load(PATH, map_location=device))
model.eval()
y_pred = model(x_test).detach().numpy()
test_AUC = roc_auc_score(y_validation.cpu().numpy(), y_pred)
y_pred = np.argmax(y_pred, axis=1)
print("Test AUC")
print(test_AUC)
print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred))
print('Classification Report')
target_names = ['Healthy', "Multiple diseases", 'rust', 'scab']
print(classification_report(y_test, y_pred, target_names=target_names))