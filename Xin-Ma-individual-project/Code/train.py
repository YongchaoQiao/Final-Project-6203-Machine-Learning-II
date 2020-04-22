# %% --------------------------------------- Imports -------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 0.05
N_EPOCHS = 50
BATCH_SIZE = 64
DROPOUT = 0.5

# %% -------------------------------------- CNN Class ------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, (3, 3),padding=1)  # output (n_examples, 16, 50, 50)
        self.convnorm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 64, (3, 3),padding=1) # output (n_examples, 64, 50, 50)
        self.convnorm2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d((2,2))                  # output (n_examples, 64, 25, 25)
        self.conv3 = nn.Conv2d(64,128,(3,3),padding=1)    # output (n_examples, 128, 25, 25)
        self.convnorm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128,256,(3,3),padding=1)   # output (n_examples, 256, 25, 25)
        self.convnorm4 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d((2,2))                  # output (n_examples, 256, 12, 12)
        self.conv5 = nn.Conv2d(256,256,(3,3),padding=1)   # output (n_examples, 256, 12, 12)
        self.convnorm5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256,512,(3,3),padding=1)  # output (n_examples, 512, 12, 12)
        self.convnorm6 = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d((2,2))                 # output (n_examples, 512, 6, 6)
        self.linear1 = nn.Linear(512*6*6, 400)
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


# %% -------------------------------------- Data Prep ------------------------------------------------------------------
x_train,y_train=np.load('/home/ubuntu/Deep_Learning/final/data/x_train.npy'),np.load('/home/ubuntu/Deep_Learning/final/data/y_train.npy')
x_val,y_val=np.load('/home/ubuntu/Deep_Learning/final/data/x_val.npy'),np.load('/home/ubuntu/Deep_Learning/final/data/y_val.npy')

x_train,y_train=torch.from_numpy(x_train).view(len(x_train),3,50,50).float().to(device),torch.from_numpy(y_train).long().to(device)
x_train.requires_grad=True
x_val,y_val=torch.from_numpy(x_val).view(len(x_val),3,50,50).float().to(device),torch.from_numpy(y_val).long().to(device)

# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = CNN().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=LR,momentum=0.9)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[25,50,75], gamma=0.1)
criterion = nn.CrossEntropyLoss()
# %% -------------------------------------- Training Loop ----------------------------------------------------------
print("Starting training loop...")
best_loss = float('inf')
train_loss, val_loss = [], []
for epoch in range(N_EPOCHS):

    loss_train = 0
    model.train()
    for batch in range(len(x_train)//BATCH_SIZE + 1):
        inds = slice(batch*BATCH_SIZE, (batch+1)*BATCH_SIZE)
        optimizer.zero_grad()
        logits = model(x_train[inds])
        loss = criterion(logits, y_train[inds])
        loss.backward()
        optimizer.step()
        loss_train += loss.item()

    train_loss.append(loss_train)

    model.eval()
    with torch.no_grad():
        y_val_pred = model(x_val)
        loss = criterion(y_val_pred, y_val)
        loss_val = loss.item()
        val_loss.append(loss_val)

    if loss_val < best_loss:
        best_loss = loss_val
        torch.save(model.state_dict(), 'model1.pt')
        print('model saved at epoch=%.i' % epoch )

    print("Epoch {} | Train Loss {:.5f} - Val Loss {:.5f}".format(epoch, loss_train / BATCH_SIZE, loss_val))

plt.plot(range(N_EPOCHS),np.log(train_loss))
plt.show()
plt.plot(range(N_EPOCHS),np.log(val_loss))
plt.show()


'''
# %% -------------------------------------- CNN Class ------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, (3, 3),padding=1)  # output (n_examples, 32, 100, 100)
        self.convnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,32,(3,3),padding=1)   # output (n_examples, 32, 100, 100)
        self.convnorm2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2,2))               # output (n_examples, 32, 50, 50)
        self.drop1 = nn.Dropout(0.2)
        self.conv3 = nn.Conv2d(32, 64, (3, 3),padding=1)  # output (n_examples, 64, 50, 50)
        self.convnorm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64,64,(3,3),padding=1)   # output (n_examples, 64, 50, 50)
        self.convnorm4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2,2))           # output (n_examples, 64, 25, 25)
        self.drop2 = nn.Dropout(0.2)
        self.conv5 = nn.Conv2d(64, 128, (3, 3),padding=1)  # output (n_examples, 128, 25, 25)
        self.convnorm5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128,128,(3,3),padding=1)   # output (n_examples, 128, 25, 25)
        self.convnorm6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d((2,2))           # output (n_examples, 128, 12, 12)
        self.drop3 = nn.Dropout(0.2)
        self.linear1 = nn.Linear(128*12*12, 400)
        self.linear1_bn = nn.BatchNorm1d(400)
        self.drop4 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(400, 7)
        self.act = torch.relu

    def forward(self, x):
        x = self.convnorm1(self.act(self.conv1(x)))
        x = self.drop1(self.pool1(self.convnorm2(self.act(self.conv2(x)))))
        x = self.convnorm3(self.act(self.conv3(x)))
        x = self.drop2(self.pool2(self.convnorm4(self.act(self.conv4(x)))))
        x = self.convnorm5(self.act(self.conv5(x)))
        x = self.drop3(self.pool3(self.convnorm6(self.act(self.conv6(x)))))
        x = self.drop4(self.linear1_bn(self.act(self.linear1(x.view(len(x), -1)))))
        return self.linear2(x)

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
x_train,y_train=np.load('x_train.npy'),np.load('y_train.npy')
x_test,y_test=np.load('x_test.npy'),np.load('y_test.npy')

x_train,y_train=torch.from_numpy(x_train).view(len(x_train),3,100,100).float().to(device),torch.from_numpy(y_train).float().to(device)
x_train.requires_grad=True
x_test,y_test=torch.from_numpy(x_test).view(len(x_test),3,100,100).float().to(device),torch.from_numpy(y_test).float().to(device)

# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = CNN().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.5)
criterion1 = nn.BCEWithLogitsLoss()
criterion2 = nn.BCELoss()
sigmoid=nn.Sigmoid()
# %% -------------------------------------- Training Loop ----------------------------------------------------------
print("Starting training loop...")
best_loss = float('inf')
train_loss, test_loss = [], []
for epoch in range(N_EPOCHS):

    loss_train = 0
    model.train()
    for batch in range(len(x_train)//BATCH_SIZE + 1):
        inds = slice(batch*BATCH_SIZE, (batch+1)*BATCH_SIZE)
        optimizer.zero_grad()
        logits = model(x_train[inds])
        loss = criterion1(logits, y_train[inds])
        loss.backward()
        optimizer.step()
        loss_train += loss.item()

    train_loss.append(loss_train)

    model.eval()
    with torch.no_grad():
        y_test_pred = model(x_test)
        y_test_pred = sigmoid(y_test_pred)
        loss = criterion2(y_test_pred, y_test)
        loss_test = loss.item()
        test_loss.append(loss_test)

    if loss_test < best_loss:
        best_loss = loss_test
        torch.save(model.state_dict(), 'model_maxin3253.pt')
        print('model saved at epoch=%.i' % epoch )

    print("Epoch {} | Train Loss {:.5f} - Test Loss {:.5f}".format(epoch, loss_train / BATCH_SIZE, loss_test))

plt.plot(range(N_EPOCHS),np.log(train_loss))
plt.plot(range(N_EPOCHS),np.log(test_loss))
plt.show()
'''