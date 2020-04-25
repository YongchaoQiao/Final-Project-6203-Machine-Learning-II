# %% --------------------------------------- Imports -------------------------------------------------------------------
import torch
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
# %% --------------------------------------- Set-Up --------------------------------------------------------------------
device = torch.device("cuda:0")
SEED = 413
torch.manual_seed(413)
torch.cuda.manual_seed_all(413)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 0.15  # 0.15 0.1 0.01
N_EPOCHS = 300
BATCH_SIZE = 430  # 200, 412, 430(Max memory)
DROPOUT1 = 0.8
DROPOUT2 = 0.7
patience = 500


# %% -------------------------------------- CNN Class ------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, (1, 1))  # output (n_examples, 8, 110, 164)
        self.convnorm1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d((2, 2))  # output (n_examples, 8, 55, 82)
        self.conv2 = nn.Conv2d(8, 16, (1, 1))  # output (n_examples, 16, 55, 82)
        self.convnorm2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, (1, 3), groups=1)  # output (n_examples, 32, 55, 80)
        self.convnorm3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d((2, 2), padding=(1, 0))  # output (n_examples, 32, 28, 40)
        self.conv4 = nn.Conv2d(32, 48, (1, 1))  # output (n_examples, 48, 28, 40)
        self.convnorm4 = nn.BatchNorm2d(48)
        self.conv5 = nn.Conv2d(48, 64, (3, 1))  # output (n_examples, 64, 26, 40)
        self.convnorm5 = nn.BatchNorm2d(64)
        self.pool5 = nn.MaxPool2d((2, 2))  # output (n_examples, 64, 13, 20)
        self.conv6 = nn.Conv2d(64, 72, (1, 1))  # output (n_examples, 72, 13, 20)
        self.convnorm6 = nn.BatchNorm2d(72)
        self.conv7 = nn.Conv2d(72, 64, (3, 3), groups=1)  # output (n_examples, 64, 11, 18)
        self.convnorm7 = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(64, 72, (1, 1))  # output (n_examples, 72, 11, 18)
        self.convnorm8 = nn.BatchNorm2d(72)
        self.conv9 = nn.Conv2d(72, 64, (3, 3), groups=1)  # output (n_examples, 64, 9, 16)
        self.convnorm9 = nn.BatchNorm2d(64)
        self.conv10 = nn.Conv2d(64, 72, (1, 1))  # output (n_examples, 72, 9, 16)
        self.convnorm10 = nn.BatchNorm2d(72)
        self.conv11 = nn.Conv2d(72, 64, (3, 3), groups=1)  # output (n_examples, 64, 7, 14)
        self.convnorm11 = nn.BatchNorm2d(64)
        self.linear1 = nn.Linear(64 * 7 * 14, 64)  # input will be flattened to (n_examples, 64*7*14)
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

x_train, x_test, y_train, y_validation = train_test_split(x_total, target_total, random_state=SEED, test_size=0.3,
                                                          stratify=target_category_coding)
x_train, x_test, y_train, y_test = train_test_split(x_total, target_category_coding, random_state=SEED, test_size=0.3,
                                                    stratify=target_category_coding)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# np.save("x_validation_120_180_9_9.npy", x_test)
# np.save("y_validation_120_180_9_9.npy", y_test)
x_train, y_train = torch.tensor(x_train).float().to(device), torch.tensor(y_train).to(device)
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
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1, eta_min=5e-7)
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.01, step_size_up=50, mode='triangular')

# Specify different criterion
criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.CrossEntropyLoss(weight=torch.tensor(np.array([3.53, 19.99, 2.93, 3.08])).float().to(device))
best_model_AUC = None  # [ 3.5293, 20.0209,  2.9280,  3.0752]
counter = 0
PATH = 'model_yongchaoqiaofinal.pt'

# %% -------------------------------------- Training Loop ----------------------------------------------------------
print("Starting training loop...")
Train_loss = torch.zeros(N_EPOCHS, device=torch.device("cpu"))
Test_loss = torch.zeros(N_EPOCHS, device=torch.device("cpu"))
for epoch in range(N_EPOCHS):

    loss_train = 0
    model = model.to(torch.device("cuda"))
    model.train()
    for batch in range(len(x_train) // BATCH_SIZE + 1):
        # index = [i for i in range(batch * BATCH_SIZE, np.min(((batch + 1) * BATCH_SIZE, len(x_train))))]
        index = [i for i in range(0, len(x_train))]
        np.random.shuffle(index)
        inds = slice(batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE)
        optimizer.zero_grad()
        logits = model(x_train[index][inds])
        loss = criterion1(logits, y_train[index][inds])
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
        criterion_c2 = nn.CrossEntropyLoss(weight=torch.tensor(np.array([3.53, 19.99, 2.93, 3.08])).float().to(torch.device("cpu")))
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
PATH = 'model_yongchaoqiaofinal.pt'
# PATH2 = 'model_yongchaoqiao9day5_2.pt'
model.load_state_dict(torch.load(PATH, map_location=device))
model.eval()
y_pred = model(x_test).detach().numpy()
test_AUC = roc_auc_score(y_validation.cpu().numpy(), y_pred)
y_pred = np.argmax(y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred))
print('Classification Report')
target_names = ['Healthy', "Multiple diseases", 'rust', 'scab']
print(classification_report(y_test, y_pred, target_names=target_names))