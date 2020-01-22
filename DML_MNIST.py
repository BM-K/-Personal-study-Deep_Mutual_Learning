from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
# 필요한 모듈 import

batch_size = 64
# batch size 64
train_dataset = datasets.MNIST(root='./',
                              train=True,
                              transform=transforms.ToTensor(),
                              download=True)

# train dataset Tensor 형태로 가져오기. 학습용, 없으면 download
test_dataset =datasets.MNIST(root='./',
                            train=False,
                            transform=transforms.ToTensor())
# test dataset Tensor 형태로 가져오기.

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)
# train data batch size 만큼 읽어오기
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                         batch_size=batch_size,
                                         shuffle=False)
# test data batch size 만큼 읽어오기

class Net(nn.Module):
    # Network
    def __init__(self):
        super(Net, self).__init__()
        # 부모 초기화
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # convolution layer. input 1, output 10, kernel 5*5 (10개의 서로 다른 5*5 필터를 통해서 새로운
        # 결과를 만드는 layer)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # convolution layer. input 10, output 20, kernel 5*5
        self.mp = nn.MaxPool2d(2)
        # maxpooling 2*2
        self.fc1 = nn.Linear(320, 10)
        # full connected input 320, output 10

    def forward(self, x):
        in_size = x.size(0)
        # 들어오는 사이즈 저장
        x = F.relu(self.mp(self.conv1(x)))
        # convoultion layer -> maxpooling -> relu
        x = F.relu(self.mp(self.conv2(x)))
        # convoultion layer -> maxpooling -> relu
        x = x.view(in_size, -1)
        # size matching
        x = self.fc1(x)
        # fully connected layer
        #F.log_softmax(x)
        return x


def train(epoch, mutual_model, mutual_optim):
    for i in range(model_num):
        mutual_model[i].train()
    # 모델 훈련
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        outputs = []
        for model in mutual_model:
            outputs.append(model(data))

        for i in range(model_num):
            ce_loss = criterion(outputs[i], target)
            kl_loss = 0

            for j in range(model_num):
                if i != j:  # model이 달라졌을 때,
                    kl_loss += loss_kl(F.log_softmax(outputs[i]),
                                       F.softmax(Variable(outputs[j])))
            loss = ce_loss + kl_loss / (model_num - 1)
            mutual_optim[i].zero_grad()
            loss.backward(retain_graph=True)
            mutual_optim[i].step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
            # print


def test(mutual_model):
    # 모델 test
    test_loss = []
    correct = []
    for i in range(model_num):
        test_loss.append(0)
        correct.append(0)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = Variable(data), Variable(target)
            outputs = []
            for i in range(model_num):
                model = mutual_model[i]
                model.eval()
                outputs.append(model(data))
            for i in range(model_num):
                ce_loss = criterion(outputs[i], target)
                kl_loss = 0
                for j in range(model_num):
                    if i != j:
                        kl_loss += loss_kl(F.log_softmax(outputs[i]),
                                           F.softmax(Variable(outputs[j])))
                loss = ce_loss + kl_loss / (model_num - 1)

                test_loss[i] += loss
                pred = outputs[i].data.max(1, keepdim=True)[1]
                correct[i] += pred.eq(target.data.view_as(pred)).cpu().sum()

    for i in range(model_num):
        print("model",i, test_loss[i].item()/(10000/64), "&", correct[i].item(),"/",len(test_loader.dataset))


if __name__ == '__main__':
    criterion = nn.CrossEntropyLoss()
    model_num = 2
    mutual_model = []
    mutual_optim = []

    for i in range(model_num):
        model = Net()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
        mutual_model.append(model)
        mutual_optim.append(optimizer)

    loss_kl = nn.KLDivLoss(reduction='batchmean')

    for epoch in range(1, 10):
        train(epoch, mutual_model, mutual_optim)
    test(mutual_model)
    # 스토케스틱 경사하강법. learning rate 0.01, momentum 0.5