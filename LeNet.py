import torch
import torch.nn as nn
import torch.nn.functional as Func
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np

batch_size = 100
learning_rate = 0.001

train_dataset = dset.MNIST(root="mnist_data", train=True, transform=transforms.ToTensor(), download=True)  # 抓取训练集
test_dataset = dset.MNIST(root="mnist_data", train=False, transform=transforms.ToTensor(), download=True)  # 抓取测试集


class LeNet(nn.Module):  # 自定义LeNet 此处没用Sequential

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)  # (输入为28*28,LeNet c1 结果为28*28,故需要pad)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=0)  # 此处与原LeNet不一样, 原c3较为复杂
        self.fc1 = nn.Linear(16*5*5, 120)  # c5
        self.fc2 = nn.Linear(120, 84)  # f6
        self.fc3 = nn.Linear(84, 10)  # output

    def forward(self, x):
        x = Func.max_pool2d(Func.relu(self.conv1(x)), 2)  # window size = 2*2
        x = Func.max_pool2d(Func.relu(self.conv2(x)), 2)  # window size = 2*2
        x = x.view(x.size(0), -1)  # 平铺
        x = Func.relu(self.fc1(x))  # 全连接后用Relu激活
        x = Func.relu(self.fc2(x))
        x = Func.softmax(self.fc3(x), dim=1)  # 最后用Softmax激活, 为了交叉熵损失函数
        return x


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)  #训练集加载'''
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

net = LeNet()  # 实例化'''
criterion = nn.CrossEntropyLoss()  # 损失函数选择交叉熵'''



# 使用Adam
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # 优化器
num_epochs = 5  # 五次Epoch

'''训练'''
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images)  # 创造Variable
        labels = Variable(labels)

        optimizer.zero_grad()  # 梯度置零
        outputs = net(images)  # 进入网络

        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 后向传播

        optimizer.step()  # 参数更新

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' % (epoch + 1, num_epochs, i + 1, len(train_dataset)
                                                               // batch_size, loss.data.item()))

'''测试'''
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images)
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()


acc = (100 * float(correct) / total)
print('Accuracy: %.2f %%' %acc)


torch.save(net, './ModelBackup/MNIST_lenet_mode_%d_%.2f.pkl'%(epoch, acc))


