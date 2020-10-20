# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 22:00:45 2020

@author: admin
"""


    #  首先当然肯定要导入torch和torchvision，至于第三个是用于进行数据预处理的模块
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
from tqdm import tqdm
import numpy as np
Batch_size = 4
transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=Batch_size,
                                              shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=Batch_size,
                                             shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
    'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 首先是调用Variable、 torch.nn、torch.nn.functional
from torch.autograd import Variable   # 这一步还没有显式用到variable，但是现在写在这里也没问题，后面会用到
import torch.nn as nn
import torch.nn.functional as F
 
 
class Net(nn.Module):                 
    def __init__(self):    
        super(Net, self).__init__()   
        self.conv1 = nn.Conv2d(3, 6, 5)       
        self.pool = nn.MaxPool2d(2, 2)        
        self.conv2 = nn.Conv2d(6, 16, 5)      
        self.fc1 = nn.Linear(16 * 5 * 5, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
 
    def forward(self, x):                
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)  #
                                   
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        features = x
        x = self.fc3(x)
        return features,x
 
 
 # 和python中一样，类定义完之后实例化就很简单了，我们这里就实例化了一个net
net = Net()
print(net)

import torch.optim as optim          #导入torch.potim模块
 
criterion = nn.CrossEntropyLoss()    #同样是用到了神经网络工具箱 nn 中的交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(30):  # loop over the dataset multiple times 指定训练一共要循环几个epoch
    feature_all = []
    label_all = []     
    running_loss = 0.0  #定义一个变量方便我们对loss进行输出
    total = 0.0
    correct = 0.0
    print('\n epoch:{}'.format(epoch))
    k = 0
    for i, data in tqdm(enumerate(trainloader, 0)): # 这里我们遇到了第一步中出现的trailoader，代码传入数据
                                                  # enumerate是python的内置函数，既获得索引也获得数据，详见下文
            # get the inputs
        inputs, labels = data   # data是从enumerate返回的data，包含数据和标签信息，分别赋值给inputs和labels
     
            # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels) # 将数据转换成Variable，第二步里面我们已经引入这个模块
                                                                # 所以这段程序里面就直接使用了，下文会分析
            # zero the parameter gradients
        optimizer.zero_grad()                # 要把梯度重新归零，因为反向传播过程中梯度会累加上一次循环的梯度
     
            # forward + backward + optimize      
        feature,outputs = net(inputs)                # 把数据输进网络net，这个net()在第二步的代码最后一行我们已经定义了
        _,predicted = torch.max(outputs,1)
        total += labels.size(0)
        correct += torch.sum(predicted == labels)
        k+=1    
        feature = feature.detach().numpy()
        label = labels.detach().numpy()
        feature_all.append(feature)
        label_all.append(label)
        loss = criterion(outputs, labels)    # 计算损失值,criterion我们在第三步里面定义了
        loss.backward()                      # loss进行反向传播，下文详解
        optimizer.step()                     # 当执行反向传播之后，把优化器的参数进行更新，以便进行下一轮
     
            # print statistics                   # 这几行代码不是必须的，为了打印出loss方便我们看而已，不影响训练过程
        running_loss += loss.item()         # 从下面一行代码可以看出它是每循环0-1999共两千次才打印一次
        if i % 2000 == 1999:    # print every 2000 mini-batches   所以每个2000次之类先用running_loss进行累加
            print('[%d, %5d] loss: %.3f' %
                 (epoch + 1, i + 1, running_loss / 2000))  # 然后再除以2000，就得到这两千次的平均损失值
            running_loss = 0.0               # 这一个2000次结束后，就把running_loss归零，下一个2000次继续使用
    print('accracy:{}%'.format(torch.round(correct*100/total),'4f'))
print('Finished Training')





def merge(metrix):
    feature_part1 = metrix[0]
    for batch in metrix[1:]:
        feature_part1 = np.concatenate([feature_part1,batch],axis = 0)
    return feature_part1

feature_part = feature_all[:100]
feature_part = merge(feature_part)
label_part = merge(label_all[:100])

import os
import scipy.io as scio
if not os.path.exists('./data/result/cifar'):
        os.mkdir('./data/result/cifar')
save1="./data/result/cifar/feature_epoch_{}_(loss{}).mat".format(epoch + 1, format(loss.data,'.4f'))
save2="./data/result/cifar/label_epoch_{}.mat".format(epoch + 1)
scio.savemat(save1, {'feature':feature_part})
scio.savemat(save2, {'label':label_part})
# 下面是测试部分
def imshow(img):
   img = img /2+0.5     # 非标准的（unnormalized）
   npimg = img.numpy()
   plt.imshow(np.transpose(npimg, (1, 2, 0)))
   plt.show()
   
   
   
dataiter = iter(testloader)      # 创建一个python迭代器，读入的是我们第一步里面就已经加载好的testloader
images, labels = dataiter.next() # 返回一个batch_size的图片，根据第一步的设置，应该是4张
     
# print images
imshow(torchvision.utils.make_grid(images))  # 展示这四张图片
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4))) # python字符串格式化 ' '.join表示用空格来连接后面的字符串，参考python的join（）方法

test_feature_all = []
test_label_all = []

correct = 0   # 定义预测正确的图片数，初始化为0
total = 0     # 总共参与测试的图片数，也初始化为0
for data in testloader:  # 循环每一个batch
    images, test_label = data
    test_feature,outputs = net(Variable(images))  # 输入网络进行测试
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)          # 更新测试图片的数量
    correct += (predicted == test_label).sum() # 更新正确分类的图片的数量
    test_feature = test_feature.detach().numpy()
    test_label = test_label.detach().numpy()
    test_feature_all.append(test_feature)
    test_label_all.append(test_label)
 
print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))          # 最后打印结果

class_correct = list(0. for i in range(10)) # 定义一个存储每类中测试正确的个数的 列表，初始化为0
class_total = list(0. for i in range(10))   # 定义一个存储每类中测试总数的个数的 列表，初始化为0
for data in testloader:     # 以一个batch为单位进行循环
    images, labels = data
    test_feature,outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).squeeze()
    for i in range(4):      # 因为每个batch都有4张图片，所以还需要一个4的小循环
        label = labels[i]   # 对各个类的进行各自累加
        class_correct[label] += c[i]
        class_total[label] += 1
     
     
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
    
    
test_feature_part = merge(test_feature_all[:100])
test_label_part = merge(test_label_all[:100])



 
save3="./data/result/cifar/test_feature_epoch_{}_(loss{}).mat".format(epoch + 1, format(loss.data,'.4f'))
save4="./data/result/cifar/test_label_epoch_{}.mat".format(epoch + 1)
scio.savemat(save3, {'feature':test_feature_part})
scio.savemat(save4, {'label':test_label_part})

