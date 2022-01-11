# 训练获得通过银行特征预测SOINN的类别模型


#coding:utf-8
import pandas as pd
import numpy as np                    #导入扩展库numpy（数组、函数等）
import matplotlib.pyplot as plt       #导入扩展库matplotlib(数据可视化、作图工具等) 
import pylab
from scipy.optimize import minimize
import random
from math import e
import scipy.signal as signal

from sklearn import svm
import matplotlib
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import  cross_val_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix
import itertools
import seaborn as sns
from torch.utils import data
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import torch.nn.functional as F




def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data, target
            output = model(data)
            pred = output.argmax(dim=1)
            # print(pred)
            # print(target)
            # input()
            correct += torch.eq(pred, target).float().sum().item()
            test_loss += criterion(output, target.long())

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)







class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(512,256)
        self.hidden2 = nn.Linear(256,64)
        self.predict = nn.Linear(64,10)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.predict(x)
        return x






# num_epochs = 20
# batch_size =  36
# learning_rate = 1e-3

# latentData =  np.load('latentDataNew32.npy')
# labelData =  np.load('labelBySOINN.npy')
# # labelData = labelData.reshape(labelData.shape[0],1)
# # dataSum = dataSum.reshape(720,4,1,100)
# tensor_x = torch.Tensor(latentData) # transform to torch tensor
# tensor_y = torch.Tensor(labelData)
# # print(tensor_y.size())
# # input(100000000000000000)

# my_dataset = data.TensorDataset(tensor_x,tensor_y) # create your datset
# my_dataloader = data.DataLoader(dataset=my_dataset, batch_size=batch_size, shuffle=True) # create your dataloader






# model = Net()
# # model = torch.load('abbbbbb70.pth')
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
#                              weight_decay=1e-5)
# # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.95)
# # total_loss = 0








# fig, ax = plt.subplots()
# x = []
# y = []
# y_Test = []
# for epoch in range(num_epochs):
#     # print(epoch)
#     for batch_idx, (data, target) in enumerate(my_dataloader):
#         # print(batch_idx)
#         # input()
#         # if batch_idx > 3+2+2+2+2+2:
#         #     break
#         data = Variable(data)

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = model(data)
#         # print(target.size())
#         # input(333)

#         loss = criterion(outputs, target.long())
#         # print('ssssssssssssssssss')
#         loss.backward()
#         optimizer.step()
#     # # 测试集测试精度
#     x.append(epoch)
#     y.append(test(model, my_dataloader))
#     # y_Test.append(test(model, my_dataloader_test))
#     ax.cla() # clear plot
#     ax.plot(x, y, 'r', lw=1) # draw line chart
#     # ax.plot(x, y_Test, 'b', lw=1) # draw line chart
#     plt.pause(0.1)

# torch.save(model, '1trainPredictClass.pth')
# input('finished!!!')
