#用于测试esoinn网络的使用


import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal
from numpy.random import uniform
from soinn import Soinn
from esoinn import ESoinn
from random import choice




# latentDataWhole = np.load('latentDataNew32.npy')
# labelData = np.load('labelDataNew32.npy')

# print(latentDataWhole.shape)
# # input('input ......')
# # T SNE 高维数据可视化
# from sklearn.manifold import TSNE
# from matplotlib import cm
# import matplotlib.pyplot as plt
# X_tsne = TSNE(n_components=2, random_state=3).fit_transform(latentDataWhole)
# X = X_tsne

# color = ['black', 'red', 'saddlebrown', 'skyblue', 'magenta', 'green', 'gold', 'peru', 'palegreen', 'yellowgreen', 'maroon', 'ivory', 'darkkhaki', 'burlywood', 'aliceblue']


# from matplotlib import cm
# fig = plt.figure(2)
# index_ = 0
# for label_ in labelData:
#     print(label_)
#     cccc = color[label_]    # 上色
#     plt.scatter(X_tsne[index_,0], X_tsne[index_,1], s=8, c=cccc)
#     index_ = index_ + 1
# plt.show()

# input('ke shi hua .........')
# np.save('XByOrder2',X_tsne)
# input(3234324311111111)







# label = np.load('labelDataNew32.npy')
# data = np.load('XByOrder2.npy')
# import torch as th
# import torch
# for i in range(len(data)):
#     # similarity = th.nn.CosineSimilarity(dim=-1)
#     # x = torch.from_numpy(data[5])
#     # y = torch.from_numpy(data[i])
#     # # print(x.size())
#     # # print(y.size())

#     # dis = similarity(x, y)

#     # dis = np.dot(np.array(data[5]),np.array(data[i])) / (np.linalg.norm(np.array(data[0])) * np.linalg.norm(np.array(data[i])) )
    
    
#     # SSSS = th.nn.CosineSimilarity(dim=-1)
    
#     # x = torch.from_numpy(data[5])
#     # y = torch.from_numpy(data[i])

#     # x = x.unsqueeze(1)
#     # # Reshape y: (2N, C) -> (1, C, 2N)
#     # y = y.unsqueeze(0)

#     # # x = x.unsqueeze(1)
#     # # # Reshape y: (2N, C) -> (1, C, 2N)
#     # # y = y.T.unsqueeze(0)
#     # similarity = SSSS(x, y)


#     # print(np.sum((np.array(data[0]))**2))
#     # print(np.linalg.norm(np.array(data[0]))**2)
#     # print(data.shape)
#     dis = np.sum((np.array(data[0]) - np.array(data[i]))**2)
#     print(i,label[i], '\t dis:\t', dis)
#     # input(23143124235345)


# input(23143124235345)



# label = np.load('labelDataNew32.npy')
data = np.load('XByOrder2.npy')



# labelData = np.load('labelDataNew32.npy')
# from matplotlib import cm
# X_tsne = data
# fig = plt.figure(2)
# index_ = 0
# for label_ in labelData:
#     print(label_)
#     cccc = cm.rainbow(int(255*label_/9))    # 上色
#     plt.scatter(X_tsne[index_,0], X_tsne[index_,1], s=8, c=cccc)
#     index_ = index_ + 1
# plt.show()
# input('ke shi hua .........')


X = data

# initialize SOINN or ESoinn
# s = Soinn()
s = ESoinn()
s.fit(X,10000)  #训练


# aa = [-20,40]
# winner, lable = s.findNearestCenter(aa)


nodes = s.nodes

print("end")
print(len(s.nodes))
print(len(s.node_labels))




# '''
# show SOINN's state

for k in s.adjacent_mat.keys():
    plt.plot(nodes[k, 0], nodes[k, 1], 'k', c='blue')
# plt.plot(nodes[:, 0], nodes[:, 1], 'ro')

color = ['black', 'red', 'saddlebrown', 'skyblue', 'magenta', 'green', 'gold', 'peru', 'palegreen', 'yellowgreen', 'maroon', 'ivory', 'darkkhaki', 'burlywood', 'aliceblue']

# label density
# for i in range(len(s.nodes)):
#     str_tmp = str(s.density[i]) + " " + str(s.N[i])
#     plt.text(s.nodes[i][0], s.nodes[i][1], str_tmp, family='serif', style='italic', ha='right', wrap=True)

color_dict = {}

labelData = np.load('labelDataNew32.npy')
from matplotlib import cm
X_tsne = data
index_ = 0
for label_ in labelData:
    print(label_)
    cccc = cm.rainbow(int(255*label_/9))    # 上色
    plt.scatter(X_tsne[index_,0], X_tsne[index_,1], s=8, c=cccc)
    index_ = index_ + 1


for i in range(len(s.nodes)):
    if not s.node_labels[i] in color_dict:
        color_dict[s.node_labels[i]] = choice(color)
    plt.plot(s.nodes[i][0], s.nodes[i][1], 'ro', c=color_dict[s.node_labels[i]])
    plt.text(s.nodes[i][0], s.nodes[i][1],s.node_labels[i],ha='center',va='bottom',fontsize=12)  




# 对新数据进行类别划分
# aa = [-20,40]
# winner, lable = s.findNearestWinner(aa)
# plt.plot(aa[0], aa[1], 'ro', markersize=14.)
# plt.plot(winner[0], winner[1], 'ro', markersize=14.)
# plt.text(aa[0], aa[1],lable,ha='center',va='bottom',fontsize=12)





plt.grid(True)
plt.show()


#######模型保存##########
input('save or no .......')
import pickle
output_hal = open("visualModle.pkl", 'wb')
str = pickle.dumps(s)
output_hal.write(str)
output_hal.close()

# '''


# input('fitAgain............')




# X = data[2400:3300,:]

# s.fitAgain(X,2000)  #训练

# nodes = s.nodes

# # show SOINN's state
# plt.plot(data[:, 0], data[:, 1], 'cx')

# for k in s.adjacent_mat.keys():
#     plt.plot(nodes[k, 0], nodes[k, 1], 'k', c='blue')
# # plt.plot(nodes[:, 0], nodes[:, 1], 'ro')

# color = ['black', 'red', 'saddlebrown', 'skyblue', 'magenta', 'green', 'gold', 'peru', 'palegreen', 'yellowgreen', 'maroon', 'ivory', 'darkkhaki', 'burlywood', 'aliceblue']

# # label density
# # for i in range(len(s.nodes)):
# #     str_tmp = str(s.density[i]) + " " + str(s.N[i])
# #     plt.text(s.nodes[i][0], s.nodes[i][1], str_tmp, family='serif', style='italic', ha='right', wrap=True)

# color_dict = {}

# print(len(s.nodes))
# print(len(s.node_labels))

# for i in range(len(s.nodes)):
#     if not s.node_labels[i] in color_dict:
#         color_dict[s.node_labels[i]] = choice(color)
#     plt.plot(s.nodes[i][0], s.nodes[i][1], 'ro', c=color_dict[s.node_labels[i]])
#     plt.text(s.nodes[i][0], s.nodes[i][1],s.node_labels[i],ha='center',va='bottom',fontsize=12)  

# plt.grid(True)
# plt.show(block=False)


