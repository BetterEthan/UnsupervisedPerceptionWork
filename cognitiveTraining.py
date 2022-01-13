from typing import overload
import numpy as np
from scipy.sparse import dok_matrix
from sklearn.base import BaseEstimator, ClusterMixin
from random import choice
import matplotlib.pyplot as plt
from esoinn import ESoinn
import pickle
from matplotlib import colors


class LearnTerrain():
    def __init__(self):
        self.visualModel = ESoinn()
        self.tactileModel = ESoinn()
        self.MappingNode = np.empty(shape=[0, 3])
        # self.MappingNode = np.append(self.MappingNode, data, axis=0)

    def  setModelFromFile(self):
        with open("personalModels/tempData/tactileModle.pkl",'rb') as file:
            self.tactileModel  = pickle.loads(file.read())

        with open("personalModels/tempData/visualModle.pkl",'rb') as file:
            self.visualModel  = pickle.loads(file.read())

    def plotTactileCognitiveMap(self):
        color = ['black', 'red', 'saddlebrown', 'skyblue', 'magenta', 'green', 'gold', 'peru', 'palegreen', 'yellowgreen', 'maroon', 'ivory', 'darkkhaki', 'burlywood', 'aliceblue']
        color_dict = {}
        for k in self.tactileModel.adjacent_mat.keys():
            plt.plot(self.tactileModel.nodes[k, 0], self.tactileModel.nodes[k, 1], 'k', c='blue')
        for i in range(len(self.tactileModel.nodes)):
            if not self.tactileModel.node_labels[i] in color_dict:
                color_dict[self.tactileModel.node_labels[i]] = choice(color)
            # plt.plot(self.tactileModel.nodes[i][0], self.tactileModel.nodes[i][1], 'ro', c=color_dict[self.tactileModel.node_labels[i]])
            plt.text(self.tactileModel.nodes[i][0], self.tactileModel.nodes[i][1],self.tactileModel.node_labels[i],ha='center',va='bottom',fontsize=12)  
        return

    def plotVisualCognitiveMap(self):
        color = ['black', 'red', 'saddlebrown', 'skyblue', 'magenta', 'green', 'gold', 'peru', 'palegreen', 'yellowgreen', 'maroon', 'ivory', 'darkkhaki', 'burlywood', 'aliceblue']
        color_dict = {}
        for k in self.visualModel.adjacent_mat.keys():
            plt.plot(self.visualModel.nodes[k, 0], self.visualModel.nodes[k, 1], 'k', c='blue')
        for i in range(len(self.visualModel.nodes)):
            if not self.visualModel.node_labels[i] in color_dict:
                color_dict[self.visualModel.node_labels[i]] = color[self.visualModel.node_labels[i]]
            plt.plot(self.visualModel.nodes[i][0], self.visualModel.nodes[i][1], 'ro', c=color_dict[self.visualModel.node_labels[i]])
            plt.text(self.visualModel.nodes[i][0], self.visualModel.nodes[i][1],self.visualModel.node_labels[i],ha='center',va='bottom',fontsize=12)  
        return

    # 重新训练+训练后得更新认知映射关系
    def trainTactileModel(self,newData, trainNum):
        if self.tactileModel.nodes.shape[0] < 1:
            self.tactileModel.fit(newData,trainNum)
        else:
            self.tactileModel.fitAgain(newData,trainNum)

        # print(self.tactileModel.nodes.shape)
        # input('.......')
        return self

    def getMappingPairClass(self,mappingRawData):
        winnerV, lableV = self.visualModel.findNearestWinner(mappingRawData[0:2])
        winnerT, lableT = self.tactileModel.findNearestWinner(mappingRawData[2:4])
        return [lableV,lableT], winnerV, winnerT

    def addMappingNode(self, data):
        
        if(len(self.MappingNode) < 1):
            self.MappingNode = np.append(self.MappingNode, np.array(np.hstack((data,[1]))).reshape(1,3), axis=0)
            return self
        
        for nn in self.MappingNode:
            # print(data.type)
            # print(np.array(data)[0:2])
            # input()
            flag_ = (nn[0:2] == np.array(data)[0:2])
            if flag_.all():
                nn[2] = nn[2] + 1
                return self
        
        self.MappingNode = np.append(self.MappingNode, np.array(np.hstack((data,[1]))).reshape(1,3), axis=0)

        
        return self


    # 建立视觉类别与触觉类别联系
    def setAssociateLayer(self):
        data = np.load('XByOrder2.npy')
        label = np.load('labelDataNew32.npy')

        for i in range(0,data.shape[0]):
            winnerV, lableV  = self.visualModel.findNearestWinner(data[i,:])
            winnerV2, lableV2  = self.visualModel.findNearestCenter(data[i,:])
            # print(lableV,lableV2)
            # input()

            self.addMappingNode([lableV,label[i]])

    # 获取训练模型的原始数据
    def getTrainData(self):
        
        tSNEData = np.load('XByOrder2.npy')
        # latentData = np.load('latentDataNew32.npy')
        labelBySOINN = []
        for i in range(0,tSNEData.shape[0]):
            winnerV, lableV  = self.visualModel.findNearestWinner(tSNEData[i,:])
            labelBySOINN.append(lableV)
            print(i,lableV)
        np.save('labelBySOINN',labelBySOINN)
        print('ok',len(labelBySOINN))

    def predictTactileInfo(self, visualType):
        mappingData = []
        mappingDataNum = []
        for nn in self.MappingNode: 
            if nn[0] == visualType:
                mappingDataNum.append(nn[2])
                mappingData.append(nn)

        if(len(mappingDataNum) == 0):
            return -1
        maxIndex = mappingDataNum.index(max(mappingDataNum))
        # print(mappingData)
        return mappingData[maxIndex][1]


# cognitiveModel = LearnTerrain()

# cognitiveModel.setModelFromFile()

# cognitiveModel.getTrainData()

# 读取类别数目
# print(len(cognitiveModel.visualModel.nodesByclass))

# print(cognitiveModel.MappingNode)

##############################################
# cognitiveModel = LearnTerrain()

# cognitiveModel.setModelFromFile()
# # cognitiveModel.setAssociateLayer()
# print(len(cognitiveModel.visualModel.nodesByclass))
# input(3333333333333)

# aas = cognitiveModel.predictTactileInfo(8)

# # print(len(cognitiveModel.MappingNode),'final')
# # print(cognitiveModel.MappingNode)
# input(111111111)

#######################绘制认知图#######################

# def getColorArray():
#     imageResize = []
#     imageResize.append(colors.to_rgba('grey'))
#     imageResize.append(colors.to_rgba('brown'))
#     imageResize.append(colors.to_rgba('orange'))
#     imageResize.append(colors.to_rgba('olive'))
#     imageResize.append(colors.to_rgba('green'))
#     imageResize.append(colors.to_rgba('cyan'))
#     imageResize.append(colors.to_rgba('blue'))
#     imageResize.append(colors.to_rgba('purple'))
#     imageResize.append(colors.to_rgba('pink'))
#     imageResize.append(colors.to_rgba('red'))
#     imageResize.append(colors.to_rgba('cornflowerblue'))
#     imageResize.append(colors.to_rgba('tomato'))
#     imageResize.append(colors.to_rgba('tan'))
#     imageResize.append(colors.to_rgba('yellow'))

#     mmmm=np.array(imageResize)
#     aaa = mmmm[:,0:3]
#     return aaa

# cognitiveModel = LearnTerrain()

# cognitiveModel.setModelFromFile()

# cognitiveModel.setAssociateLayer()


# labelData = np.load('labelDataNew32.npy')
# data = np.load('XByOrder2.npy')

# from matplotlib import cm
# X_tsne = data


# index_ = 0
# count_ = 0
# # for label_ in labelData:
# for ddd_ in data:
#     colorList = getColorArray()
#     # cccc = colorList[label_+1]
#     winnerV, lableV  = cognitiveModel.visualModel.findNearestWinner(ddd_)
#     # print(lableV)
#     # input()
#     cccc = np.array(colorList[int(0)]).reshape(1,-1)
#     #1
#     # if(lableV == 0):
#     #     cccc = np.array(colorList[int(1)]).reshape(1,-1)
#     # if(lableV == 8):
#     #     cccc = np.array(colorList[int(1)]).reshape(1,-1)
    
#     # #2
#     # if(lableV == 4):
#     #     cccc = np.array(colorList[int(2)]).reshape(1,-1)

#     # #3
#     # if(lableV == 2):
#     #     cccc = np.array(colorList[int(1)]).reshape(1,-1)
#     # if(lableV == 10):
#     #     cccc = np.array(colorList[int(1)]).reshape(1,-1)

#     # #4
#     # if(lableV == 7):
#     #     cccc = np.array(colorList[int(2)]).reshape(1,-1)

#     # #5
#     # if(lableV == 3):
#     #     cccc = np.array(colorList[int(3)]).reshape(1,-1)

#     # #6
#     # if(lableV == 6):
#     #     cccc = np.array(colorList[int(3)]).reshape(1,-1)

#     # #7
#     # if(lableV == 1):
#     #     cccc = np.array(colorList[int(1)]).reshape(1,-1)
#     # if(lableV == 5):
#     #     cccc = np.array(colorList[int(1)]).reshape(1,-1)

#     plt.scatter(X_tsne[index_,0], X_tsne[index_,1], s=8, c=cccc)
#     index_ = index_ + 1
#     count_ = count_ + 1
#     print(count_)
#     if(count_ > 3000):
#         break

# # cognitiveModel.plotVisualCognitiveMap()
# plt.axis('off')
# # plt.savefig('./videoFinal2/999.png',bbox_inches='tight',pad_inches=0.0,transparent = True)
# plt.grid(True)

# plt.show()
# print('ok.................')







######################### 构建mapping data  #####################################################



'''
data = np.load('XByOrder.npy')

mean = (1, 1)
cov = np.array([[1.1, 0.0], [0.0, 1.3]])
X = data[0:200,:]
Y = np.random.multivariate_normal(mean, cov, (200,), 'raise')   # nx2
mapData= np.hstack((X,Y))


mean = (14, 2)
cov = np.array([[1.6, 0.0], [0.0, 1]])
X = data[600:800,:]
Y = np.random.multivariate_normal(mean, cov, (200,), 'raise')   # nx2
mapData= np.vstack((mapData,np.hstack((X,Y)))) 


mean = (5, 7)
cov = np.array([[1.1, 0.0], [0.0, 2.1]])
X = data[1100:1300,:]
Y = np.random.multivariate_normal(mean, cov, (200,), 'raise')   # nx2
mapData= np.vstack((mapData,np.hstack((X,Y)))) 


mean = (7, 14)
cov = np.array([[1.1, 0.0], [0.0, 0.4]])
X = data[1550:1750,:]
Y = np.random.multivariate_normal(mean, cov, (200,), 'raise')   # nx2
mapData= np.vstack((mapData,np.hstack((X,Y)))) 


mean = (23, 12)
cov = np.array([[2.1, 0.0], [0.0, 1]])
X = data[2050:2250,:]
Y = np.random.multivariate_normal(mean, cov, (200,), 'raise')   # nx2
mapData= np.vstack((mapData,np.hstack((X,Y)))) 



mean = (13, 12)
cov = np.array([[1.1, 0.0], [0.0, 1]])
X = data[3100:3300,:]
Y = np.random.multivariate_normal(mean, cov, (200,), 'raise')   # nx2
mapData= np.vstack((mapData,np.hstack((X,Y)))) 


# np.save('mapData2',mapData)
# # print(mapData[0,:])
# input(22222222)

'''


##########################################################################################################

# mapData = np.load('mapData.npy')

'''
cognitiveModel = LearnTerrain()

cognitiveModel.setModelFromFile()

for i in range(0,mapData.shape[0]):
    aa, winnerV, winnerT = cognitiveModel.getMappingPairClass(mapData[i,:])
    print(i)
    cognitiveModel.addMappingNode(aa)


print(len(cognitiveModel.MappingNode),'final')
print(cognitiveModel.MappingNode)








plt.figure()
plt.plot(mapData[0,0], mapData[0,1], 'ro', markersize=14.)
plt.plot(winnerV[0], winnerV[1], 'ro', markersize=14.)
for i in range(0,mapData.shape[0]):
    if i<200:
        plt.plot(mapData[i,0], mapData[i,1],'ro')
    elif i<400:
        plt.plot(mapData[i,0], mapData[i,1],'go')
    elif i<600:
        plt.plot(mapData[i,0], mapData[i,1],'bo')
    elif i<800:
        plt.plot(mapData[i,0], mapData[i,1],'co')
    elif i<1000:
        plt.plot(mapData[i,0], mapData[i,1],'ko')
    elif i<1200:
        plt.plot(mapData[i,0], mapData[i,1],'mo')

cognitiveModel.plotVisualCognitiveMap()
plt.grid(True)


plt.figure()
plt.plot(mapData[0,2], mapData[0,3], 'ro', markersize=14.)
plt.plot(winnerT[0], winnerT[1], 'ro', markersize=14.)
for i in range(0,mapData.shape[0]):
    if i<200:
        plt.plot(mapData[i,2], mapData[i,3],'ro')
    elif i<400:
        plt.plot(mapData[i,2], mapData[i,3],'go')
    elif i<600:
        plt.plot(mapData[i,2], mapData[i,3],'bo')
    elif i<800:
        plt.plot(mapData[i,2], mapData[i,3],'co')
    elif i<1000:
        plt.plot(mapData[i,2], mapData[i,3],'ko')
    elif i<1200:
        plt.plot(mapData[i,2], mapData[i,3],'mo')

cognitiveModel.plotTactileCognitiveMap()
plt.grid(True)


plt.grid(True)
plt.show()
'''



'''  增量学习
data = np.load('XByOrder.npy')


plt.figure()
plt.plot(data[:, 0], data[:, 1], 'cx')
X = data[1:470,:]
cognitiveModel.trainTactileModel(X,1000)
cognitiveModel.plotTactileCognitiveMap()
plt.grid(True)
print('11111.......')
plt.savefig('1.png')
# plt.show()

plt.figure()
plt.plot(data[:, 0], data[:, 1], 'cx')
X = data[470:470*2,:]
cognitiveModel.trainTactileModel(X,1000)
cognitiveModel.plotTactileCognitiveMap()
plt.grid(True)
print('2222222.......')
plt.savefig('2.png')
# plt.show()


plt.figure()
plt.plot(data[:, 0], data[:, 1], 'cx')
X = data[470*2:470*3,:]
cognitiveModel.trainTactileModel(X,1000)
cognitiveModel.plotTactileCognitiveMap()
plt.grid(True)
print('3333333.......')
plt.savefig('3.png')
# plt.show()



plt.figure()
plt.plot(data[:, 0], data[:, 1], 'cx')
X = data[470*3:470*4,:]
cognitiveModel.trainTactileModel(X,1000)
cognitiveModel.plotTactileCognitiveMap()
plt.grid(True)
print('444444444.......')
plt.savefig('4.png')
# plt.show()


plt.figure()
plt.plot(data[:, 0], data[:, 1], 'cx')
X = data[470*4:470*5,:]
cognitiveModel.trainTactileModel(X,1000)
cognitiveModel.plotTactileCognitiveMap()
plt.grid(True)
print('5555555.......')
plt.savefig('5.png')
# plt.show()


plt.figure()
plt.plot(data[:, 0], data[:, 1], 'cx')
X = data[470*5:470*6,:]
cognitiveModel.trainTactileModel(X,1000)
cognitiveModel.plotTactileCognitiveMap()
plt.grid(True)
print('666666666.......')
plt.savefig('6.png')
# plt.show()



plt.figure()
plt.plot(data[:, 0], data[:, 1], 'cx')
X = data[470*6:470*7,:]
cognitiveModel.trainTactileModel(X,1000)
cognitiveModel.plotTactileCognitiveMap()
plt.grid(True)
print('777777777777.......')
plt.savefig('7.png')
# plt.show()
'''
