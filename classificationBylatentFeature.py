import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
from matplotlib import cm

def feature_normalize(data):
    mu = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    return (data - mu)/std



latentDataWhole = feature_normalize(np.load('latentDataC.npy'))
labelDataWhole =  np.load('labelDataC.npy')
labelDataWhole = labelDataWhole.reshape(latentDataWhole.shape[0],1)



# T SNE 高维数据可视化
from sklearn.manifold import TSNE
from matplotlib import cm
import matplotlib.pyplot as plt
X_tsne = TSNE(n_components=2, random_state=0).fit_transform(latentDataWhole)


# kmeans 手拐法
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
K = range(1, 15)
meandistortions = []
for k in K:
    print(k)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X_tsne)
    meandistortions.append(sum(np.min(cdist(X_tsne, kmeans.cluster_centers_, 'euclidean'), axis=1))/X_tsne.shape[0])
plt.plot(K, meandistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average Dispersion')
plt.title('Selecting k with the Elbow Method')
plt.show()
input("Press Enter to continue...")

radio = 0.06
trainNum = latentDataWhole.shape[0]*radio
print(trainNum)

latentDataTrain = torch.from_numpy(latentDataWhole[1:int(trainNum),:]).cuda().to(torch.float32)
latentDataTest = torch.from_numpy(latentDataWhole[int(trainNum):latentDataWhole.shape[0],:]).cuda().to(torch.float32)
labelDataTrain = torch.from_numpy(labelDataWhole[1:int(trainNum),:]).cuda().to(torch.float32)
labelDataTest = torch.from_numpy(labelDataWhole[int(trainNum):labelDataWhole.shape[0],:]).cuda().to(torch.float32)


xx , yy =(Variable(latentDataTrain),Variable(labelDataTrain))
testX, testY = (Variable(latentDataTest),Variable(labelDataTest))


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.hidden1 = nn.Linear(512,1024)
        self.hidden2 = nn.Linear(1024,1024)
        self.hidden3 = nn.Linear(1024,512)
        self.hidden4 = nn.Linear(512,128)
        self.predict = nn.Linear(128,1)
    def forward(self,input):
        out = self.hidden1(input)
        out = F.sigmoid(out)
        out = self.hidden2(out)
        out = F.sigmoid(out)
        out = self.hidden3(out)
        out = F.sigmoid(out)
        out = self.hidden4(out)
        out = F.sigmoid(out)
        out =self.predict(out)

        return out


# '''
net = Net().cuda()
print(net)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
# optimizer = torch.optim.SGD(net.parameters(),lr = 0.001,momentum = 0.8)
loss_func = torch.nn.MSELoss()

fig, ax = plt.subplots()
x = []
y = []
y_Test = []
res = 0
for t in range(500):
    # for j in range(xx.shape[0]):
    prediction = net(xx)
    loss = loss_func(prediction,yy)

    # print(prediction,yy.data[j],loss)
    # input("Press Enter to continue...")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    




    if t%5 ==0:

        ## 可视化loss
        x.append(t)
        y.append(loss.data.cpu().detach().numpy())
        ax.cla() # clear plot
        ax.plot(x, y, 'r', lw=1) # draw line chart
        # ax.bar(y, height=y, width=0.3) # draw bar chart

        prediction = net(testX)
        lossTest = loss_func(prediction,testY)
        y_Test.append(lossTest.data.cpu().detach().numpy())
        ax.plot(x, y_Test, 'b', lw=1) # draw line chart

        print(loss.data.cpu().detach().numpy(),lossTest.data.cpu().detach().numpy())

        plt.pause(0.1)
        # input("Press Enter to continue...")

    #     plt.cla()
    #     plt.scatter(x.data.numpy(), y.data.numpy())
    #     plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    #     plt.text(0.5, 0, 'Loss = %.4f' % loss.data, fontdict={'size': 20, 'color': 'red'})
    #     plt.pause(0.05)

torch.save(net, './predict.pth')
input("Press Enter to continue...")



# net = Net().cuda()
# net = torch.load('predict.pth')
# for j in range(latentDataTest.shape[0]):
#     predict_ = net(latentDataTest.data[j])
#     print(predict_.cpu().detach().numpy(),labelDataTest.data[j].cpu().detach().numpy())

#     input("Press Enter to continue...")









