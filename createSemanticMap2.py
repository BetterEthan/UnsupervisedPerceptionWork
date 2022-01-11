"""
# 输入指定图片，尝试采用SimClr+SOINN进行语义分割
"""





# coding:utf-8 
import pylab 
import imageio 
#注释的代码执行一次就好，以后都会默认下载完成 
#imageio.plugins.ffmpeg.download() 
import skimage 
import numpy as np 
 
import cv2 


import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size
from src.model import ContrastiveEncoder
from utils.arguments import get_arguments, get_config
import torch
import numpy as np
from esoinn import ESoinn
from cognitiveTraining import LearnTerrain
from trainPredictClass import Net
from torch.autograd import Variable
import time
from matplotlib import colors

# Get parser / command line arguments
args = get_arguments()
# Get configuration file
config = get_config(args)


# print("img5:",img5.shape)
# print("img5:",type(img5))


model = ContrastiveEncoder(config)
# Load contrastive encoder
model.load_models()
# Move the model to the device
model.contrastive_encoder.to(config["device"])

trainPredictClassModel = Net()
trainPredictClassModel = torch.load('1trainPredictClass.pth')



# latentData =  np.load('latentDataNew32.npy')
# labelData =  np.load('labelBySOINN.npy')

# i = 0
# for ttt in latentData:
#     data = Variable(torch.Tensor(ttt.reshape(1,ttt.shape[0]))).cpu()
#     output  = trainPredictClassModel(data)
#     lableV = output.argmax(dim=1)

#     print(lableV,labelData[i])
#     i = i+1
# input('xxxxxxxxxxx')     


cognitiveModel = LearnTerrain()

cognitiveModel.setModelFromFile()

# cognitiveModel.setAssociateLayer()



def getColorArray():
    imageResize = []
    imageResize.append(colors.to_rgba('grey'))
    imageResize.append(colors.to_rgba('brown'))
    imageResize.append(colors.to_rgba('orange'))
    imageResize.append(colors.to_rgba('olive'))
    imageResize.append(colors.to_rgba('green'))
    imageResize.append(colors.to_rgba('cyan'))
    imageResize.append(colors.to_rgba('blue'))
    imageResize.append(colors.to_rgba('purple'))
    imageResize.append(colors.to_rgba('pink'))
    imageResize.append(colors.to_rgba('red'))
    imageResize.append(colors.to_rgba('cornflowerblue'))
    imageResize.append(colors.to_rgba('tomato'))
    imageResize.append(colors.to_rgba('tan'))
    imageResize.append(colors.to_rgba('yellow'))

    mmmm=np.array(imageResize)* 255
    aaa = mmmm[:,0:3]
    return aaa
    


# img5 a x b x channels
def getSemanticPicture(img5, num): 
    # plt.ion()
    plt.figure(1, figsize=(10, 3))
    plt.subplot(121)
    plt.axis('off')

    plt.imshow(img5,cmap='Greys')
    img5 = np.transpose(img5, (2,0,1))
    img5 = img5.reshape(img5.shape[0],img5.shape[1],img5.shape[2])

    S = 96
    stride = 96
    N1 = int(img5.shape[1]/S)
    N2 = int(img5.shape[2]/S)
  
    semanticMap = np.mat(np.zeros([N1,N2]))
    blank_image = np.zeros((N1,N2,3), np.uint8)

    imageResize = []

    

    colorlist_ = getColorArray()
    for i in range(N1):
        imageResize = []
        for j in range(N2):
            image_ = img5[:,i*stride:i*stride+S,j*stride:j*stride+S]
            imageResize.append(image_)




        IImage = torch.tensor(imageResize)
        Xbatch = IImage.to(model.device).float()
        _, h  = model.contrastive_encoder(Xbatch)  # 维数是512， 中间层输出 
        data = Variable(h).cpu()
        output  = trainPredictClassModel(data)

        lableV = output.argmax(dim=1)

        print(lableV)
        input("..................")
        # print(IImage.size())

        patches = imageResize[0]
        pp = np.transpose(patches, (1,2,0))
        cv2.imshow('xxx',pp)

        cv2.waitKey(1000)
        print(pp[:,:,1])
        input("xxxxxxxxxxxxx")

        # #添加第一种地形：沥青 0
        if(num > 0):
            cognitiveModel.addMappingNode([0,0]) 
        if(num > 250):
            cognitiveModel.addMappingNode([8,0])
        # 人行道
        if(num > 430):
            cognitiveModel.addMappingNode([2,0])
        if(num > 450):
            cognitiveModel.addMappingNode([10,0])
        # 大块人行道
        if(num > 7080):
            cognitiveModel.addMappingNode([1,0])


        # 路沿
        if(num > 370):
            cognitiveModel.addMappingNode([4,1])
        if(num > 870):
            cognitiveModel.addMappingNode([7,1])



        # 草
        if(num > 950):
            cognitiveModel.addMappingNode([3,3])
        if(num > 1570):
            cognitiveModel.addMappingNode([6,3])



        for jj in range(len(lableV)):
            # 
            # lll_ = cognitiveModel.predictTactileInfo(lableV[jj])
            lll_ = lableV[jj]
            if(lll_ == -1): #该地形类别不认识
                blank_image[i,jj,:] = (255,255,255)
            else:
                # print(lll_)
                # input('.........')
                blank_image[i,jj,:] = colorlist_[int(lll_)]

            # print(lll_)
            # input('.........')

        # for jj in range(len(lableV)):
        #     blank_image[i,jj,:] = colorlist_[lableV[jj]]


    image_ = semanticMap.reshape(N1,N2,1)

    time_start=time.time()


    # plt.figure(1, figsize=(10, 4))
    # plt.subplot(122)
    # plt.axis('off')
    # image = plt.imshow(blank_image)
    # plt.show()


    # plt.savefig('./xxxxxxxxtt/'+str(num)+'picture.png',bbox_inches='tight',dpi=500,pad_inches=0.0)
    

    # plt.savefig('./soinnLable/'+str(num)+'picture.png',bbox_inches='tight',pad_inches=0.0,transparent = True)
    # plt.close()
    # del(image)
    # input('.........')

    time_end=time.time()
    print('totally cost',time_end-time_start)


# img5 = plt.imread('xx.png')

# getSemanticPicture(img5)    
# input('...xxx......')

#视频的绝对路径 
filename = './SLIC/fast.mp4' 
#可以选择解码工具 
vid = imageio.get_reader(filename, 'ffmpeg') 
for num,im in enumerate(vid): 
    #image的类型是mageio.core.util.Image可用下面这一注释行转换为arrary 
    if(num % 1 == 0 and num > 0):
        print(num)

        # print(im.shape)
        # input("xx..........xx")
        image = skimage.img_as_float(im).astype(np.float64) 
        print(type(image))
        # fig = pylab.figure() 
        # fig.suptitle('image #{}'.format(num), fontsize=20) 
        # pylab.imshow(im) 
        # pylab.show() 
        getSemanticPicture(image, num)    

input('..............')



'''
time_start=time.time()
for i in range(N1):
    for j in range(N2):
        image_ = img5[:,:,i*stride:i*stride+S,j*stride:j*stride+S]
        imageResize.append(image_[0,:,:,:])

IImage = torch.tensor(imageResize)
Xbatch = IImage.to(model.device).float()


_, h1  = model.contrastive_encoder(Xbatch[0:1500])  # 维数是512， 中间层输出 
_, h2  = model.contrastive_encoder(Xbatch[1500:1564])  # 维数是512， 中间层输出 
h = torch.cat([h1, h2], 0)
data = Variable(h).cpu()
output  = trainPredictClassModel(data)
lableV = output.argmax(dim=1)
aaa = np.mat(np.zeros([1,N1*N2]))
for jj in range(len(lableV)):
    aaa[0,jj] = cognitiveModel.predictTactileInfo(lableV[jj])
    # aaa[jj] = cognitiveModel.predictTactileInfo(lableV[jj])

semanticMap = aaa.reshape(N1,N2)
print(aaa.shape)
time_end=time.time()

print('totally cost',time_end-time_start)
image_ = semanticMap.reshape(N1,N2,1)
plt.imshow(image_,cmap='Greys')
plt.show()
'''




# # Create list to hold encodings and labels
# h_list, y_list = [], []
# # Turn on training mode for each model.
# model.set_mode(mode="evaluation")
# # Compute total number of batches per epoch
# self.total_batches = len(train_loader)
# print(f"Total number of samples / batches in data set: {len(train_loader.dataset)} / {len(train_loader)}")
#     # Attach progress bar to data_loader to check it during training. "leave=True" gives a new line per epoch
# self.tqdm = tqdm(enumerate(train_loader), total=self.total_batches, leave=True)
# # Go through batches
# for i, ((Xbatch, _), Ybatch) in self.tqdm:
#     # Move batch to the device
#     Xbatch = Xbatch.to(self.device).float()
#     # Forward pass on contrastive_encoder
#     _, h = self.contrastive_encoder(Xbatch)  # 维数是512， 中间层输出
#     # print(h.size())
#     # input(33333333333333333333333)
#     # print(h.cpu().detach().numpy().shape)
#     # input()
#     # Collect encodings
#     h_list.append(h.cpu().detach().numpy())
#     # Collect labels
#     y_list.append(Ybatch.cpu().detach().numpy().reshape(-1,1))
# # Return values after concatenating encodings along row dimension. Flatten Y labels.



