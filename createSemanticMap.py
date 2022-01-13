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

from ctypes import *
import numpy as np
import ctypes as C
import cv2
from numpy.ctypeslib import ndpointer
import ctypes
import numpy as np

def cpp_canny(input):
    # if len(img.shape)>=3 and img.shape[-1]>1:
    #     gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    h,w=img.shape[0],img.shape[1] 
    
    # 獲取numpy對象的數據指針
    frame_data = np.asarray(img, dtype=np.uint8)
    frame_data = frame_data.ctypes.data_as(ctypes.c_char_p)
    
    # 設置輸出數據類型爲uint8的指針
    dll.getSuperPixelMap.restype = ctypes.POINTER(ctypes.c_uint8)
     
    # 調用dll裏的cpp_canny函數
    pointer = dll.getSuperPixelMap(h,w,frame_data)
    
    # 從指針指向的地址中讀取數據，並轉爲numpy array
    np_canny =  np.array(np.fromiter(pointer, dtype=np.uint8, count=h*w*1))
    
    return pointer,np_canny.reshape((h,w,1))


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
trainPredictClassModel = torch.load('./personalModels/14classes/1trainPredictClass.pth')


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
    # plt.show()
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
        input("xxxxxxxxxxx")

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



# img5 a x b x channels
def getSemanticPicture2(patchList, num): 
    IImage = torch.tensor(patchList)


    Xbatch = IImage.to(model.device).float()
    _, h  = model.contrastive_encoder(Xbatch)  # 维数是512， 中间层输出 
    data = Variable(h).cpu()

    output  = trainPredictClassModel(data)

    lableV = output.argmax(dim=1)
    # print(output)
    # input("&&&&&&&&&&&&&&")
    return lableV

    




def processImageBySLC(img):
    # if len(img.shape)>=3 and img.shape[-1]>1:
    #     gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    h,w=img.shape[0],img.shape[1] 
    
    # 獲取numpy對象的數據指針
    frame_data = np.asarray(img, dtype=np.uint8)
    frame_data = frame_data.ctypes.data_as(ctypes.c_char_p)
    
    # 設置輸出數據類型爲uint8的指針
    dll.processImageBySLC.restype = ctypes.POINTER(ctypes.c_uint8)
     
    # 調用dll裏的cpp_canny函數
    pointer = dll.processImageBySLC(h,w,frame_data)
    
    # 從指針指向的地址中讀取數據，並轉爲numpy array
    # np_canny =  np.array(np.fromiter(pointer, dtype=np.uint8, count=h*w*1))

def getPatches(num):
    h,w=96,96
    cimg = np.zeros((h,w,3))
    # 獲取numpy對象的數據指針
    frame_data = np.asarray(cimg, dtype=np.uint8)
    frame_data = frame_data.ctypes.data_as(ctypes.c_char_p)
    
    # 設置輸出數據類型爲uint8的指針
    dll.getPatches.restype = ctypes.POINTER(ctypes.c_uint8)
     
    # 調用dll裏的cpp_canny函數
    pointer = dll.getPatches(num)
        
    # 從指針指向的地址中讀取數據，並轉爲numpy array
    data = np.fromiter(pointer, dtype=np.uint8, count=h*w*3)
    np_canny =  np.array(data)
    return pointer,np_canny.reshape((h,w,3))




def processImageFromLabels(lableV):
    # if len(img.shape)>=3 and img.shape[-1]>1:
    #     gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    h,w=lableV.shape[0],1 
    
    # 獲取numpy對象的數據指針
    frame_data = np.asarray(lableV, dtype=np.uint8)
    frame_data = frame_data.ctypes.data_as(ctypes.c_char_p)
    
    # 設置輸出數據類型爲uint8的指針
    dll.processImageFromLabels2.restype = ctypes.POINTER(ctypes.c_uint8)
    # 調用dll裏的cpp_canny函數
    pointer = dll.processImageFromLabels2(h,w,frame_data)
    # 從指針指向的地址中讀取數據，並轉爲numpy array
    data = np.fromiter(pointer, dtype=np.uint8, count=1920*1080*1)
    np_canny =  np.array(data)
    return pointer,np_canny.reshape((1080,1920,1))

import time


dll=CDLL('./SLIC/build/libadder.so') 
#视频的绝对路径 
filename = './videos/fffast2.mp4' 
#可以选择解码工具 
vid = imageio.get_reader(filename, 'ffmpeg') 
for num,im in enumerate(vid): 
    #image的类型是mageio.core.util.Image可用下面这一注释行转换为arrary 
    if(num % 3 == 0 and num > 23):
        t1 = time.time()

        # 通过SLIC处理图像
        processImageBySLC(im)

        patchList = []
        
        t2 = time.time()
        print("time1:",t2-t1)
        for i in range(190):
            ptr2,patches=getPatches(i)

            dll.release(ptr2) # 將內存釋放

            patches = np.transpose(patches, (2,0,1))

            pat_ = skimage.img_as_float(patches).astype(np.float64)

            patchList.append(pat_)

        t2 = time.time()
        print("time2:",t2-t1)
        dll.clearVector() # 將內存釋放

        lableV = getSemanticPicture2(patchList, num)
        # print(lableV.numpy().reshape(1,190))
        # print(lableV.numpy().reshape(10,19))
        t2 = time.time()
        print("time3:",t2-t1)

        ptr,resultM_ = processImageFromLabels(lableV)
        dll.release(ptr) # 將內存釋放


        # resultMap = np.zeros((semanticMap.shape[0],semanticMap.shape[1]))
        # for aa in range(semanticMap.shape[0]):
        #     for bb in range(semanticMap.shape[1]):
        #         resultMap[aa][bb] = lableV[int(semanticMap[aa][bb])]

        t2 = time.time()
        print("time4:",t2-t1)




        plt.figure(1, figsize=(10, 3))
        plt.subplot(121)
        plt.axis('off')

        plt.imshow(im) # ,cmap='Greys'

        plt.figure(1, figsize=(10, 4))
        plt.subplot(122)
        plt.axis('off')
        imagexx = pylab.imshow(resultM_) 
        
        # plt.savefig('./14classesImage/'+str(num)+'picture.png',bbox_inches='tight',dpi=500,pad_inches=0.0)


        t2 = time.time()
        print("time5:",t2-t1)
        print(num)
        print()


        pylab.show() 
        # 加快保存速度
        plt.close()
        # input("xxxxxxvvvxxx")



input('..............')




