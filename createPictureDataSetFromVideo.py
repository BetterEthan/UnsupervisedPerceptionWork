# coding:utf-8 
import pylab 
import imageio 
#注释的代码执行一次就好，以后都会默认下载完成 
#imageio.plugins.ffmpeg.download() 
import skimage 
import numpy as np 
 



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
from PIL import Image
from skimage import io

# img5 a x b x channels
def getSemanticPicture(img5,num): 
    S = 96
    stride = 96
    N1 = int(img5.shape[0]/S)
    N2 = int(img5.shape[1]/S)
    aanum = 0
    for i in range(N1):
        imageResize = []
        for j in range(N2):
            image_ = img5[i*stride:i*stride+S,j*stride:j*stride+S,:]
            io.imsave('./videoData/image' + str(num) +
            '_' + str(aanum)+'.jpg', image_)
            # image_array = np.array(image_)
            # image_output = Image.fromarray(image_array)
            # image_output.save('./videoData/image'+str(aanum)+'new_car.jpg')
            aanum = aanum + 1
            print(num, aanum)
            # input('.........')



#视频的绝对路径 
filename = '1012-2.mp4' 
#可以选择解码工具 
vid = imageio.get_reader(filename, 'ffmpeg') 
for num,im in enumerate(vid): 
    #image的类型是mageio.core.util.Image可用下面这一注释行转换为arrary 
    if(num % 10 == 0 and num > 0):
        print(num)
        image = skimage.img_as_float(im).astype(np.float64) 
        # fig = pylab.figure() 
        # fig.suptitle('image #{}'.format(num), fontsize=20) 
        # pylab.imshow(im) 
        # pylab.show() 
        getSemanticPicture(image,num)    