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

# imageResize = []
# imageResize.append(colors.to_rgba('grey'))
# imageResize.append(colors.to_rgba('brown'))
# imageResize.append(colors.to_rgba('orange'))
# imageResize.append(colors.to_rgba('olive'))
# imageResize.append(colors.to_rgba('green'))
# imageResize.append(colors.to_rgba('cyan'))
# imageResize.append(colors.to_rgba('blue'))
# imageResize.append(colors.to_rgba('purple'))
# imageResize.append(colors.to_rgba('pink'))
# imageResize.append(colors.to_rgba('red'))
# imageResize.append(colors.to_rgba('cornflowerblue'))
# imageResize.append(colors.to_rgba('tomato'))
# imageResize.append(colors.to_rgba('tan'))
# imageResize.append(colors.to_rgba('yellow'))

# print(imageResize[0][0]*255)
# mmmm=np.array(imageResize)
# mmmm = mmmm * 255
# print(mmmm[1,0:3])
# # N1=10
# # N2=10
# # blank_image = np.zeros((N1,N2,3), np.uint8)
# # blank_image[:,1:5,:] = mmmm[1,1:3]
# # blank_image[:,7:9,:] = (0,22,11)
# # plt.imshow(blank_image)
# # plt.show()
from skimage import io

from scipy import io
mat = np.load('XByOrder2.npy')
io.savemat('aaaaaaaaaaaaaa.mat', {'gene_features': mat})

