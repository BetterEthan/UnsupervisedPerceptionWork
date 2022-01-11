import numpy as np
# from skimage import io
# from skimage import transform
import matplotlib.pyplot as plt
import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.utils import make_grid



# ''' 读取并设定lable
def file_name2(file_dir):
    L=[]
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                L.append(os.path.join(root, file))
                # L.append(file)

    return L


def listdir(path, list_name):  #传入存储的list
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)



def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        print(root) #当前目录路径
        # print(dirs) #当前路径下所有子目录
        print(files) #当前路径下所有非目录子文件
        input("Press Enter to continue...")



fileAddress = '../data/SLICSelect'
aa = file_name2(fileAddress)

for data in aa:
    with open(fileAddress + "/aaaaaaa.txt","a") as f:
        f.write(data+ '\t' + '3' '\n')  # 自带文件关闭功能，不需要再写f.close()
    print(data)
    # input("Press Enter to continue...")
# '''