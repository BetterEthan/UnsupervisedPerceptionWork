#用于测试esoinn网络的使用，并创建触觉数据



import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal
from numpy.random import uniform
from soinn import Soinn
from esoinn import ESoinn
from random import choice


mean = (1, 1)
cov = np.array([[1.1, 0.0], [0.0, 1.3]])
x1 = np.random.multivariate_normal(mean, cov, (500,), 'raise')   # nx2

mean = (14, 2)
cov = np.array([[1.6, 0.0], [0.0, 1]])
x2 = np.random.multivariate_normal(mean, cov, (500,), 'raise')   # nx2


mean = (5, 7)
cov = np.array([[1.1, 0.0], [0.0, 2.1]])
x3 = np.random.multivariate_normal(mean, cov, (500,), 'raise')   # nx2


mean = (7, 14)
cov = np.array([[1.1, 0.0], [0.0, 0.4]])
x4 = np.random.multivariate_normal(mean, cov, (500,), 'raise')   # nx2


mean = (23, 12)
cov = np.array([[2.1, 0.0], [0.0, 1]])
x5 = np.random.multivariate_normal(mean, cov, (500,), 'raise')   # nx2


mean = (13, 12)
cov = np.array([[1.1, 0.0], [0.0, 1]])
x6 = np.random.multivariate_normal(mean, cov, (500,), 'raise')   # nx2
X = np.vstack((x1,x2,x3,x4,x5,x6))
# plt.scatter(x[:, 0], x[:, 1])
# # plt.xlim(-3, 5)
# # plt.ylim(-3, 5)
# plt.show()

data = X
s = ESoinn()


s.fit(X,4000)  #训练


#######模型保存##########
# import pickle
# output_hal = open("tactileModle.pkl", 'wb')
# str = pickle.dumps(s)
# output_hal.write(str)
# output_hal.close()
# input('mmmmmmmmmmmmm')

##########模型加载
# import pickle
# s = ESoinn()
# with open("aaa.pkl",'rb') as file:
#     s  = pickle.loads(file.read())
#####################








nodes = s.nodes

print(len(nodes))

print("end")



# show SOINN's state
plt.plot(data[:, 0], data[:, 1], 'cx')

for k in s.adjacent_mat.keys():
    plt.plot(nodes[k, 0], nodes[k, 1], 'k', c='blue')
# plt.plot(nodes[:, 0], nodes[:, 1], 'ro')

color = ['black', 'red', 'saddlebrown', 'skyblue', 'magenta', 'green', 'gold', 'peru', 'palegreen', 'yellowgreen', 'maroon', 'ivory', 'darkkhaki', 'burlywood', 'aliceblue']

# label density
# for i in range(len(s.nodes)):
#     str_tmp = str(s.density[i]) + " " + str(s.N[i])
#     plt.text(s.nodes[i][0], s.nodes[i][1], str_tmp, family='serif', style='italic', ha='right', wrap=True)

color_dict = {}

print(len(s.nodes))
print(len(s.node_labels))

for i in range(len(s.nodes)):
    if not s.node_labels[i] in color_dict:
        color_dict[s.node_labels[i]] = choice(color)
    plt.plot(s.nodes[i][0], s.nodes[i][1], 'ro', c=color_dict[s.node_labels[i]])
    plt.text(s.nodes[i][0], s.nodes[i][1],s.node_labels[i],ha='center',va='bottom',fontsize=12)  

plt.grid(True)
plt.show(block=False)

input('xxxxxxxxxxx')


