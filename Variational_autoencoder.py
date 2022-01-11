__author__ = 'SherlockLiao'

import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from torch.utils.data import Dataset,DataLoader,TensorDataset

if not os.path.exists('./vae_img'):
    os.mkdir('./vae_img')

from sklearn.decomposition import PCA

num_epochs = 100
batch_size = 1
learning_rate = 1e-3


x_train = np.load('latentDataC.npy')
latentDataWhole = x_train
 
print(latentDataWhole.shape)
# input('input ......')
# T SNE 高维数据可视化
from sklearn.manifold import TSNE
from matplotlib import cm
import matplotlib.pyplot as plt
X_tsne = TSNE(n_components=2, random_state=3).fit_transform(latentDataWhole)
X = X_tsne
x_train=torch.from_numpy(x_train)
# print(type(x_train))



train_dataset= TensorDataset(x_train)



class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(512, 256)
        self.fc12 = nn.Linear(256, 64)
        self.fc21 = nn.Linear(64, 2)
        self.fc22 = nn.Linear(64,2)

        self.fc31 = nn.Linear(2, 64)
        self.fc3 = nn.Linear(64, 256)
        self.fc4 = nn.Linear(256, 512)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h1 = F.relu(self.fc12(h1))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc31(z))
        h3 = F.relu(self.fc3(h3))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE()
if torch.cuda.is_available():
    model.cuda()

reconstruction_function = nn.MSELoss(size_average=False)


def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD


# optimizer = optim.Adam(model.parameters(), lr=1e-3)

# for epoch in range(num_epochs):
#     model.train()
#     train_loss = 0
#     for batch_idx, data in enumerate(train_dataset):
#         img = data[0]
#         # img = img.view(img.size(0), -1)
#         img = Variable(img)
#         # print(img.size())
#         # input(11111)
#         if torch.cuda.is_available():
#             img = img.cuda()
#         optimizer.zero_grad()
#         recon_batch, mu, logvar = model(img)
#         loss = loss_function(recon_batch, img, mu, logvar)
#         loss.backward()
#         # print(loss.data.cpu().numpy())
#         train_loss += loss.data.cpu().numpy()
#         optimizer.step()
#         if batch_idx % 100 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch,
#                 batch_idx * len(img),
#                 len(train_dataset), 100. * batch_idx / len(train_dataset),
#                 loss.data.cpu().numpy() / len(img)))

#     print('====> Epoch: {} Average loss: {:.4f}'.format(
#         epoch, train_loss / len(train_dataset)))
#     # if epoch % 10 == 0:
#     #     save = to_img(recon_batch.cpu().data)
#     #     save_image(save, './vae_img/image_{}.png'.format(epoch))

# torch.save(model, './vae.pth')




'''
model = VAE().cuda()
model = torch.load('vae.pth')

for data in dataloader:
    img, aaa = data
    inputt = img.view(28,28)
    print(img.size())
    plt.ion()
    plt.figure(1, figsize=(10, 3))
    plt.subplot(121)
    plt.title('test_data')
    plt.imshow(inputt.numpy(),cmap='Greys')
    plt.show()

    img = img.view(img.size(0), -1)
    img = Variable(img).cuda()
    recon_batch, mu, logvar = model(img)

    print(mu)

    plt.figure(1, figsize=(10, 4))
    plt.subplot(122)
    plt.title('result_data')
    outputt = recon_batch.view(28,28)
    outputt2 = outputt.cpu()

    plt.imshow(outputt2.detach().numpy(),cmap='Greys')
    plt.show()


    input("Press Enter to continue...")
'''


# '''
model = VAE().cuda()
model = torch.load('vae.pth')


fig = plt.figure(2)
# ax = Axes3D(fig)    # 3D 图
count = 0
for data in train_dataset:
    img = data[0]
    # img = img.view(img.size(0), -1)
    img = Variable(img).cuda()
    recon_batch, mu, logvar = model(img)
    print(mu.size())
    cccc = cm.rainbow(int(255*2/9))    # 上色
    # print()
    plt.scatter(mu.data[0].cpu().numpy(), mu.data[1].cpu().numpy(), s=8, c=cccc)
    # ax.text(mu.data[0,0].cpu().numpy(), mu.data[0,1].cpu().numpy(), 1, aaa, backgroundcolor=cccc)  # 标位子
    count = count + 1
    print(count)
    
    # input("Press Enter to continue...")
    if count > 1000:
        break
# ax.set_xlim(-2, 2)
# ax.set_ylim(-2, 2)

plt.show()
# '''