"""
Author: Talip Ucar
Email: ucabtuc@gmail.com
Version: 0.1

A library of functions to transform and load data.

"""

import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class DataTransforms(object):
    """
    Returns two transformed samples (positive pairs) from a given input for each sample of the dataset.
    """

    def __init__(self, size, train=True, jitter_strength=1.0):
        # Image size
        self.size = size
        # Whether the mode is train or not
        self.train = train
        # Jitter strength to use for color
        js = jitter_strength
        # Define color jitter
        color_jitter = transforms.ColorJitter(0.8 * js, 0.8 * js, 0.8 * js, 0.2 * js)
        # Define transformations for training set
        train_transform = [
                transforms.RandomResizedCrop(size=self.size),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
            ]
        # Define transformations for validation and test sets
        test_transform = [
                transforms.Resize(size=self.size),
                transforms.ToTensor(),
            ]
        # Compose train transformation
        self.train_transform = transforms.Compose(train_transform)
        # Compose validation/test transformation
        self.test_transform = transforms.Compose(test_transform)

    def __call__(self, x):
        # Get transformation to be used, based on the mode = (train, or evaluation/test)
        transform = self.train_transform if self.train else self.test_transform
        # Get two transformed samples from original sample
        xi, xj = transform(x), transform(x)
        # Return samples
        return xi, xj


class Loader(object):
    def __init__(self, config, download=True, get_all_train=False, get_all_test=False, train=True, kwargs={}):
        # Get config
        self.config = config
        # Get mode. If true, it will be used get the transformation for training. Else, get transformation for test
        self.train = train
        # Get the name of the dataset to be used
        dataset_name = config["dataset"]
        # Create dictionary to define number of classes in each dataset
        num_class = {'STL10': 10, 'CIFAR10': 10, 'MNIST': 10}
        # Set main results directory using database name. Exp:  processed_data/dpp19
        paths = config["paths"]
        # data > dataset_name
        file_path = os.path.join(paths["data"], dataset_name.lower())
        # Create the directory if it is missing
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        # Get the datasets  
        train_dataset, test_dataset = self.get_dataset(dataset_name, file_path, download)


        # Get training batch size. If "get_all_train" is True, we get all training examples, else it is defined batch size.
        train_batch_size = config["batch_size"]
        # Get test batch size. If "get_all_test" is True, we get all test examples, else it is defined batch size.
        test_batch_size = config["batch_size"]
        '''
        data_transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),

        ])

        train_dataset = datasets.ImageFolder(root='./../data/hitData',transform=data_transform)
        self.train_loader = DataLoader(dataset=train_dataset,
                                    batch_size=train_batch_size,
                                    shuffle=True)
        
        self.test_loader = self.train_loader
        '''


        # Set the loader for training set
        self.train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=True, **kwargs)
        # Set the loader for test set
        self.test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, drop_last=True, **kwargs)
        
        # from matplotlib import pyplot as plt
        # for i, ((xi, xj), aa) in enumerate(self.train_loader):
        #     print(aa)
        #     plt.subplot(2, 1, 1)
        #     img1 = xi[0,:,:,:].reshape(3,96,96)
        #     img1 = img1.swapaxes(0,1)
        #     img1 = img1.swapaxes(1,2)
        #     print(img1.shape)
        #     plt.imshow(img1)
        #     plt.subplot(2, 1, 2)
        #     img1 = xj[0,:,:,:].reshape(3,96,96)
        #     img1 = img1.swapaxes(0,1)
        #     img1 = img1.swapaxes(1,2)
        #     plt.imshow(img1)
        #     plt.show()
        #     input('please press any keys to continue...')

        # Extract one batch to get image shape
        ((xi, _), _) = self.train_loader.__iter__().__next__()
        # Image shape after removing first dimension (which is batch size)
        self.img_shape = list(xi.size())[1:]
        # Number of classes in the dataset
        self.num_class = num_class[dataset_name]
        # print(len(train_dataset.data),len(test_dataset.data))
        # print(train_batch_size,test_batch_size)
        # input('please press any keys to continue...')


    def get_dataset(self, dataset_name, file_path, download):
        # If True, use unlabelled data in addition to labelled training data
        train_split = 'train+unlabeled' if self.config["unlabelled_data"] else 'train'
        # Get train transformation  数据增强
        train_transform = DataTransforms(size=self.config["img_size"], train=self.train)
        # Get test transformation  数据增强
        test_transform = DataTransforms(size=self.config["img_size"], train=self.train)
        # Create dictionary for loading functions of datasets
        loader_map = {'STL10': datasets.STL10, 'CIFAR10': datasets.CIFAR10, 'MNIST': datasets.MNIST}
        # print(train_split)
        # # print(train_batch_size,test_batch_size)
        # input('please press any keys to continue...')
        # Get dataset
        dataset = loader_map[dataset_name]
        # print(dataset)   # torchvision.datasets.stl10.STL10
        # input('please press any keys to continue...')
        # Get datasets - Note that STL10 has a different input sets such as "split=" rather than "train="

        data_transform = transforms.Compose([
            transforms.Resize((96,96)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

                                    

        if dataset_name in ['STL10']:
            # Training and Validation datasets
            # train_dataset = dataset(file_path, split=train_split, download=download, transform=train_transform)

            # Set the loader for training set
            # self.train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True)

            train_dataset = datasets.ImageFolder(root='../data/SLICSelect',transform=train_transform)

            test_dataset = train_dataset

            # self.train_loader = DataLoader(dataset=train_dataset,
            #                             batch_size=1,
            #                             shuffle=True)

            # from matplotlib import pyplot as plt
            # for i, ((xi, xj), aa) in enumerate(self.train_loader):
            #     print(aa)
            #     plt.subplot(2, 1, 1)
            #     img1 = xi[0,:,:,:].reshape(3,96,96)
            #     img1 = img1.swapaxes(0,1)
            #     img1 = img1.swapaxes(1,2)
            #     print(img1.shape)
            #     plt.imshow(img1)
            #     plt.subplot(2, 1, 2)
            #     img1 = xj[0,:,:,:].reshape(3,96,96)
            #     img1 = img1.swapaxes(0,1)
            #     img1 = img1.swapaxes(1,2)
            #     plt.imshow(img1)
            #     plt.show()
            #     input('please press any keys to continue...')


            # Test dataset
            # test_dataset = dataset(file_path, split='test', download=download, transform=test_transform)
            #105000 8000  训练集有500*10张照片，测试集有800*10张照片, 无标签数据有100000张
            # print(len(train_dataset.data),len(test_dataset.data))   
            # print(train_dataset.data.shape)
            # input('please press any keys to continue...')

        else:
            # Training and Validation datasets
            train_dataset = dataset(file_path, train=True, download=download, transform=train_transform)
            # Test dataset
            test_dataset = dataset(file_path, train=False, download=download, transform=test_transform)
        # Return
        
        
        return train_dataset, test_dataset


