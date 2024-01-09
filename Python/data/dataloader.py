import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from logging import root



class CustomDatasetsSelector():
    '''
    Input: 
    - dataset: pytorch built-in dataset name (e.g ,'CIFAR10', 'FashionMNIST', 'MNIST')
    
    Return:
    - traindataset
    - testdataset
    - classes
    '''
    def __init__(self, dataset_name='CIFAR10', BATCH_SIZE = 256, download = False, pin_memory = True, shuffle= True, num_workers=4):
        # Define dataset folder
        self.data_dir = './' + dataset_name

        '''
        # Define data transformations
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        '''
        
        # Define data transformations
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        if dataset_name == 'CIFAR10':
            self.traindataset = torchvision.datasets.CIFAR10(root=self.data_dir, train=True, download=True, transform=self.transform) #
            self.testdataset = torchvision.datasets.CIFAR10(root=self.data_dir, train=False, download=True,transform=self.transform)
        elif dataset_name == 'FashionMNIST':
            self.traindataset =  torchvision.datasets.FashionMNIST(root=self.data_dir, train=True, download=True, transform=self.transform)
            self.testdataset = torchvision.datasets.FashionMNIST(root=self.data_dir, train=False, download=True, transform=self.transform)
        elif dataset_name == 'MNIST':
            self.traindataset = torchvision.datasets.MNIST(root=self.data_dir, train=True, download=True, transform=self.transform)
            self.testdataset = torchvision.datasets.MNIST(root=self.data_dir, train=False, download=True, transform=self.transform)
            # Add more datasets as needed
        else:
            raise ValueError("Unsupported dataset name.") 
            
        self.classes = self.traindataset.classes

            
        
        
    

class DatasetsLoader():

    def __init__(self, CustomDatasetsSelector:CustomDatasetsSelector, BATCH_SIZE = 256, download = False, pin_memory = True, shuffle= True, num_workers=4):
        
        self.batch_size = BATCH_SIZE
        self.shuffle = shuffle
        self.classes = CustomDatasetsSelector.classes
        
      
        # Create DataLoader for training set
        self.train_loader = DataLoader(
            dataset=CustomDatasetsSelector.traindataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )

        # Create DataLoader for test set
        self.test_loader = DataLoader(
            dataset=CustomDatasetsSelector.testdataset,
            batch_size=self.batch_size,
            shuffle=False,  # No need to shuffle test set
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )

    def get_data_loader(self):
        return self.train_loader, self.test_loader, self.classes  
       
              
    '''    
    def visualize_images(self, num_images=5, train=True):
        # Visualize a few images from the training or test set
        if train:
            data_iter = iter(self.train_loader)
            dataset_name = self.train_dataset_name
        else:
            data_iter = iter(self.test_loader)
            dataset_name = self.test_dataset_name

        images, labels = next(data_iter) #fixed

        for i in range(num_images):
            plt.subplot(1, num_images, i + 1)
            img = images[i] / 2 + 0.5  # Unnormalize
            np_img = img.numpy()
            plt.imshow(np.transpose(np_img, (1, 2, 0)))
            #plt.title(f"Label: {labels[i].item()} - Class: {self.classes[labels[i].item()]}")
            plt.title(f"{self.classes[labels[i].item()]}")
            plt.axis('off')
        plt.suptitle(f"{dataset_name} {'Training' if train else 'Test'} Set")
        plt.show()
        '''

    def print_labels(self, num_labels=10, train=True):
        # Print the first few labels in the training or test set
        if train:
            data_iter = iter(self.train_loader)
        else:
            data_iter = iter(self.test_loader)

        _, labels = next(data_iter)
        #print(f"Labels: {labels[:num_labels].tolist()}")
        print("Labels:")
        for i in range(num_labels):
            print(f"  Label: {labels[i].item()} - Class: {self.classes[labels[i].item()]}")
        
    def get_classes(self):
        return self.classes
    
    def get_train_labels(self):
        return self.train_dataset.targets.tolist()

    def get_test_labels(self):
        return self.test_dataset.targets.tolist()

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader      

    

     


