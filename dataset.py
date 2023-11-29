import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image


#=======================================================================================
#Data preparation

'''
Preparing the Data: 
1. downloading the CIFAR10 dataset 
2. creating PyTorch datasets

'''
#==================================================================================

# Define transformations for the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
BATCH_SIZE = 256

# Get CIFAR-10 data for training
data_dir = './CIFAR10'
trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,num_workers=4,pin_memory=True,drop_last=True)
testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True,num_workers=4,pin_memory=True,drop_last=True)


#load images from data folder
def load_images_from_folder(folder_path):
    images = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
            image_path = os.path.join(folder_path, filename)
            img = Image.open(image_path)
            images.append(img)
            filenames.append(filename)
    return images, filenames

#load and show images from data folder
def load_and_show_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
            image_path = os.path.join(folder_path, filename)
            img = Image.open(image_path)
            #print(f'img: {(img)}\n')
            images.append(img)
            # Visualizza l'immagine
            plt.imshow(img)
            plt.title(f'Immagine: {filename}')
            plt.show()
    return images

def show_batch(dataloader):
    for images, labels in dataloader:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images[:64], nrow=8).permute(1, 2, 0))
        break
    
images, labels = next(iter(trainloader)) 
print(f'images-size: {(images.shape)}\n')

out = torchvision.utils.make_grid(images)
print(f'out-size: {(out.shape)}\n')

image_folder = data_dir
#images = load_and_show_images(image_folder)
show_batch(trainloader)