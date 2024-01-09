import torch
from data import dataloader
from models import resnet
from train import train
import wandb
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F




def training(device):
    
    #data loader
    dataset = dataloader.CustomDatasetsSelector('CIFAR10')
    data_loader = dataloader.DatasetsLoader(dataset,download = True) #dataset=dataset
    print(type(data_loader))
    '''
    # Visualize training set images and print labels with classes
    dataloader.DatasetsLoader.visualize_images(data_loader.get_train_loader, num_images=5,train=True)
    '''
    trainloader, testloader, classes = data_loader.get_data_loader()
    num_classes = len(classes)
    print(f"num_classes", {num_classes})
    
    #model chosen
    model_name = 'ResNet18'
    print(type(model_name))
    model = resnet.model_selector(model_name, num_classes) 
    model.to(device)

    # setup loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    #model training
    num_epochs = 10
    BATCH_SIZE = 256
    try:
        train.trainModel(model, device, BATCH_SIZE, criterion,optimizer, num_epochs, trainloader, testloader)
    except KeyboardInterrupt:
        print('manually interrupt')
        train.save_model(model)
    
    

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"USING DEVICE: ",{device})    
    
    wandb.login()
    wandb.init(
        project = "resnet18-on-Cifar10_test",
    )
        
    training(device)
    
    wandb.finish()