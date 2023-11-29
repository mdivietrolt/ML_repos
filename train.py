#Resnet for CIFAR10

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataset import trainloader, testloader, BATCH_SIZE
import numpy as np
from resnet import ResNet18,resnet18_model


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Load resnet18 model for CIFAR10
resnet18_model = ResNet18(num_classes=10).to(device)

# setup loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet18_model.parameters(), lr=0.01, momentum=0.9)

#get learning rate
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# resnet18 training
num_epochs = 10
# Set up one-cycle learning rate scheduler
sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, epochs=num_epochs, steps_per_epoch=len(trainloader))

max_lr = 0.01 #max learning rate
valid_loss_min = np.Inf
val_acc = []
train_loss = []
train_acc = []
lr = [] #learning rate
total_step = len(trainloader)
for epoch in range(num_epochs):
    resnet18_model.train()
    running_loss = 0.0
    accura = 0
    correct = 0
    total=0
    for i, data in enumerate(trainloader, 1):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device) 

        optimizer.zero_grad()

        outputs = resnet18_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Record & update learning rate
        lr.append(get_lr(optimizer))
        sched.step()

        _, predicted_classes = torch.max(outputs, 1)
        accura += (predicted_classes == labels).sum().item()
        running_loss += loss.item()
        correct += torch.sum(predicted_classes == labels).item()
        total += labels.size(0)
        if i % 100 == 0:  # print every 100 batch
            #print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / i))
            print ('\nEpoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch, num_epochs, i, total_step, loss.item()))
    accuracy = accura / (i * BATCH_SIZE)
    print("TRAIN LOSS : {:.4f}".format(running_loss/i))
    print("TRAIN ACCURACY : {:.2f}".format(accuracy))
    train_acc.append(100 * correct / total)
    train_loss.append(running_loss/total_step)
    print(f'train-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct/total):.4f}')

    batch_loss = 0
    total_t=0
    correct_t=0
    with torch.no_grad():
        resnet18_model.eval()
        for data_t, target_t in (testloader):
            data_t, target_t = data_t.to(device), target_t.to(device)
            outputs_t = resnet18_model(data_t)
            loss_t = criterion(outputs_t, target_t)
            batch_loss += loss_t.item()
            _,pred_t = torch.max(outputs_t, dim=1)
            correct_t += torch.sum(pred_t==target_t).item()
            total_t += target_t.size(0)
        val_acc.append(100 * correct_t/total_t)
        network_learned = batch_loss < valid_loss_min
        print(f'validation acc: {(100 * correct_t/total_t):.4f}\n')


print('End of model training')

