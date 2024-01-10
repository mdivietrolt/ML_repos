#Training step

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#from data import dataloadet
import numpy as np
import wandb





#get learning rate
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def trainModel(model, device, BATCH_SIZE, criterion, optimizer, num_epochs, trainloader, testloader):
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, epochs=num_epochs, steps_per_epoch=len(trainloader))

    max_lr = 0.01 #max learning rate
    valid_loss_min = np.Inf
    val_acc = []
    train_loss = []
    train_acc = []
    lr = 0.0 #[] #learning rate
    total_step = len(trainloader)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        accura = 0
        correct = 0
        total=0
        
        optimizer.zero_grad()
        optimizer.step()
        
           
        for i, data in enumerate(trainloader, 1):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device) 

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


            # Record & update learning rate
            #lr.append(get_lr(optimizer))
            sched.step()
            lr = sched.get_last_lr()[0] #fix in wandbi
            print(lr)
          

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
        #print(f'train-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct/total):.4f}')
        wandb.log(
            {
                "TRAIN LOSS" : running_loss/i,
                "TRAIN ACCURACY" : accuracy,
                #"train-loss:" : train_loss,
                #"train-acc" : (100 * correct/total),
                "learning rate" : lr
            }
            
        )

        batch_loss = 0
        total_t=0
        correct_t=0
        with torch.no_grad():
            model.eval()
            for data_t, target_t in (testloader):
                data_t, target_t = data_t.to(device), target_t.to(device)
                outputs_t = model(data_t)
                loss_t = criterion(outputs_t, target_t)
                batch_loss += loss_t.item()
                _,pred_t = torch.max(outputs_t, dim=1)
                correct_t += torch.sum(pred_t==target_t).item()
                total_t += target_t.size(0)
            val_acc.append(100 * correct_t/total_t)
            network_learned = batch_loss < valid_loss_min
            print(f'validation acc: {(100 * correct_t/total_t):.4f}\n')
            wandb.log(
            {
                "validation acc" : (100 * correct_t/total_t)
            }
            
        )
            
    print('End of model training__')

