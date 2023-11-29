# Plot metrics for train and validation step

import matplotlib.pyplot as plt
from train import *

def plot_accuracy(acc_history):
    fig = plt.figure(figsize=(20,10))
    plt.title("Train-Validation Accuracy")
    plt.title("Accuracy")
    plt.plot(acc_history, label='training')
    plt.plot(val_acc, label='validation')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('accuracy', fontsize=12)
    plt.legend(loc='best')

def plot_losses(train_loss_history):
    fig = plt.figure(figsize=(20,10))
    plt.title('Loss vs. No. of epochs')
    train_losses = ['loss' for x in train_loss_history]
    plt.plot(train_loss, '-bx', label='training')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')
 
def plot_lr(lr_history):
    fig = plt.figure(figsize=(20,10))
    lr = ['lr' for x in lr_history]
    plt.plot(lr_history, '-bx', label='training')
    plt.xlabel('Batch no.')
    plt.ylabel('Learning rate')
    plt.title('Learning Rate vs. Batch no.'); 

plot_accuracy(train_acc)
plot_losses(train_loss)
plot_lr(lr)