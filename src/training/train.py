from models.resnet50 import get_resnet50
from data.dataloader import get_dataloaders
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import numpy as np
import torch


def train_one_epoch(model,dataloader,loss_fn,optimizer,device):
    total_loss = []
    
    model.to(device)
    model.train()
    for image, label in dataloader:
       
        image = image.to(device)
        label = label.to(device)
        pred = model(image)
        loss =  loss_fn(pred, label)
        total_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(total_loss)

def validate_one_epoch(model,dataloader,loss_fn,device):
    total_val_loss = []

    model.to(device)
    model.eval()
    with torch.no_grad():
        for image,label in dataloader:
            image = image.to(device)
            label = label.to(device)
            pred = model(image)
            loss = loss_fn(pred, label)
            total_val_loss.append(loss.item())
    
    return np.mean(total_val_loss)

def train(model,train_dataloader,val_dataloader,loss_fn,optimizer,num_epochs,device):
    for i in range(1,num_epochs+1):
        train_loss = train_one_epoch(model,train_dataloader,loss_fn,optimizer,device)
        val_loss = validate_one_epoch(model,val_dataloader,loss_fn,device)
        print(f"Epoch: {i}, Training Loss: {train_loss}, Validation Loss: {val_loss}")

            

