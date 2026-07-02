from src.models.resnet50 import get_resnet50
from src.data.dataloader import get_dataloaders
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import numpy as np
import torch
from tqdm import tqdm
import wandb


def train_one_epoch(model,dataloader,loss_fn,optimizer,device):
    total_loss = []
    total_correct = 0
    total_samples = 0
    
    model.to(device)
    model.train()
    for image, label in tqdm(dataloader,desc="Training",mininterval=1):
       
        image = image.to(device)
        label = label.to(device)
        pred = model(image)
        pred_label = torch.argmax(pred, dim=1)
        total_correct += (pred_label == label).sum().item()
        total_samples += label.size(0)
        loss =  loss_fn(pred, label)
        total_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    accuracy = (total_correct/total_samples) * 100
    return accuracy, np.mean(total_loss)

def validate_one_epoch(model,dataloader,loss_fn,device):
    total_val_loss = []
    total_correct = 0
    total_samples = 0

    model.to(device)
    model.eval()
    with torch.no_grad():
        for image,label in tqdm(dataloader,desc="Validation",mininterval=1):
            image = image.to(device)
            label = label.to(device)
            pred = model(image)
            pred_label = torch.argmax(pred, dim=1)
            total_correct += (pred_label == label).sum().item()
            total_samples += label.size(0)
            loss = loss_fn(pred, label)
            total_val_loss.append(loss.item())
    
    accuracy = (total_correct/total_samples) * 100
    return accuracy, np.mean(total_val_loss)

def train(model,train_dataloader,val_dataloader,loss_fn,optimizer,num_epochs,device,scheduler):
    best_val_accuracy = 0.0
    
    wandb.init(project="driver-drowsiness-detection",config={
        'epochs': num_epochs,
        'learning_rate':optimizer.param_groups[0]['lr']
    })
    for i in range(1,num_epochs+1):
        train_accuracy,train_loss = train_one_epoch(model,train_dataloader,loss_fn,optimizer,device)
        val_accuracy, val_loss = validate_one_epoch(model,val_dataloader,loss_fn,device)
        print(f"Epoch: {i}, Training Loss: {train_loss}, Validation Loss: {val_loss}\nTraining Acccuracy: {train_accuracy}, Validation Acuracy: {val_accuracy}")
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(),"experiments/best_resnet50.pth")
            print(f"Model saved with accuracy: {best_val_accuracy}")
        scheduler.step(val_loss)
        wandb.log({
            'train_loss': train_loss,
            'validation_loss': val_loss,
            'training_accuracy': train_accuracy,
            'validation_accuracy': val_accuracy,
            'epoch': i,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
    wandb.finish()
            

