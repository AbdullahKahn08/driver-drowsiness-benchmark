import torch
from models.resnet50 import get_resnet50
from data.dataloader import get_dataloaders
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from training.train import train


device = torch.cuda._get_device(0)

#GLOBAL PARAMETERS
PATH = "G:\Driver Drowsiness Detection\data\raw\Driver Drowsiness Dataset (DDD)"
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_OF_EPOCHS = 10


training,val,test = get_dataloaders(path=PATH)

model = get_resnet50()

loss_function = CrossEntropyLoss()

optimzer =  Adam()

train(model=model,train_dataloader=training,val_dataloader=val,loss_fn=loss_function,optimizer=optimzer,num_epochs=NUM_OF_EPOCHS,device=device)

