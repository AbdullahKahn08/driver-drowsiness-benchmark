import torch
from src.models.resnet50 import get_resnet50
from src.data.dataloader import get_dataloaders
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from src.training.train import train
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model",type=str,required=True,choices=["resnet50","early_fusion_resnet50","late_fusion_resnet50"])
parser.add_argument("--epochs",type=int,default=20)
parser.add_argument("--lr",type=float,default=0.0001)
parser.add_argument("--batch_size",type=int,default=32)
args = parser.parse_args()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #GLOBAL PARAMETERS
    PATH = r"data/raw/Driver Drowsiness Dataset (DDD)"
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 32
    NUM_OF_EPOCHS = 20


    training,val,test = get_dataloaders(path=PATH,batchSize=BATCH_SIZE)

    model = get_resnet50()

    loss_function = CrossEntropyLoss()

    optimzer =  Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=LEARNING_RATE)

    scheduler = ReduceLROnPlateau(optimizer=optimzer,mode='min',patience=5,factor=0.5)

    train(model=model,train_dataloader=training,val_dataloader=val,loss_fn=loss_function,optimizer=optimzer,num_epochs=NUM_OF_EPOCHS,device=device,scheduler=scheduler)

