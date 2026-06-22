from models.resnet50 import get_resnet50
from data.dataloader import get_dataloaders
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

DATA_DIR = "data/raw/Driver Drowsiness Dataset (DDD)"

def train_one_epoch(model,dataloader,loss_fn,optimizer):
    pass