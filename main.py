import torch
from src.models.resnet50 import get_resnet50
from src.models.early_fusion_resnet50  import EarlyFusionResNet
from src.models.late_fusion_resnet50 import LateFusionResNet
from src.models.mobilenetv3 import get_mobilenetv3
from src.models.early_fusion_mobilenetv3 import EarlyFusionMobileNetv3
from src.models.late_fusion_mobilenetv3 import LateFusionMobileNetv3
from src.data.dataloader import get_dataloaders
from src.data.dual_stream_dataloader import get_dual_stream_dataloaders
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from src.training.train import train
from src.training.dual_stream_train import dual_stream_train
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse


if __name__ == '__main__':

    RAW_PATH = r"data/raw/Driver Drowsiness Dataset (DDD)"
    PROCESSED_PATH = r"data/processed"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",type=str,required=True,choices=["resnet50","early_fusion_resnet50","late_fusion_resnet50",
                                                                  "mobilenetv3","early_fusion_mobilenetv3","late_fusion_mobilenetv3"])
    parser.add_argument("--epochs",type=int,default=20)
    parser.add_argument("--lr",type=float,default=0.0001)
    parser.add_argument("--batch_size",type=int,default=32)
    args = parser.parse_args()


    if args.model == "resnet50" or args.model == "mobilenetv3":
        if args.model == "resnet50":
            model = get_resnet50()
        else:
            model = get_mobilenetv3()
        training, val, test = get_dataloaders(path=RAW_PATH, batchSize=args.batch_size)

    elif args.model == "early_fusion_resnet50":
        model = EarlyFusionResNet()
        training, val, test = get_dual_stream_dataloaders(path=PROCESSED_PATH, batchSize=args.batch_size)

    elif args.model == "late_fusion_resnet50":
        model = LateFusionResNet()
        training, val, test = get_dual_stream_dataloaders(path=PROCESSED_PATH, batchSize=args.batch_size)

    elif args.model == "early_fusion_mobilenetv3":
        model = EarlyFusionMobileNetv3()
        training, val, test = get_dual_stream_dataloaders(path=PROCESSED_PATH, batchSize=args.batch_size)

    elif args.model == "late_fusion_mobilenetv3":
        model = LateFusionMobileNetv3()
        training, val, test = get_dual_stream_dataloaders(path=PROCESSED_PATH, batchSize=args.batch_size)
        
    loss_function = CrossEntropyLoss()

    optimzer =  Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=args.lr)

    scheduler = ReduceLROnPlateau(optimizer=optimzer,mode='min',patience=5,factor=0.5)

    if args.model in ["resnet50","mobilenetv3"]:
        train(model=model,train_dataloader=training,val_dataloader=val,loss_fn=loss_function,optimizer=optimzer,
          num_epochs=args.epochs,device=device,scheduler=scheduler,save_name=args.model)
    
    else:
        dual_stream_train(model=model,train_dataloader=training,val_dataloader=val,loss_fn=loss_function,optimizer=optimzer,
          num_epochs=args.epochs,device=device,scheduler=scheduler,save_name=args.model)






