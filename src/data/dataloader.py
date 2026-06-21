from torch.utils.data import DataLoader,Dataset, random_split
from dataset import DrowsinessDataset
from transforms import training_transform_pipeline, testVal_transform_pipeline
from pathlib import Path

class TransformSubset(Dataset):
    def __init__(self,subset,transformtype):
        self.subset = subset
        self.transform = transformtype

    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, index):
        image,label = self.subset[index]
        transformed_image = self.transform(image)
        return transformed_image, label
       

def get_dataloaders(path):
    dataSet = DrowsinessDataset(path=path)

    trainDataSet, valDataSet, testDataSet = random_split(dataset=dataSet,lengths=[0.8,0.1,0.1],)

    transformedTrain = TransformSubset(subset=trainDataSet,transformtype=training_transform_pipeline)
    transformedValidation = TransformSubset(subset=valDataSet,transformtype=testVal_transform_pipeline)
    transformedTest = TransformSubset(subset=testDataSet,transformtype=testVal_transform_pipeline)

    train_data = DataLoader(dataset=transformedTrain,batch_size=32,shuffle=True,num_workers=3)
    val_data = DataLoader(dataset=transformedValidation,batch_size=32,shuffle=False,num_workers=3)
    test_data = DataLoader(dataset=transformedTest,batch_size=32,shuffle=False,num_workers=3)
    
    return train_data,val_data,test_data


