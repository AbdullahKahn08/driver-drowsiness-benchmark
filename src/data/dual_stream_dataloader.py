from torch.utils.data import DataLoader,Dataset, random_split
from src.data.dual_stream_dataset import DualStreamDataset
from src.data.transforms import training_transform_pipeline, testVal_transform_pipeline
from pathlib import Path

class DualStreamTransformSubset(Dataset):
    def __init__(self,subset,transformtype):
        self.subset = subset
        self.transform = transformtype

    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, index):
        eye_image,mouth_image,label = self.subset[index]
        if self.transform:
            eye_image = self.transform(eye_image)
            mouth_image = self.transform(mouth_image)
        return eye_image,mouth_image, label
       

def get_dual_stream_dataloaders(path,batchSize):
    dataSet = DualStreamDataset(path=path)

    trainDataSet, valDataSet, testDataSet = random_split(dataset=dataSet,lengths=[0.8,0.1,0.1],)

    transformedTrain = DualStreamTransformSubset(subset=trainDataSet,transformtype=training_transform_pipeline)
    transformedValidation = DualStreamTransformSubset(subset=valDataSet,transformtype=testVal_transform_pipeline)
    transformedTest = DualStreamTransformSubset(subset=testDataSet,transformtype=testVal_transform_pipeline)

    train_data = DataLoader(dataset=transformedTrain,batch_size=batchSize,shuffle=True,num_workers=3)
    val_data = DataLoader(dataset=transformedValidation,batch_size=batchSize,shuffle=False,num_workers=3)
    test_data = DataLoader(dataset=transformedTest,batch_size=batchSize,shuffle=False,num_workers=3)
    
    return train_data,val_data,test_data


