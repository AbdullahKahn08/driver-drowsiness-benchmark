from pathlib import Path
from torch.utils.data.dataset import Dataset
from PIL import Image


class DrowsinessDataset(Dataset):
    def __init__(self,path,transform):

        self.image_paths = []
        self.image_labels = []

        self.dataset_path = Path(path)
        self.transform = transform


        drowsy_folder = self.dataset_path / "Drowsy"
        for image in drowsy_folder.glob("*.png"):
            self.image_paths.append(image)
            self.image_labels.append(1)
        
        non_drowsy_folder = self.dataset_path / "Non Drowsy"
        for image in non_drowsy_folder.glob("*.png"):
            self.image_paths.append(image)
            self.image_labels.append(0)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx])
        label = self.image_labels[idx]
        
        if self.transform == None:
            transformed_image = self.transform(image)
            return transformed_image, label
            
        else:
            return image, label
        
        

