from pathlib import Path
from torch.utils.data.dataset import Dataset
from PIL import Image

class DualStreamDataset(Dataset):
    def __init__(self,path,transforms=None):
        self.eyes = []
        self.mouths = []
        self.labels = []
        self.transforms = transforms

        root_folder = Path(path)
        drowsy_folder = root_folder / "Drowsy"
        non_drowsy_folder = root_folder / "Non Drowsy"

        for image in (drowsy_folder/"eyes").glob("*.png"):
            mouth_image_path = drowsy_folder / "mouth" / image.name
            if mouth_image_path.exists():
                self.eyes.append(image)
                self.mouths.append(mouth_image_path)
                self.labels.append(1)
        
        for image in (non_drowsy_folder/"eyes").glob("*.png"):
            mouth_image_path = non_drowsy_folder / "mouth" / image.name
            if mouth_image_path.exists():
                self.eyes.append(image)
                self.mouths.append(mouth_image_path)
                self.labels.append(0)
    
    def __len__(self):
        if len(self.eyes) == len(self.mouths) == len(self.labels):
            return len(self.eyes)
        else:
            raise ValueError("Mismatach occured in the index of eyes and mouth")

    def __getitem__(self, index):
        eye_image = Image.open(self.eyes[index])
        mouth_image = Image.open(self.mouths[index])

        if self.transforms: 
            eye_image = self.transforms(eye_image)
            mouth_image = self.transforms(mouth_image)

        return eye_image,mouth_image,self.labels[index]

        
