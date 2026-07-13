from pathlib import Path
from torch.utils.data.dataset import Dataset
from PIL import Image

class DualStreamDataset(Dataset):
    def __init__(self,path,transforms=None):
        self.eye_images = []
        self.mouth_images = []
        self.labels = []
        
        self.dataset_path = Path(path)
        self.transforms = transforms

        eyes_folder = self.dataset_path / "eyes"
        mouths_folder = self.dataset_path / "mouth"

        if "Drowsy" in str(self.dataset_path):
            for image in eyes_folder.glob("*.png"):
                self.eye_images.append(image)
                self.labels.append(1)
            for image in mouths_folder.glob("*.png"):
                self.mouth_images.append(image)
        else:
            for image in eyes_folder.glob("*.png"):
                self.eye_images.append(image)
                self.labels.append(0)
            for image in mouths_folder.glob("*.png"):
                self.mouth_images.append(image)
            

        
