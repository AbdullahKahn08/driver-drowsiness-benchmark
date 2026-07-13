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

        
