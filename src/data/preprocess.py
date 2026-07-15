from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

def preprocess(images,eye_points,mouth_points,eye_path,mouth_path):
    for image in tqdm(images.glob("*.png"),desc="Progress",mininterval=1):
        img = Image.open(image)
        eye_crop = img.crop(eye_points)
        mouth_crop = img.crop(mouth_points)
        eye_crop.save(eye_path/image.name)
        mouth_crop.save(mouth_path/image.name)

EYES_COORDINATES = (0,0,227,110)
MOUTH_COORDINATES = (0,130,227,227)

root_directory = Path.cwd().parent.parent

drowsy_images_path = root_directory / "data" / "raw" / "Driver Drowsiness Dataset (DDD)" / "Drowsy"
non_drowsy_images_path = root_directory / "data" / "raw" / "Driver Drowsiness Dataset (DDD)" / "Non Drowsy"

drowsy_eyes_path = root_directory / "data" / "processed" / "Drowsy" / "eyes"
drowsy_mouth_path = root_directory / "data" / "processed" / "Drowsy" / "mouth"

non_drowsy_eyes_path = root_directory / "data" / "processed" / "Non Drowsy" / "eyes"
non_drowsy_mouth_path = root_directory / "data" / "processed" / "Non Drowsy" / "mouth"

preprocess(images=drowsy_images_path,eye_points=EYES_COORDINATES,mouth_points=MOUTH_COORDINATES,
           eye_path=drowsy_eyes_path,mouth_path=drowsy_mouth_path)

preprocess(images=non_drowsy_images_path,eye_points=EYES_COORDINATES,mouth_points=MOUTH_COORDINATES,
           eye_path=non_drowsy_eyes_path,mouth_path=non_drowsy_mouth_path)



