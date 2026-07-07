from PIL import Image
import pathlib
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from tqdm import tqdm

LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

def preprocess_images(images,face_mesh,left_eye,right_eye,mouth,eyes_path,mouth_path):
    for image_path in tqdm(images,desc="Progress: ",mininterval=1):
        left_x = []
        left_y = []
        right_x = []
        right_y = []
        mouth_x = []
        mouth_y = []

        image = Image.open(image_path)
        image_array = np.asarray(image)
        results = face_mesh.process(image_array)

        if results.multi_face_landmarks:

            face_landmarks = results.multi_face_landmarks[0]
            landmarks = face_landmarks.landmark

            for i in left_eye:
                left_x.append(landmarks[i].x * image.size[0])
                left_y.append(landmarks[i].y * image.size[1])

            for i in right_eye:
                right_x.append(landmarks[i].x * image.size[0])
                right_y.append(landmarks[i].y * image.size[1])

            for i in mouth:
                mouth_x.append(landmarks[i].x * image.size[0])
                mouth_y.append(landmarks[i].y * image.size[1])
            
            eyes_x = left_x + right_x
            eyes_y = left_y + right_y

            padding = 15

            eye_crop = image.crop((min(eyes_x)-padding,min(eyes_y)-padding,max(eyes_x)+padding,max(eyes_y)+padding))
            mouth_crop = image.crop((min(mouth_x)-padding,min(mouth_y)-padding,max(mouth_x)+padding,max(mouth_y)+padding))

            eye_crop.save(eyes_path/image_path.name)
            mouth_crop.save(mouth_path/image_path.name)
        
        else:
            print("No Face found")

root_directory = pathlib.Path.cwd().parent.parent

drowsy_images_path = root_directory / "data" / "raw" / "Driver Drowsiness Dataset (DDD)" / "Drowsy"
non_drowsy_images_path = root_directory / "data" / "raw" / "Driver Drowsiness Dataset (DDD)" / "Non Drowsy"

drowsy_eyes_path = root_directory / "data" / "processed" / "Drowsy" / "eyes"
drowsy_mouth_path = root_directory / "data" / "processed" / "Drowsy" / "mouth"

non_drowsy_eyes_path = root_directory / "data" / "processed" / "Non Drowsy" / "eyes"
non_drowsy_mouth_path = root_directory / "data" / "processed" / "Non Drowsy" / "mouth"

drowsy_images = [drowsy_image for drowsy_image in drowsy_images_path.iterdir() if drowsy_image.is_file()]
non_drowsy_images = [non_drowsy_image for non_drowsy_image in non_drowsy_images_path.iterdir() if non_drowsy_image.is_file()]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1)

preprocess_images(images=drowsy_images,face_mesh=face_mesh,left_eye=LEFT_EYE,right_eye=RIGHT_EYE,mouth=MOUTH,eyes_path=non_drowsy_eyes_path,mouth_path=non_drowsy_mouth_path)
preprocess_images(images=non_drowsy_images,face_mesh=face_mesh,left_eye=LEFT_EYE,right_eye=RIGHT_EYE,mouth=MOUTH,eyes_path=non_drowsy_eyes_path,mouth_path=non_drowsy_mouth_path)

