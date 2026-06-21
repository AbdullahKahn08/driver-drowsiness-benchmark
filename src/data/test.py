import pathlib

path = "data/raw/Driver Drowsiness Detection DDD"
print(pathlib.Path(path.glob("*.png")))