import os
import numpy as np
from skimage import io, img_as_float # type: ignore

def list_files(folder_path) -> np.ndarray:
    files = os.listdir(folder_path)
    valid_files = []
        
    for file in files:
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path): valid_files.append(file_path) 
        
    return np.array(valid_files,dtype=object) 
    
def load_image(path) -> np.ndarray:
    img = io.imread(path)
    img = img_as_float(img)
    return img

def stringify_params(params:dict):
    def stringify(v):
        if callable(v):
            return f"{v.__qualname__}"
        return v
    return {k: stringify(v) for k, v in params.items()}
