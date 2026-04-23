import numpy as np


class DETECTOR:
    @staticmethod
    def naive_3D(img:np.ndarray, bg:np.ndarray, tresh:float=0, **params) -> np.ndarray:
        #difference along rgb axis
        res = np.linalg.norm(img - bg, axis=2)
        return res > tresh

    @staticmethod
    def naive_1D(img:np.ndarray, bg:np.ndarray, tresh:float=0, **params) -> np.ndarray:
        #difference along grayscale images
        res = np.abs(img.mean(axis=2) - bg.mean(axis=2))
        return res > tresh