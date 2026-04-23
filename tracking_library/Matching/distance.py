import numpy as np
from ..Splitting.blob import Blob

class DISTANCE:
    @staticmethod
    def iou(a: Blob, b: Blob, **params):
        i = np.sum(a.mask & b.mask)
        u = np.sum(a.mask | b.mask)
        return -i/u #minus to make that bigger==movement
    
    @staticmethod
    def overlap(a: Blob, b: Blob, **params):
        a_only = np.sum(a.mask & ~b.mask) / np.sum(a.mask)
        b_only = np.sum(b.mask & ~a.mask) / np.sum(b.mask)
        return min(a_only,b_only)
    
    @staticmethod
    def xy(a: Blob, b: Blob, **params):
        dist = np.linalg.norm(a.centroid - b.centroid, 2) # type: ignore
        return dist
    
    @staticmethod
    def rgb(a: Blob, b: Blob, **params):
        dist = np.linalg.norm((a.mean_rgb-b.mean_rgb), 2)
        return dist