from ..Splitting.blob import Blob
from .distance import DISTANCE
import numpy as np

class SIMILARITY:
    @staticmethod
    def xy(a: Blob, b: Blob,s_xy=1, **params):   
        dist = DISTANCE.xy(a,b,**params)**2
        return np.exp(-dist/(2*s_xy**2))
    
    @staticmethod
    def rgb(a: Blob, b: Blob, s_rgb=1,**params):
        dist = DISTANCE.rgb(a,b,**params)**2
        return np.exp(-dist/(2*s_rgb**2))
    
    @staticmethod
    def rgb_xy(a: Blob, b: Blob, alpha=0.5,**params):
        xy = SIMILARITY.xy(a,b,**params)
        rgb = SIMILARITY.rgb(a,b,**params)
        return (alpha * rgb + (1-alpha) * xy)
    
    @staticmethod
    def iou(a: Blob, b: Blob,**params):
        i = np.sum(a.mask & b.mask)
        u = np.sum(a.mask | b.mask)
        dist = i/u
        return dist
    
    @staticmethod
    def overlap(a: Blob, b: Blob, **params):
        dist = DISTANCE.overlap(a, b, **params)
        return 1- dist
    
    @staticmethod
    def xy_overlap(a: Blob, b: Blob, alpha=0.5,**params):
        xy = SIMILARITY.xy(a,b,**params)
        overlap = SIMILARITY.overlap(a,b,**params)
        return (alpha * xy + (1-alpha) * overlap)