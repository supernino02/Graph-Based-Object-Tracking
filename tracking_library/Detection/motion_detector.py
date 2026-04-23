import numpy as np
from typing import Callable
from skimage import morphology as mp # type: ignore


class MotionDetector():        
    def __init__(self, 
            timestamp:int, 
            image:np.ndarray, 
            bg:np.ndarray, 
            type:Callable[[np.ndarray, np.ndarray], np.ndarray], 
            fill_holes:int|None = None,
            remove_small:int|None = None,
            **params) -> None:
        if image.shape != bg.shape:
            raise ValueError(f"Mismatch from image and background: {image.shape} and {bg.shape}")
        self.ts = timestamp
        self.image = image
        self.bg = bg

        self.motion_mask = type(self.image, self.bg, **params)

        #post processing
        if fill_holes is not None and fill_holes > 0:
            self.motion_mask = mp.closing(self.motion_mask, mp.disk(fill_holes))
            
        if remove_small  is not None and remove_small > 0:
            self.motion_mask = mp.remove_small_objects(self.motion_mask, remove_small)
            
    def __repr__(self):
        return f"MotionDetector(ts={self.ts})"
    
    #use the running average to compensate 
    def get_updated_bg(self,bg_rate) -> np.ndarray:
        if not (0 < bg_rate <= 1):
            raise ValueError(f"bg_rate must be in range (0,1]; got {bg_rate}")

        still_mask = ~self.motion_mask #only undetected pixels
        new_bg = np.copy(self.bg)      #start by copying the bg

        #runnin average
        new_bg[still_mask] = (1-bg_rate) * self.bg[still_mask] + bg_rate * self.image[still_mask]
        return new_bg