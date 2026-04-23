from typing import Tuple
import numpy as np
from scipy import ndimage

class SPLITTER:
    @staticmethod
    def naive(motion_mask:np.ndarray, **params) -> Tuple[np.ndarray, int]:
        cc_mask, cc_tot = ndimage.label(motion_mask) # type: ignore
        return cc_mask - 1, cc_tot