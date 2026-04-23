from ..Kalmann.kalmann import KALMANObject

import numpy as np
from scipy.ndimage import center_of_mass



class Blob(KALMANObject):  
    @staticmethod
    def copy(blob: 'Blob') -> 'Blob':
        new_blob = Blob(blob.ts, blob.image, blob._cc_mask, blob.idx)
        new_blob.state = blob.state.copy()
        new_blob.P     = None if blob.P is None else blob.P.copy()
        return new_blob

    def __init__(self, timestamp: int, image: np.ndarray, cc_mask: np.ndarray, idx: int):
        self.ts       = timestamp
        self.image    = image
        self._cc_mask = cc_mask
        self.idx      = idx

        #initialize composed fields
        self._init_from_mask()
        self.reset_info()

    def _init_from_mask(self) -> None:
        mask = (self._cc_mask == self.idx)
        #skimage
        self.centroid = np.array(center_of_mass(mask))

        #how many pixels in the mask
        self.weight = np.sum(mask)

        #evaluaate bounding box
        ys, xs = np.where(mask)
        if ys.size == 0 or xs.size == 0:
            x_min = y_min = 0
            w = h = 1
        else:
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            w = x_max - x_min + 1
            h = y_max - y_min + 1

        self.bounding_box = [int(x_min), int(y_min), int(w), int(h)]

        # KALMAN STATE INITIALIZER
        y_c, x_c = self.centroid
        init_state = np.array([x_c, y_c, 0, 0], dtype='float64').reshape(4, 1)
        super().__init__(init_state)

    #cheamge an alredy created blob (to create a superblob)
    def recreate(self, new_idx: int) -> None:
        self.idx = new_idx
        self._init_from_mask()
        self.reset_info()

    
    def set_state(self, edges_coll) -> None:
        estimates = []
        weights   = []

        measurement = self.centroid[::-1].reshape(2, 1)
        for edge in edges_coll:
            x_upd, P_upd = edge.dest.update(measurement)
            estimates.append((x_upd, P_upd))
            w = edge.dest.weight * edge.weight 
            weights.append(w)

        # fusion
        self.state, self.P = Blob.merge_states(estimates, weights)
    
    #lazy evaluations to avoid memory leaks
    @property
    def mask(self):
        if self._mask is None:
            self._mask = (self._cc_mask == self.idx)
        return self._mask
    
    @property
    def image_masked(self):
        if self._image_masked is None:
            self._image_masked = np.zeros_like(self.image)
            self._image_masked[self.mask] = self.image[self.mask] 
        return self._image_masked
    
    @property
    def mean_rgb(self):
        if self._mean_rgb is None:
            self._mean_rgb = self.image[self.mask].mean(axis=0)
        return self._mean_rgb
    
    #avoid escessive usage of ram
    def reset_info(self):
        self._image_masked = None
        self._mask = None
        self._mean_rgb = None
    
    def __repr__(self):  
        return f"Blob(ts={self.ts}, idx={self.idx})"
        
    # 2 blobs are the same if they have same id and ts
    #it also can paragonate with a tuple (ts,idx)
    def __eq__(self, other):
        if isinstance(other, Blob):
            return self.idx == other.idx and self.ts == other.ts
        elif isinstance(other, tuple) and len(other) == 2:
            return (self.ts, self.idx) == other
        return NotImplemented

    def __hash__(self):
        return hash((self.idx, self.ts))
    
    
    

class VoidBlob(Blob):
    def __init__(self):
        self.idx = -1
        self.ts = -1

    def __repr__(self):  
        return "VoidBlob"
    