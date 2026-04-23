from ..Detection.motion_detector import MotionDetector
from .blob_collection import Blob_collection
from .blob import Blob

import numpy as np
from typing import Callable

class CCSplitter():            
    def __init__(self,
            timestamp, 
            motion_detector:MotionDetector, 
            type:Callable[[np.ndarray], np.ndarray],
            remove_inner:bool=False,
            **params) -> None:
        self.ts = timestamp
        self.detector = motion_detector #wrapper for the object
        
        self.cc_mask, self.cc_tot = type(self.detector.motion_mask,**params)
        
        self.blobs = Blob_collection.create_from_splitter(self)

        if remove_inner:
            new_bc = self.remove_inner_blobs(self.blobs)
            self.renumber_blob_collection(new_bc)
                         
    def __repr__(self):
        return f"CCSplitter(ts={self.ts})"
    
    @staticmethod
    def remove_inner_blobs(bc: Blob_collection) -> Blob_collection:
        if not bc:
            return Blob_collection.create_from_blobs(set())

        blobs = list(bc)
        keep = set()
        for i, blob_i in enumerate(blobs):
            x1_i, y1_i, w_i, h_i = blob_i.bounding_box
            x2_i, y2_i = x1_i + w_i, y1_i + h_i
            is_inside = False

            for j, blob_j in enumerate(blobs):
                if i == j:
                    continue

                x1_j, y1_j, w_j, h_j = blob_j.bounding_box
                x2_j, y2_j = x1_j + w_j, y1_j + h_j

                #check if they are one inside the other (obunding box-wise)
                if (x1_i >= x1_j and y1_i >= y1_j and x2_i <= x2_j and y2_i <= y2_j):
                    is_inside = True
                    break

            if not is_inside:
                keep.add(blob_i)
        return Blob_collection.create_from_blobs(keep)
    
    
    def renumber_blob_collection(self,old_bc: Blob_collection) -> None:
        if not old_bc:
            self.blobs = Blob_collection.create_from_blobs(set())
            self.cc_mask = np.array([], dtype=int)

        #examples fiels
        sample = next(iter(old_bc))
        ts = sample.ts
        image = sample.image
        old_mask = sample._cc_mask

        #sort
        old_indices = sorted(b.idx for b in old_bc)

        #create mapping
        mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(old_indices)}

        #new mask of blobs
        new_mask = np.full_like(old_mask,-1,dtype=int)
        for old_idx, new_idx in mapping.items():
            new_mask[old_mask == old_idx] = new_idx

        #bc
        new_bc = Blob_collection(ts)
        for new_idx in range(len(old_indices)):
            blob = Blob(ts, image, new_mask, new_idx)
            new_bc.add_blob(blob)

        self.blobs = new_bc
        self.cc_mask = new_mask    
    
    def remap_blob_indices(self,new_order: list[int]) -> None:
        #sort
        old_blobs = [b for b in self.blobs]
        old_blobs.sort(key=lambda b: (b.idx),reverse=False)
        if len(old_blobs) != len(new_order):
            raise ValueError(f"new_order length ({len(new_order)}) must match number of blobs ({len(old_blobs)})")

        #mapping
        mapping = {new_order[i]:old_b.idx for i, old_b in enumerate(old_blobs)}
        old_mask = self.cc_mask.copy()
        new_mask = np.full_like(old_mask, -1, dtype=int)

        for old_idx, new_idx in mapping.items():
            new_mask[old_mask == old_idx] = new_idx
            
        #update in place the mask
        self.cc_mask[:] = new_mask[:]
        for b in self.blobs: b.recreate(mapping[b.idx])