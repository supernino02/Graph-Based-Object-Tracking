from .blob import Blob
from .blob_collection import Blob_collection

import numpy as np
from typing import List

class SuperBlob(Blob):
    def __init__(self, bc: Blob_collection, idx: int = 1):
        first = next(iter(bc.blobs))  #supppose al the blobs with the same image
        combined_mask = np.zeros_like(first.mask, dtype=int)
        #merge all the masks
        for b in bc.blobs: combined_mask |= b.mask
        bc.flush_blob_info() #reduce ram usage
        #create it as normal blob
        super().__init__(bc.ts, first.image, combined_mask, idx) # type: ignore
        self.components = bc