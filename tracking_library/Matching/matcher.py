from ..Splitting.cc_splitter import CCSplitter
from .edge_collection import EdgesCollection
from ..Splitting.blob_collection import Blob_collection
from .edge import Edge,OriginEdge
from ..Splitting.blob import Blob

from typing import Callable,Optional
import numpy as np


class Matcher:
    def __init__(self,timestamp :int, curr:CCSplitter, prev: Optional[CCSplitter] = None, **params) -> None:
        self.ts = timestamp
        self.prev_CC = prev
        self.curr_CC = curr

        if prev is None: 
            self.simil_matrix = np.zeros((len(curr.blobs),1),dtype=np.float32) #dummy prev layer
            self.edges = EdgesCollection(self.ts)
            #create dummy edge to  avoid blob
            for a in self.curr_CC.blobs: self.edges.add_edge(OriginEdge(a))

            return
        self.simil_matrix = self.create_matrix(**params)
        self.edges = self.extract_edges(**params)
                
        new_order = self.sort_blobs_idx(self.curr_CC.blobs,self.edges)
        self.curr_CC.remap_blob_indices(new_order)
        
        #again with the new order
        self.simil_matrix = self.create_matrix(**params)
        self.edges = self.extract_edges(**params)
        
        #reset infos for the previous
        self.prev_CC.blobs.flush_blob_info() # type: ignore
     

    def create_matrix(self,type:Callable[[Blob, Blob], float],**params) -> np.ndarray:
        prev_blobs = self.prev_CC.blobs # type: ignore
        curr_blobs = self.curr_CC.blobs
        matrix = np.empty((len(curr_blobs),len(prev_blobs)),dtype=np.float32)

        #iterate over the cells
        for a in curr_blobs:
            for b in prev_blobs:
                matrix[a.idx,b.idx] = type(a,b,**params)
        
        return matrix
    
    def extract_edges(self, treshold: float=0, max_parents:int|None=1, **params) -> EdgesCollection:
        if not 0 <= treshold <= 1:
            raise ValueError(f"treshold ({treshold}) must be in range [0, 1]")
        
        if max_parents is not None and max_parents <= 0:
            raise ValueError(f"max_parents ({max_parents}) must be positive or None")

        coll = EdgesCollection(self.ts)

        for i, row in enumerate(self.simil_matrix):
            a = self.curr_CC.blobs[i]
            #sort 
            sorted_indices = sorted(range(len(row)), key=lambda j: row[j], reverse=True)

            parents_added = 0
            for j in sorted_indices:
                similarity = row[j]
                if similarity > treshold and (max_parents is None or parents_added < max_parents):
                    b = self.prev_CC.blobs[j]  # type: ignore
                    coll.add_edge(Edge(a, b, similarity))
                    parents_added += 1

                if max_parents is not None and parents_added >= max_parents:
                    break

            if parents_added == 0:
                coll.add_edge(OriginEdge(a))

        return coll

    
    def extract_all_edges(self) -> EdgesCollection:
        coll = EdgesCollection(self.ts)

        for i, row in enumerate(self.simil_matrix):
            a = self.curr_CC.blobs[i]
            for j, similarity in enumerate(row):
                b = self.prev_CC.blobs[j]  # type: ignore
                coll.add_edge(Edge(a, b, similarity))

        return coll
    
    @staticmethod
    def sort_blobs_idx(bc:Blob_collection, ec: EdgesCollection) -> list[int]:
        edges = [e for e in ec]
        edges.sort(key=lambda e: (e.dest.idx,e.origin.idx),reverse=False)

        #create new order
        origins = []
        seen = set()
        for e in edges:
            idx = e.origin.idx
            if idx not in seen:
                seen.add(idx)
                origins.append(idx)
        return origins


    def __repr__(self):
        prev_size = len(self.prev_CC.blobs) if self.prev_CC is not None else 0
        return (f"Matcher(ts={self.ts}, "
            f"prev_size={prev_size}, "
            f"curr_size={len(self.curr_CC.blobs)}, "
            f"edges_size={len(self.edges)})")

