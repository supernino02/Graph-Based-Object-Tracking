from ..Splitting.blob_collection import Blob_collection
from ..Splitting.superblob import SuperBlob
from ..Matching.edge_collection import EdgesCollection
from ..Splitting.blob import Blob,VoidBlob
from ..Matching.edge import Edge,OriginEdge
from .blob_chain import BlobChain

from typing import List
import numpy as np

class Graph:    
    def __init__(self) -> None:
        self.layers = {}
        self.edges = {}

    @property
    def last_ts(self):
        return sorted(self.layers)[-1]

    def __repr__(self):
        return f"Graph(n_layers={len(self.layers)}, n_edges_collections={len(self.edges)})"


    def add_blob_coll(self, bc: Blob_collection) -> None:
        if bc.ts in self.layers:
            raise ValueError(f"BlobsCollection with timestamp {bc.ts} already exists:{self.layers[bc.ts]}")
        self.layers[bc.ts] = bc

    def get_blob_coll(self, ts: int) -> Blob_collection|None:
        return self.layers.get(ts, None)
        
    def add_edges_coll(self, ec: EdgesCollection,update_state:bool=False) -> None:
        if ec.ts not in self.layers:
            raise ValueError(f"BlobsCollection with timestamp {ec.ts} does not exists")
        if ec.ts in self.edges:
            raise ValueError(f"EdgesCollection with timestamp {ec.ts} already exists: {self.edges[ec.ts]}")
        
        #check correspondence
        for e in ec: 
            self._check_edge(e)           
        
        self.edges[ec.ts] = ec

        #kalmann filtering,if needed
        if not update_state:return 

        #remove OriginEdges
        edges = [e for e in ec if not isinstance(e, OriginEdge)]

        dict = {}
        for e in edges:
            if e.origin not in dict: dict[e.origin] = EdgesCollection(ec.ts)
            dict[e.origin].add_edge(e)

        #set the state conisdering all the edges created
        for blob, edges_for_blob in dict.items():
            blob.set_state(edges_for_blob)

    def get_edges_coll(self, ts: int) -> EdgesCollection|None:
        return self.edges.get(ts, None)

    def get_sub_graph(self, start_ts: int, end_ts: int) -> 'Graph':
        min_ts = min(self.layers)
        max_ts = max(self.layers)

        if start_ts == -1: start_ts = min_ts
        if end_ts == -1: end_ts = max_ts

        if start_ts == end_ts == -1: return self

        if start_ts > end_ts:   raise ValueError("start_ts must be less than or equal to end_ts.")
        if not self.layers:     raise ValueError("The graph has no layers.")

        if start_ts < min_ts: start_ts = max(start_ts, min_ts)
        if end_ts > max_ts:   end_ts = min(end_ts, max_ts)

        subgraph = Graph()
        for ts in range(start_ts, end_ts + 1):
            subgraph.add_blob_coll(self.layers[ts])

            edges_coll = self.edges[ts].extract_originEdges() if ts == start_ts else self.edges[ts]
            subgraph.add_edges_coll(edges_coll)

        return subgraph

    def get_last_sub_graph(self, n: int) -> "Graph":
        if n  == -1: return self
        if n <= 0:           raise ValueError("n must be a positive integer.")
        if not self.layers:  raise ValueError("The graph has no layers.")
        if n > len(self.layers):
            n = len(self.layers)
            #raise ValueError(f"Requested {n} layers, but graph only contains {len(self.layers)} layers.")

        sorted_ts = sorted(self.layers)
        return self.get_sub_graph(sorted_ts[-n], sorted_ts[-1])

    # verify the it can exists, before adding to the graph
    def _check_edge(self, e: Edge) -> None:
        #origin control
        ts_from = e.origin.ts
        coll_from = self.get_blob_coll(ts_from)
        if coll_from is None:          raise ValueError(f"Missing Blob_collection for origin timestamp {ts_from}")
        if e.origin not in coll_from:  raise ValueError(f"Origin blob {e.origin} not found in {coll_from}")    
    
        #destination control
        if e == OriginEdge(e.origin): return #if is a origin node, no control needed

        #consistency control
        ts_to = e.dest.ts
        coll_to = self.get_blob_coll(ts_to)
        if coll_to is None:           raise ValueError(f"Missing Blob_collection for destination timestamp {ts_to}")        
        if e.dest not in coll_to:     raise ValueError(f"Destination blob {e.dest} not found in {coll_to}")


    def get_superblob_chains(
        self,
        idx: int,
        ts: int = -1,
        max_chains: int = -1
    ) -> List[BlobChain]:
        if ts == -1: ts = self.last_ts

        bc_start = self.get_blob_coll(ts)
        if bc_start is None: raise ValueError(f"No layer at ts={ts}")
        start_blob = bc_start.get_blob(ts, idx)
        if start_blob is None: raise ValueError(f"No blob idx={idx} at ts={ts}")

        chain = BlobChain()
        prev = None

        subg = self.get_backward_subgraph(idx, ts, max_chains)
        for t in sorted(subg.layers.keys()):
            bc_layer = subg.get_blob_coll(t)
            superblob = SuperBlob(bc_layer) # type: ignore
            chain.add_blob(superblob)

            if prev is not None:
                meas = np.array(superblob.centroid[::-1]).reshape(2, 1)
                superblob.state, superblob.P = prev.update(meas)
                
            ##better usage of ram, slower runtime
            bc_layer.flush_blob_info() # type:ignore 
            superblob.reset_info()
            
            prev = superblob

        return [chain]

    def get_ending_blobs(self, ts_list: List[int]|int|None = None) -> Blob_collection:
        if isinstance(ts_list, int):  layers_to_check = [ts_list]
        elif ts_list is None:         layers_to_check = list(self.layers.keys())
        else:                         layers_to_check = ts_list

        result = Blob_collection(None)
        for ts in layers_to_check:
            bc = self.layers.get(ts)
            if bc is None:
                continue

            next_ts = ts + 1
            dest_set = set()
            ec_next = self.get_edges_coll(next_ts)
            if ec_next is not None:
                for e in ec_next:
                    if not isinstance(e, OriginEdge):
                        dest_set.add(e.dest)

            for b in bc.blobs:
                if b not in dest_set:
                    result.blobs.add(b)
        return result

    def get_backward_subgraph(self, idx: int, ts: int = -1, max_chains: int = -1) -> 'Graph':
        if ts == -1: ts = self.last_ts

        #extract blob
        bc = self.get_blob_coll(ts)
        if bc is None:
            raise ValueError(f"No layer at ts={ts}")
        start_blob = bc.get_blob(ts, idx)
        if start_blob is None:
            raise ValueError(f"No blob idx={idx} at ts={ts}")

        #void graph
        discovered_blobs = {start_blob}
        discovered_edges = set()
        current_layer = {start_blob}

        #backpropagation
        while current_layer:
            next_layer = set()
            for blob in current_layer:
                ec = self.get_edges_coll(blob.ts)
                if not ec:
                    continue

                #only going out from the curr blob
                outgoing = [e for e in ec if e.origin == blob]
                if not outgoing: continue

                #sort
                outgoing.sort(key=lambda e: e.weight, reverse=True)
                to_take = outgoing if max_chains == -1 else outgoing[:max_chains]

                for e in to_take:
                    discovered_edges.add(e)

                    #avoid originEdge
                    if not isinstance(e, OriginEdge):
                        if e.dest not in discovered_blobs:
                            discovered_blobs.add(e.dest)
                            next_layer.add(e.dest)

            current_layer = next_layer #nexzt

        result = Graph()

        #bc
        blobs_by_ts = {}
        for b in discovered_blobs:  
            blobs_by_ts.setdefault(b.ts, set()).add(b)
        for t, bs in blobs_by_ts.items():
            bc = Blob_collection.create_from_blobs(bs)
            result.add_blob_coll(bc)

        #ec
        edges_by_ts = {}
        for e in discovered_edges:
            edges_by_ts.setdefault(e.origin.ts, set()).add(e)
        for t, es in edges_by_ts.items():
            ec = EdgesCollection(t)
            for e in es:
                ec.add_edge(e)
            result.add_edges_coll(ec)

        return result

