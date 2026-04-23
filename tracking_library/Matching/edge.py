from ..Splitting.blob import Blob,VoidBlob

class Edge:
    def __init__(self, origin:Blob, dest:Blob, similarity: float):
        self.origin = origin
        self.dest = dest
        self.weight = similarity
        
        if origin.ts - dest.ts != 1 and self.dest != VoidBlob():
            raise ValueError(f"Invalid edge: must point to the previous timestamp {self}")
        
    def __repr__(self):
        return f"Edge({self.origin} -> {self.dest};  similarity={self.weight:.2f})"

    #2 edges are the same if they have same origin and dest (ignore weight)
    def __eq__(self, other):
        if not isinstance(other, Edge):
            return NotImplemented
        return self.origin == other.origin and self.dest == other.dest

    def __hash__(self):
        return hash((self.origin, self.dest))
    

class OriginEdge(Edge):
    def __init__(self, origin: Blob):
        super().__init__(origin, VoidBlob(), 1)

    def __repr__(self):  
        return f"OriginEdge({self.origin})"
   