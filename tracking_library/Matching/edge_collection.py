from ..Splitting.blob import Blob
from .edge import Edge,OriginEdge


class EdgesCollection:
    @staticmethod
    def create_origin_edge_collection(blob:Blob) -> 'EdgesCollection':
        coll = EdgesCollection(blob.ts)
        coll.add_edge(OriginEdge(blob))
        return coll    
    
    def __init__(self, timestamp: int):
        self.ts = timestamp
        self.edges = set()

    def add_edge(self, e: Edge) -> 'EdgesCollection':
        if e in self.edges:
            raise ValueError(f"{e} already exists in {self}")
        if e.origin.ts != self.ts:
            raise ValueError(f"{e} cannot be added to {self}: ts mismatch")
        self.edges.add(e)
        return self
    
    def extract_originEdges(self) -> 'EdgesCollection':
        origin_edges = EdgesCollection(self.ts)
        for edge in self.edges:
            if isinstance(edge, OriginEdge):
                origin_edges.add_edge(edge)
        return origin_edges
    
    def __repr__(self):
        return f"EdgesCollection(ts={self.ts}, n_edges={len(self.edges)})"
    
    def __iter__(self):
        return iter(self.edges)
    
    def __len__(self):
        return len(self.edges)

    def __getitem__(self, key: tuple[Blob, Blob]) -> Edge:
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError(f"Edge key must be a tuple of (origin_blob, dest_blob), got {key!r}")
        origin, dest = key
        for edge in self.edges:
            if edge.origin == origin and edge.dest == dest:
                return edge
        raise KeyError(f"Edge from {origin} to {dest} not found in {self}")

    def __bool__(self):
        return bool(self.edges)
