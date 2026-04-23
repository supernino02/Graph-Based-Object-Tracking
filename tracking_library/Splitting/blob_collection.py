from .blob import Blob
from typing import Optional

class Blob_collection():
    @classmethod
    def create_from_splitter(cls, splitter: 'CCSplitter') -> 'Blob_collection': # type: ignore
        coll = cls(splitter.ts)
        for idx in range(splitter.cc_tot):
            blob = Blob(splitter.ts, splitter.detector.image, splitter.cc_mask, idx)
            coll.add_blob(blob)
        return coll

    @classmethod
    def create_from_blobs(cls, blobs: set[Blob]) -> 'Blob_collection':
        if not blobs:
            return cls(None)
        ts_values = {b.ts for b in blobs}
        if len(ts_values) != 1:
            raise ValueError("All blobs must have the same timestamp")
        ts = ts_values.pop()
        coll = cls(ts)
        coll.blobs = set(blobs)
        return coll
        
    def __init__(self, timestamp: Optional[int] = None) -> None:
        self.ts = timestamp
        self.blobs = set() 
    
    def add_blob(self, blob: Blob) -> None:
        if blob in self.blobs:
            raise ValueError(f"{blob} already exists in {self.get_blob(blob.ts,blob.idx)}")
        if self.ts is not None and blob.ts != self.ts:
            raise ValueError(f"{blob} cannot be added to {self}")
        self.blobs.add(blob)

    def get_blob(self, ts:int, idx: int) -> Optional['Blob']:
        return next((b for b in self.blobs if b == (ts,idx)), None)
    
    def get_sub_collection(self, ts: int) -> Optional['Blob_collection']:
        return Blob_collection.create_from_blobs({b for b in self if b.ts == ts})

    def flush_blob_info(self) -> None:
        for b in self: b.reset_info()
    
    def __iter__(self):
        return iter(self.blobs)

    def __len__(self):
        return len(self.blobs)
    
    def __getitem__(self, idx: int) -> Blob:
        if self.ts is None:
            raise ValueError(f"{self} cannot get item {idx}; self.ts cannot be null")
        blob = self.get_blob(self.ts, idx)
        if blob is None:
            raise KeyError(f"Blob {idx} not in {self}")
        return blob
    
    def __bool__(self):
        return bool(self.blobs)
    
    def __eq__(self, other):
        if not isinstance(other, Blob_collection):
            return NotImplemented
        return self.ts == other.ts and self.blobs == other.blobs

    def __hash__(self):
        return hash((self.ts, frozenset(self.blobs)))
        
    def __repr__(self):
        return f"Blob_collection(ts={self.ts}, n_blobs={len(self.blobs)})"
