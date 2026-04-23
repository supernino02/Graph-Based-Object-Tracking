from ..Splitting.blob import Blob
from typing import List

class BlobChain:
    def __init__(self):
        self.chain: List[Blob] = []

    def add_blob(self, blob: Blob):
        if self.chain:
            last_ts = self.chain[-1].ts
            if blob.ts != last_ts + 1:
                raise ValueError(f"Cannot add blob with ts={blob.ts} after ts={last_ts}")
        self.chain.append(blob)

    def __iter__(self):
        return iter(self.chain)

    def __len__(self):
        return len(self.chain)

    def __getitem__(self, idx: int) -> Blob:
        return self.chain[idx]