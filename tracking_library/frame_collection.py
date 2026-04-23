from .Detection.detector import DETECTOR
from .Splitting.splitter import SPLITTER
from .Matching.similarity import SIMILARITY
from .Matching.matcher import Matcher
from .Detection.motion_detector import MotionDetector
from .Splitting.cc_splitter import CCSplitter
from .Splitting.blob_collection import Blob_collection
from .misc import load_image
from .Graph.graph import Graph

import numpy as np
from .Matching.edge_collection import EdgesCollection
from tqdm.notebook import tqdm # type: ignore

class FramesCollection():
    DEFAULT_DETECTOR_PARAMS = {
        'type': DETECTOR.naive_3D, 
        'tresh': 0.5
    }

    DEFAULT_SPLITTER_PARAMS = {
        'type': SPLITTER.naive
    }

    DEFAULT_MATCHER_PARAMS = {
        'type': SIMILARITY.xy
    }
    
    TIMESTAMP = 0

    @classmethod
    def update_timestamp(cls) -> int:
        cls.TIMESTAMP += 1
        return cls.TIMESTAMP

    @classmethod
    def get_timestamp(cls) -> int:
        return cls.TIMESTAMP

    @classmethod
    def reset_timestamp(cls) -> None:
        cls.TIMESTAMP = 0

    def __init__(self, frames_paths, bg_path, name="default", detector_params=None,splitter_params=None,matcher_params=None) -> None:
        self.name = name
        self.frames_bank = frames_paths

        self.graph = Graph()
        self.history = []
        self.next_frames = frames_paths

        self.curr_bg = load_image(bg_path)

        #eventually initialize them
        self.detector_params = self.check_detector_params(detector_params)
        self.splitter_params = self.check_splitter_params(splitter_params)
        self.matcher_params = self.check_matcher_params(matcher_params)
    
    def __repr__(self):
        return f"FramesCollection(name={self.name!r})"
    
    def check_detector_params(self,params = None) -> dict:
        if params is None: return self.DEFAULT_DETECTOR_PARAMS
    
        if not isinstance(params, dict):
            raise TypeError(f"Expected dict for params, got {type(params)}")
        
        if 'type' not in params.keys():
            raise KeyError(f"Must be defined a type of DETECTOR")
    
        return params    

    def check_splitter_params(self,params = None) -> dict:
        if params is None: return self.DEFAULT_SPLITTER_PARAMS
    
        if not isinstance(params, dict):
            raise TypeError(f"Expected dict for params, got {type(params)}")
        
        if 'type' not in params.keys():
            raise KeyError(f"Must be defined a type of SPLITTER")
    
        return params
    
    def check_matcher_params(self,params = None) -> dict:
        if params is None: return self.DEFAULT_MATCHER_PARAMS
    
        if not isinstance(params, dict):
            raise TypeError(f"Expected dict for params, got {type(params)}")
        
        if 'type' not in params.keys():
            raise KeyError(f"Must be defined a type of SIMILARITY")
    
        return params
    
    
    def _compute_frame(self) -> Matcher: 
        curr_frame = self.__get_new_frame()
        curr_ts = self.__class__.update_timestamp()

        #FIND MOTION
        detector = MotionDetector(curr_ts,curr_frame,self.curr_bg, **self.detector_params)
        
        #if needed, update the background
        if ((bg_rate := self.detector_params.get("bg_rate", 0)) > 0):
            self.curr_bg = detector.get_updated_bg(bg_rate)

        #FIND BLOBS
        blob_splitter = CCSplitter(curr_ts,detector, **self.splitter_params)

        #MATCH WITH PREVIOUS
        #retrieve the last produced frame, if exists
        prev_splitter = self.history[-1].curr_CC if self.history else None
        matcher = Matcher(curr_ts,blob_splitter,prev_splitter,**self.matcher_params)

        self.history.append(matcher)
        return matcher
        
    def _update_graph(self,bc:Blob_collection,ec:EdgesCollection) -> None:  
        self.graph.add_blob_coll(bc)
        self.graph.add_edges_coll(ec,update_state=True)
        
    def __get_new_frame(self) -> np.ndarray:
        if len(self.next_frames) == 0:
            raise Exception("No frames left to compute")
        img = load_image(self.next_frames[0])
        self.next_frames = self.next_frames[1:]
        return img
    
    def compute(self) -> 'FramesCollection':         
        if len(self.next_frames) == 0: 
            return self
        for _ in tqdm(range(len(self.next_frames)), desc="Computing Frames"):
            matcher = self._compute_frame()
            self._update_graph(matcher.curr_CC.blobs,matcher.edges)

        self.reset_timestamp()  
        return self