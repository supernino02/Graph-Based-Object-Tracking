# Directory Structure Documentation

This document explains the overall directory structure of the repository, with a particular focus on the `tracking_library`.

## Top-Level Directories and Files

- `cars.ipynb` and `subway.ipynb`: Jupyter Notebooks demonstrating the tracking system on specific video sequences.
- `requirements.txt`: List of dependencies required to run the library and the notebooks.
- `README.md`: Overview of the project and its goals.
- `REPORT/`: Contains the detailed technical report of the project, explaining the theory, methodology, and the intermediate outputs (like `original/`, `detection/`, `matching/`, `splitting/`).
- `cars_results/` and `subway_results/`: Directories where the visual outputs of the pipeline are saved (e.g., intermediate frames and generated videos). Subdirectories include:
  - `original/`: Original extracted frames.
  - `detection/`: Results from the motion detection phase.
  - `splitting/`: Results from separating merged objects.
  - `matching/`: Frame-by-frame object matching visualization.
  - `graph/`: Graph representations of object trajectories.
  - `paths/`: Final paths highlighted on the video sequences.

## The `tracking_library`

The core of the tracking system is implemented in `tracking_library`. It is designed in a modular way, separating the responsibilities into different sub-packages:

### `Detection`
Handles the problem of discerning moving objects from the background.
- `motion_detector.py`: Implements background subtraction and motion detection.
- `detector.py`: General interfaces for detection logic.

### `Splitting`
Deals with identifying distinct blobs and splitting them if multiple objects overlap or merge.
- `blob.py`, `superblob.py`, `blob_collection.py`: Data structures to represent distinct objects, merged objects, and collections of them for a given frame.
- `cc_splitter.py`, `splitter.py`: Algorithms to split Connected Components (CC) into separate blobs using heuristics or known object characteristics.

### `Matching`
Responsible for linking blobs across consecutive frames.
- `matcher.py`: Core logic for matching objects over time.
- `similarity.py`, `distance.py`: Metrics to evaluate how likely it is for a blob in frame N to be the same blob in frame N+1.
- `edge.py`, `edge_collection.py`: Graph-based representations for matches.

### `Kalmann`
- `kalmann.py`: Implements Kalman filters to estimate and predict the state (position, velocity) of objects over time, aiding in the tracking and resolving occlusions.

### `Graph`
Analyzes the entire lifetime of objects across the sequence.
- `graph.py`: Builds a global graph of all blobs across all frames to find optimal trajectories.
- `blob_chain.py`: Represents the path (chain) of a single tracked object over time.

### `Plots`
Utilities to generated visualizations of the intermediate and final steps.
- `static_plots.py`, `multiple_plots.py`: Generating 2D layout plots of frames, detections, and matches.
- `video.py`: Assembling processed frames back into an output video.
- `tree_viewer.py`: Rendering structural or graph trees if needed.

### Root Library Files
- `__init__.py`: Exposes the primary interfaces so that the library can be imported easily in the notebooks.
- `frame_collection.py`: Manages the processing of a sequence of frames, orchestrating detection, splitting, and matching.
- `misc.py`: Miscellaneous helper functions (e.g., file loading, image operations).
