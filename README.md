# Multi-Object Tracking with Blob Graphs and Kalman Filtering

This repository contains a Computer Vision university project for tracking moving objects in video sequences.

The project is implemented as:

- `tracking_library/`: custom Python tracking library
- `cars.ipynb`: full pipeline for a road traffic sequence
- `subway.ipynb`: full pipeline for a crowded subway sequence

The processing flow is based on background subtraction, blob extraction, frame-to-frame matching, graph-based trajectory linking, and Kalman smoothing.

## Quick Final Results (MP4)

To immediately view the final outputs, open these files:

- Cars (Kalman comparison): [cars_results/paths/video_comp_kal.mp4](cars_results/paths/video_comp_kal.mp4)
- Cars (Centroid comparison): [cars_results/paths/video_comp_centr.mp4](cars_results/paths/video_comp_centr.mp4)
- Subway (Kalman comparison): [subway_results/paths/video_comp_kal.mp4](subway_results/paths/video_comp_kal.mp4)
- Subway (Centroid comparison): [subway_results/paths/video_comp_centr.mp4](subway_results/paths/video_comp_centr.mp4)

If your Markdown viewer supports inline video, these are the two main final videos:

<p><strong>Cars - Final Kalman Comparison</strong></p>
<video src="cars_results/paths/video_comp_kal.mp4" controls width="720"></video>

<p><strong>Subway - Final Kalman Comparison</strong></p>
<video src="subway_results/paths/video_comp_kal.mp4" controls width="720"></video>

## Project Pipeline

Given a sequence of frames and a background image, the code performs the following steps:

1. Motion detection (`DETECTOR`, `MotionDetector`)
2. Connected-component splitting (`CCSplitter`)
3. Blob creation (`Blob`, `Blob_collection`)
4. Blob matching between consecutive frames (`Matcher`, `SIMILARITY`)
5. Temporal graph construction (`Graph`)
6. Trajectory estimation, including Kalman-filtered states (`KALMANObject`)
7. Export of videos and static visualizations

## Repository Hierarchy

Current project hierarchy:

```text
project-root/
  README.md
  README.txt
  report.pdf
  cars.ipynb
  subway.ipynb

  tracking_library/
    Detection/
    Splitting/
    Matching/
    Graph/
    Kalmann/
    Plots/
    frame_collection.py
    misc.py

  sources/
    cars/
      bg.jpg
      frames/
    subway/
      bg.jpg
      frames/

  cars_results/
    original/
    detection/
    matching/
    graph/
    paths/

  subway_results/
    original/
    detection/
    matching/
    graph/
    paths/

  REPORT/
    report.pdf
    cars/
    subway/
```

Top-level folder and file purpose:

- `tracking_library/`: core implementation of the full tracking pipeline
- `sources/`: input data for the notebooks (background images and frame sequences)
- `cars_results/`: generated outputs from the cars experiment
- `subway_results/`: generated outputs from the subway experiment
- `cars.ipynb` and `subway.ipynb`: complete runnable experiments
- `report.pdf`: main report, directly available in the root directory
- `REPORT/`: report package with assets and media
- `README.txt`: original short notes

## Main Library Modules

Inside `tracking_library/`:

- `Detection/`: background subtraction and motion-mask post-processing
- `Splitting/`: connected components, blob objects, and superblobs
- `Matching/`: similarity matrix, edge extraction, and parent selection
- `Graph/`: graph layers, edges, and backward subgraph extraction
- `Kalmann/`: Kalman filter implementation used for blob state smoothing
- `Plots/`: plot aggregators and video generation utilities
- `frame_collection.py`: orchestration class (`FramesCollection`) that runs the full pipeline across all frames
- `misc.py`: helper utilities (`list_files`, `load_image`, `stringify_params`)

## How to Run

### 1. Environment

Recommended Python version: `3.10+`

Install dependencies:

```bash
pip install numpy scipy scikit-image matplotlib networkx imageio tqdm ipython notebook
```

### 2. Run the Notebooks

Open one notebook:

- `cars.ipynb`
- `subway.ipynb`

Run all cells.

The notebooks are configured to:

- read inputs from `sources/...`
- save outputs to `cars_results/` and `subway_results/`

## Generated Outputs

Each experiment generates outputs in these categories:

- `original/video.mp4`: original sequence replay
- `detection/video.mp4`: motion-detection recap
- `matching/recap_matcher.mp4`: frame-to-frame matching recap
- `graph/video.mp4` and `graph/full*.jpg`: graph evolution and snapshots
- `paths/video*.mp4` and `paths/img*.jpg`: trajectory visualizations

Final comparison videos are also generated:

- `video_comp_centr.mp4`: centroid-based comparison
- `video_comp_kal.mp4`: Kalman-based comparison

## Notes and Limitations

- Parameters are manually tuned in the notebooks (thresholds and Kalman matrices).
- Superblob trajectories can produce artifacts in difficult merge/split situations.
- Notebooks contain more visual outputs than the ones discussed in the final report.
