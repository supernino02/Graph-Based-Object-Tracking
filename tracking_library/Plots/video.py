from ..Plots.static_plots import StaticPlot

import math
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import imageio.v2 as imageio # type: ignore




class VideoCreator:

    @staticmethod
    def create_frames_from_plots(
        coll, 
        plot_functions,
        output_dir=None,
        dpi=300,
        create_video=False,
        video_name=None,
        **params):

        if output_dir is None:
            output_dir = "./frames"
            
        os.makedirs(output_dir, exist_ok=True) # type: ignore

        num_plots = len(plot_functions)
        cols = math.ceil(math.sqrt(num_plots))
        rows = math.ceil(num_plots / cols)
        figsize = (StaticPlot.GRID_DIM * cols, StaticPlot.GRID_DIM * rows)  # Use a fixed grid size or pass as param if needed

        filenames = []

        for i, matcher in enumerate(tqdm(coll.history, desc="Processing Frames")):
            fig, axs = plt.subplots(rows, cols, figsize=figsize)
            axs = np.array(axs).reshape(-1)
            fig.suptitle(f"Frame {i+1}", fontsize=16)

            for j, plot_func in enumerate(plot_functions):
                ax = axs[j]
                plot_func(matcher, ax=ax,**params)

            # Hide unused subplots
            for j in range(len(plot_functions), len(axs)):
                axs[j].axis("off")

            filename = os.path.join(output_dir, f"frame_{i:05d}.png")
            fig.savefig(filename, dpi=dpi)
            plt.close(fig)

            filenames.append(filename)

        if create_video:
            if video_name is None:
                video_name = "video.mp4"
            VideoCreator.save_video(filenames, video_name)

        return filenames
    

    @staticmethod
    def create_frames_from_aggregator(
        coll, 
        plot_staticmethod,  # static method that returns a matplotlib Figure
        output_dir=None,
        dpi=300,
        create_video=False,
        video_name=None,
        **params
    ):
        if output_dir is None:
            output_dir = "./frames"
        os.makedirs(output_dir, exist_ok=True)

        filenames = []

        for i, matcher in enumerate(tqdm(coll.history, desc="Creating images")):
            try:
                fig = plot_staticmethod(matcher, **params)
                fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1, wspace=0.1, hspace=0.1)
            except Exception as e:
                print(f"frame {i} skipped:{e}")
                continue

            filename = os.path.join(output_dir, f"frame_{i:05d}.png")
            fig.savefig(filename, dpi=dpi)
            plt.close(fig)

            filenames.append(filename)

        if create_video:
            if video_name is None:
                video_name = "video.mp4"
            VideoCreator.save_video(filenames, video_name)

        return filenames


    @staticmethod
    def save_video(filenames, video_path, fps=7):
        frames = []
        for filename in tqdm(filenames, desc="Merging images"):
            image = imageio.imread(filename)
            frames.append(image)

        for filename in filenames:
            os.remove(filename)

        imageio.mimsave(
            video_path,
            frames,
            fps=fps,
            format='FFMPEG',
            codec='libx264',
            ffmpeg_params=['-crf', '0', '-preset', 'veryslow'],  # max quality
            pixelformat='yuv420p',
            macro_block_size=None
        ) # type: ignore

        return video_path
