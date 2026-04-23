from ..Matching.matcher import Matcher
from ..Plots.static_plots import StaticPlot
from ..Graph.graph import Graph

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class PlotsAggregator:
    @staticmethod
    def recap_motion_detection( 
        matcher: Matcher,
        title: bool|str = True,
        **params
        ):
        det = matcher.curr_CC.detector

        fig, axs = plt.subplots(2, 2, figsize=(StaticPlot.GRID_DIM * 2, StaticPlot.GRID_DIM * 2))

        # Top left: original image
        axs[0, 0].imshow(det.image, cmap='gray')
        axs[0, 0].set_title("Image")
        axs[0, 0].axis('off')

        # Top right: background image
        axs[0, 1].imshow(det.bg, cmap='gray')
        
        bg_rate = params.get("bg_rate",None)
        axs[0, 1].set_title("Background"+(f" (alpha={bg_rate})"if bg_rate is not None else ""))
        axs[0, 1].axis('off')

        # Bottom left: motion mask
        axs[1, 0].imshow(det.motion_mask, cmap='gray')
        axs[1, 0].set_title("Motion Mask")
        axs[1, 0].axis('off')

        # Bottom right: histogram
        StaticPlot.motion_histogram(det,tresh=params.get("tresh",None), ax=axs[1, 1])
        axs[1,1].set_yticks([]) #remove ticks on the y

        if title:
            fig.suptitle(title if isinstance(title,str) else  f"ts: {det.ts}\nMOTION DETECTION", fontsize=16)

        plt.tight_layout(rect=StaticPlot.TIGHT_RECT) # type: ignore
        return fig
    
    @staticmethod
    def recap_matcher( 
        matcher: Matcher,
        title: bool|str = True,
        **params):

        if matcher.prev_CC is None:
            raise ValueError(f"{matcher} is invalid; must have defined field prev_CC")


        fig, axs = plt.subplots(2, 2, figsize=(StaticPlot.GRID_DIM * 2, StaticPlot.GRID_DIM * 2))


        #prevoius
        StaticPlot.bounding_boxes(matcher.prev_CC,image=False,centroids=False,ids=params.get("ids",True),legend=False,title=False,ax=axs[0, 0])
        axs[0, 0].set_title("Previous")

        #current
        StaticPlot.bounding_boxes(matcher.curr_CC,image=False,centroids=False,ids=params.get("ids",True),legend=False,title=False,ax=axs[0, 1])
        axs[0, 1].set_title("Current")

        #matrix
        StaticPlot.similarity_matrix(matcher,type=params.get("type",None),minimize=params.get("minimize",False),ax=axs[1,0])

        #graph
        StaticPlot.matching_graph(matcher,colorbar=False,type=params.get("type",None),weights=False,ax=axs[1,1],title="BIpartite Graph")


        if title:#TODO: maybe remove the param description
            fig.suptitle(title if isinstance(title,str) else f"ts: {matcher.ts}\nMATCHING BLOBS", fontsize=16)

        plt.tight_layout(rect=StaticPlot.TIGHT_RECT) # type: ignore
        return fig
    
    @staticmethod
    def real_time_graph( 
        matcher: Matcher,
        graph:Graph,
        memory:int=10,
        title: bool|str = True,
        **params):

        fig = plt.figure(figsize=(StaticPlot.GRID_DIM * 2, StaticPlot.GRID_DIM * 2))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])  # 2 rows, 2 cols grid

        ax1 = fig.add_subplot(gs[0, 0])  # top-left
        ax2 = fig.add_subplot(gs[0, 1])  # top-right
        ax3 = fig.add_subplot(gs[1, :])  # bottom row, spans both columns

        #temporary change color
        old = StaticPlot.EMPTY_IMAGE_COLOR 
        StaticPlot.EMPTY_IMAGE_COLOR = (0.463, 1, 0)
        StaticPlot.bounding_boxes(
            matcher.curr_CC,
            image=False,
            centroids=False,
            ids=True,
            legend=True,
            title="Current Image",
            ax=ax1)
        StaticPlot.EMPTY_IMAGE_COLOR = old

        StaticPlot.similarity_matrix(
            matcher,
            minimize=True,
            ax=ax2)

        StaticPlot.draw_graph_mini(
            graph.get_sub_graph(matcher.ts-memory,matcher.ts),
            ts_freq=5,
            legend=False,
            title=f"Graph of the last {memory} frames",
            ax=ax3)


        if title:
            fig.suptitle(title if isinstance(title,str) else  f"ts: {matcher.ts}\nREAL TIME GRAPH", fontsize=16)

        plt.tight_layout(rect=StaticPlot.TIGHT_RECT) # type: ignore
        return fig

    @staticmethod
    def real_time_trails(
        matcher: Matcher,
        graph: Graph,
        n_chains: int = 1,
        boundary_boxes: bool=False,
        title: bool | str = True,
        **params
    ):
        ts = matcher.ts
        fig, ax = plt.subplots(1,2, figsize=(StaticPlot.GRID_DIM * 2, StaticPlot.GRID_DIM * 1))

        for a in ax.flat:
            if boundary_boxes:
                StaticPlot.bounding_boxes(matcher.curr_CC, legend=False, ids=False, ax=a)
            else:
                StaticPlot.original_frame(matcher,ax=a)
            a.axis('off')

        subgraph = graph.get_sub_graph(1, ts)
        bc = subgraph.get_ending_blobs(ts)

        StaticPlot.draw_trail_2d(
            graph, bc, trail_kalman=False, max_chains=n_chains,
            ax=ax[0], title="Centroids", legend=False
        )
        StaticPlot.draw_trail_2d(
            graph, bc, trail_centroid=False, max_chains=n_chains,
            ax=ax[1], title="Kalman Filter", legend=False
        )

        if title:
            title_str = title if isinstance(title, str) else 'Trajectories'
            fig.suptitle(f"ts: {ts}\n{title_str}", fontsize=16)

        plt.tight_layout(rect=StaticPlot.TIGHT_RECT)
        return fig
    
    @staticmethod
    def real_time_trail_comparison(
        matcher: Matcher,
        graph_1: Graph,
        graph_n: Graph,
        kalman: bool = False,
        boundary_boxes: bool = False,
        title: bool | str = True,
        **params
    ):
        ts = matcher.ts
        modes = [
            (graph_1, 100,"Trajectories of the blobs (max 1 parent)"),
            (graph_n, 1,  "Trajectories of the blobs (max N parent)"),
            (graph_n, -1, "Trajectories of the Super-Blobs"),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(StaticPlot.GRID_DIM*3, StaticPlot.GRID_DIM))

        for ax, (g, max_chains, suptitle) in zip(axes, modes):
            if boundary_boxes:
                StaticPlot.bounding_boxes(matcher.curr_CC,legend=False, ids=False, title=False,ax=ax)
            else:
                StaticPlot.original_frame(matcher, ax=ax)
            ax.axis('off')

            bc = g.get_sub_graph(1, ts).get_ending_blobs(ts)
            StaticPlot.draw_trail_2d(
                g, bc,
                trail_kalman=kalman, trail_centroid=not kalman,
                max_chains=max_chains,
                ax=ax,
                legend=False
            )
            ax.set_title(suptitle, fontsize=10)

        #suptitle
        if title:
            title_str = title if isinstance(title, str) else f"{'Kalman' if kalman else 'Centroid'} trajectories"
            fig.suptitle(f"ts: {ts}\n{title_str}", fontsize=16)

        fig.tight_layout(rect=StaticPlot.TIGHT_RECT)
        return fig