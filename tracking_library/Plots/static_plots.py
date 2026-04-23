from functools import wraps
from typing import Callable,List
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Patch,Rectangle
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D




from ..Detection.motion_detector import MotionDetector
from ..Splitting.blob_collection import Blob_collection
from ..Splitting.blob import Blob
from ..Splitting.cc_splitter import CCSplitter
from ..Matching.matcher import Matcher
from ..Matching.edge import OriginEdge

from ..Graph.graph import Graph
from ..frame_collection import FramesCollection



#for a easier way of defining the functions
def ensure_ax(func):
    @wraps(func)
    def wrapper(cls, *args, ax: Axes | None = None, **kwargs):
        created_ax = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(cls.GRID_DIM, cls.GRID_DIM))
            created_ax = True

        # Call the original method with the guaranteed ax
        result = func(cls, *args, ax=ax, **kwargs)

        # If we created the figure/axis, show it
        if created_ax:
            plt.show()

        return result
    return wrapper


class StaticPlot:
    #dims
    GRID_DIM = 5
    TIGHT_RECT = [0, 0, 1, 0.95]

    #colors
    EMPTY_IMAGE_COLOR = (0.5, 0.5, 0.5)
    TEXT_COLOR = (0,0,0)

    #histogram
    HIST_COLOR =  (0.486, 0.851, 0.91)
    TRESH_COLOR = (0.95, 0.094, 0.251)

    #bb motion
    BOUNDING_BOX_COLOR = (0, 0.5, 1)
    CENTROID_COLOR = (0.871, 0.318, 0.318)

    #graph
    NODE_COLOR = (0.18, 0.812, 0.788)
    ORIGIN_NODE_COLOR = (0.016, 0.529, 0)
    END_NODE_COLOR = (0.871, 0.318, 0.318)

    GRAPH_PREV_LAYER_COLOR = (0.922, 0.671, 0.306)
    GRAPH_CURR_LAYER_COLOR = (0.078, 0.78, 0.059)

    TRAIL_CENTROID = (1,0,0)
    TRAIL_KALMAN = (0,1,0)

    ### UTILIITY ###
    @classmethod
    def create_empty_image(cls,img:np.ndarray,color=None) -> np.ndarray:
        return np.ones_like(img) * np.array(color if color is not None else cls.EMPTY_IMAGE_COLOR, dtype=np.float32)
    
    @classmethod
    @ensure_ax
    def original_frame(cls,
        matcher: Matcher,
        title:bool|str= True,
        ax: Axes|None=None,        
        ) -> None:
        assert ax is not None

        img = matcher.curr_CC.detector.image
        ax.imshow(img)
        ax.axis("off")
            
        
        if title:
            ax.set_title(title if isinstance(title,str) else "Original Video")
    
    ### MOTIONDETECTOR ###
    @classmethod
    @ensure_ax
    def motion_histogram(cls,
        det: MotionDetector,
        tresh:float|None=None,
        legend: bool= True,
        title: bool|str= True,
        ax: Axes|None=None
        ) -> None: 
        assert ax is not None
        
        diff = np.linalg.norm(det.image - det.bg, axis=2)
        values = diff.ravel()

        bins = np.linspace(0, 1, 100)
        hist, bin_edges = np.histogram(values, bins=bins, density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]

        ax.bar(bin_centers, hist, width=bin_width, color=cls.HIST_COLOR, edgecolor='black')
        if tresh is not None: 
            if not (0 <= tresh <= 1): raise ValueError("Tresh ({tresh}) must be in range[0,1]")                
            ax.axvline(x=tresh, color=cls.TRESH_COLOR, linestyle=':', linewidth=2)
        ax.set_xlim(-0.05,tresh*1.25 if tresh is not None else 1.05)
        ax.set_xlabel("Difference")
        ax.set_ylabel("Pixel Count")

        if title:
            ax.set_title(title if isinstance(title,str) else "Pixel-wise Differences")

        if legend:
            legend_elements = [
                Patch(facecolor=cls.HIST_COLOR,  edgecolor='black', label=f'Histogram'),
                Patch(facecolor=cls.TRESH_COLOR, edgecolor='black', label=f'Treshold={tresh}') if tresh is not None else None
            ]
            ax.legend(
                handles=[elem for elem in legend_elements if elem is not None],
                loc='upper right',
                frameon=True,              # Show the legend frame
                facecolor='white',         # Set background color
                edgecolor='black'          # Optional: frame edge color
            )

    ### CCSPLITTER ###
    @classmethod
    @ensure_ax
    def bounding_boxes(cls,
        splitter: CCSplitter,
        centroids:bool=True,
        ids:bool = True,
        image:bool=True,
        lwidth:float=2,
        legend: bool= True,
        title:bool|str= True,
        ax: Axes|None=None,        
        ) -> None:
        assert ax is not None

        #load image
        if image: 
            img = splitter.detector.image
        else:
            img = cls.create_empty_image(splitter.detector.image)
            mask = splitter.detector.motion_mask
            img[mask] = splitter.detector.image[mask]

        ax.imshow(img)
        ax.axis("off")

        for b in splitter.blobs:
            x, y, w, h = b.bounding_box

            rect = Rectangle((x, y), w, h, linewidth=lwidth, edgecolor=cls.BOUNDING_BOX_COLOR, facecolor='none')
            ax.add_patch(rect)

            if centroids:
                cy, cx = b.centroid
                ax.plot(cx, cy, marker='x', color = cls.CENTROID_COLOR)  # red 'x'

            if ids:
                ax.text(x, y - 2, str(b.idx),
                        color=cls.TEXT_COLOR,
                        fontsize=10,
                        verticalalignment='bottom',
                        horizontalalignment='left')
            
        
        if title:
            ax.set_title(title if isinstance(title,str) else "Bounding Boxes")

        if legend:
            legend_elements = [
                Patch(facecolor=cls.BOUNDING_BOX_COLOR,  edgecolor='black', label=f'Bounding boxes'),
                Patch(facecolor=cls.CENTROID_COLOR, edgecolor='black', label=f'Blob\'s centroid') if centroids else None
            ]
            ax.legend(
                handles=[elem for elem in legend_elements if elem is not None],
                loc='upper right',
                frameon=True,              # Show the legend frame
                facecolor='white',         # Set background color
                edgecolor='black'          # Optional: frame edge color
            )

    ### MATCHER ###
    @classmethod
    @ensure_ax
    def similarity_matrix(cls,
        matcher: Matcher,
        type: Callable|None = None,
        colorbar:bool=True,
        title:bool|str = True,
        minimize:bool=False,
        ax: Axes|None=None       
        ) -> None:
        assert ax is not None

        matrix = matcher.simil_matrix
        cax = ax.imshow(matrix, cmap='viridis_r')
                # Make the axes itself square-shaped, so the displayed image fits a square box
        ax.set_aspect('auto')  # allow stretching of pixels (default)
        ax.set_box_aspect(1)   # THIS forces the axes box itself to be a square

        # Fix limits to cover the whole matrix
        nrows, ncols = matrix.shape
        ax.set_xlim(-0.5, ncols - 0.5)
        ax.set_ylim(nrows - 0.5, -0.5)
        ax.grid(False)

        if minimize: 
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        else:
            #put ticks as ints
            ax.set_xticks(np.arange(matrix.shape[1]))
            ax.set_yticks(np.arange(matrix.shape[0]))
            ax.set_xticklabels([f"{j}" for j in range(matrix.shape[1])])
            ax.set_yticklabels([f"{j}" for j in range(matrix.shape[0])])

            for r in range(matrix.shape[0]):
                for c in range(matrix.shape[1]):
                    ax.text(c, r, f"{matrix[r, c]:.2f}", ha='center', va='center', color='white' if matrix[r, c] > 0.5 else 'black')

        ax.set_xlabel(f"Blobs at ts = {matcher.prev_CC.ts}" if matcher.prev_CC is not None else f"Dummy blobs")
        ax.set_ylabel(f"Blobs at ts = {matcher.curr_CC.ts}")

        if colorbar:
            cbar = ax.figure.colorbar(cax, ax=ax) # type: ignore
            cbar.set_label("Similarity", fontsize=10)
        
        if title:
            ax.set_title(title if isinstance(title,str) else ("Similarity Matrix " + (f"of {type.__qualname__}" if type is not None else "")))


    #### GRAPH ####
    #utilities
    @classmethod
    def _compute_node_positions(cls, 
            layer_nodes: dict[int, list], 
            x_spacing=200, 
            y_spacing=150,
            fill_space=True
        ) -> dict:
        pos = {}
        sorted_layers = sorted(layer_nodes.keys())

        if not fill_space:
            max_nodes_in_layer = max((len(nodes) for nodes in layer_nodes.values()), default=1)
            max_height = (max_nodes_in_layer - 1) * y_spacing

        for i, layer in enumerate(sorted_layers):
            nodes = sorted(layer_nodes[layer], key=lambda x: x[1], reverse=True)
            n = len(nodes)
            x = i * x_spacing

            if fill_space:
                height = (n - 1) * y_spacing
                y_positions = [-height / 2 + j * y_spacing for j in range(n)]
            else:
                if n == 1:
                    y_positions = [0]
                else:
                    y_positions = np.linspace(-max_height / 2, max_height / 2, n) # type: ignore

            for node, y in zip(nodes, y_positions):
                pos[node] = (x, y)

        return pos

    @classmethod
    def _draw_layer_rectangles(
        cls,
        ax: Axes,
        layer_nodes: dict[int, list],
        pos: dict,
        layer_colors: dict[int, str],
        layer_names: dict[int, str],
        padding_x=50,
        padding_y=37.5
    ):

        for layer, nodes in layer_nodes.items():
            if not nodes:
                continue
            xs = [pos[n][0] for n in nodes]
            ys = [pos[n][1] for n in nodes]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)


            if layer in layer_colors.keys():
                rect = Rectangle(
                    (x_min - padding_x, y_min - padding_y),
                    (x_max - x_min) + 2 * padding_x,
                    (y_max - y_min) + 2 * padding_y,
                    linewidth=2,
                    edgecolor=layer_colors.get(layer, cls.GRAPH_CURR_LAYER_COLOR),
                    facecolor='none',
                    linestyle='--',
                    zorder=0
                )
                ax.add_patch(rect)

                ax.text(
                    (x_min + x_max) / 2, y_max + padding_y,
                    layer_names.get(layer, f"ts={layer}"),
                    ha='center', va='bottom',
                    fontsize=15, fontweight='bold',
                    color=layer_colors.get(layer, cls.GRAPH_CURR_LAYER_COLOR),
                )

    @classmethod
    def _get_node_color(cls, node, origin_nodes, graph):
        #if is no targer, else is a dead end
        has_incoming = any(True for _ in graph.predecessors(node))

        if node in origin_nodes:
            return cls.ORIGIN_NODE_COLOR
        elif not has_incoming:
            return cls.END_NODE_COLOR
        else:
            return cls.NODE_COLOR

    @classmethod
    def _graph_legend(cls,ax):
        legend_elements = [
            Patch(facecolor=cls.NODE_COLOR, edgecolor='black', label='Regular Node'),
            Patch(facecolor=cls.ORIGIN_NODE_COLOR, edgecolor='black', label='Origin Node'),
            Patch(facecolor=cls.END_NODE_COLOR, edgecolor='black', label='End Node'),
            ]
        ax.legend(
            handles=legend_elements,
            loc='upper right',
            frameon=True,
            facecolor='white',
            edgecolor='black'
        )

    #callable
    @classmethod
    @ensure_ax
    def matching_graph(cls,
        matcher: Matcher,
        type: Callable | None = None,
        weights: bool = True,
        rects: bool = True,
        colorbar: bool = True,
        title: bool|str = True,
        ax: Axes | None = None,
        ) -> None:
        assert ax is not None

        if matcher.prev_CC is None:
            raise ValueError(f"{matcher} is invalid; must have defined field prev_CC")

        ax.set_axis_off()
        G = nx.DiGraph()

        for blob in matcher.prev_CC.blobs:
            G.add_node((blob.ts, blob.idx), label=blob.idx, layer=0)
        for blob in matcher.curr_CC.blobs:
            G.add_node((blob.ts, blob.idx), label=blob.idx, layer=1)

        for edge in matcher.extract_all_edges():
            G.add_edge((edge.origin.ts, edge.origin.idx), (edge.dest.ts, edge.dest.idx), weight=edge.weight)

        layer_nodes = {0: [], 1: []}
        for node, data in G.nodes(data=True):
            layer_nodes[data['layer']].append(node)

        x_spacing, y_spacing = 150, 250
        pos = cls._compute_node_positions(layer_nodes, x_spacing, y_spacing)

        if rects:
            layer_colors = {
                0: cls.GRAPH_PREV_LAYER_COLOR,
                1: cls.GRAPH_CURR_LAYER_COLOR
            }
            layer_names = {
                0: "Previous",
                1: "Current"
            }
            cls._draw_layer_rectangles(ax, layer_nodes, pos, layer_colors, layer_names, x_spacing * 0.25, y_spacing * 0.25) # type: ignore

        node_labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_nodes(G, pos, node_size=1000, node_color=[cls.NODE_COLOR] * G.number_of_nodes(), ax=ax) # type: ignore
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, ax=ax)

        edge_weights = [d["weight"] for _, _, d in G.edges(data=True)]
        vmin, vmax = 0.0, 1.0
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        edge_cmap = plt.colormaps['viridis_r'] # type: ignore
        edge_colors = [edge_cmap(norm(w)) for w in edge_weights]
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=[2 + 4 * w for w in edge_weights], arrows=False, ax=ax) # type: ignore

        if weights:
            edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_weight='bold', label_pos=0.75, ax=ax)

        if colorbar:
            sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=norm) # type: ignore
            sm.set_array([])
            cbar = ax.figure.colorbar(sm, ax=ax, shrink=0.7, pad=0.03)
            cbar.set_label("Edge Weight", fontsize=10)

        if title:
            ax.set_title(title if isinstance(title,str) else (f"Similarity graph " + (f"of {type.__qualname__}" if type is not None else "")))

    @classmethod
    @ensure_ax
    def draw_graph(cls,
        graph: Graph,
        weights: bool = True,
        rects: list[int] | None = None,
        colorbar: bool = True,
        legend:bool =True,
        title: bool|str = True,
        ax: Axes | None = None,
        ) -> None:
        assert ax is not None

        if not graph.layers:
            raise ValueError("Graph has no layers to draw.")
        if rects is None:
            rects = []

        ax.set_axis_off()
        G = nx.DiGraph()
        origin_nodes = set()

        for ts, blob_coll in graph.layers.items():
            for blob in blob_coll.blobs:
                G.add_node((ts, blob.idx), label=blob.idx, layer=ts)

        for ec in graph.edges.values():
            for edge in ec:
                if edge == OriginEdge(edge.origin):
                    origin_nodes.add((edge.origin.ts, edge.origin.idx))
                    continue
                G.add_edge((edge.origin.ts, edge.origin.idx), (edge.dest.ts, edge.dest.idx), weight=edge.weight)

        layer_nodes = {}
        for node, data in G.nodes(data=True):
            layer = data['layer']
            layer_nodes.setdefault(layer, []).append(node)

        x_spacing, y_spacing = 300, 250
        pos = cls._compute_node_positions(layer_nodes, x_spacing, y_spacing)

        if rects:
            if rects == [-1]: rects = list(layer_nodes.keys())
            layer_colors = {layer: cls.GRAPH_PREV_LAYER_COLOR for layer in rects}
            layer_names = {layer: f"ts={layer}" for layer in rects}
            cls._draw_layer_rectangles(ax, layer_nodes, pos, layer_colors, layer_names, x_spacing * 0.4, y_spacing * 0.4) # type: ignore

        node_labels = nx.get_node_attributes(G, 'label')
        node_colors = [
            cls._get_node_color(node, origin_nodes, G)
            for node in G.nodes()
        ]
        nx.draw_networkx_nodes(G, pos, node_size=1000, node_color=node_colors, ax=ax) # type: ignore
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, ax=ax)

        edge_weights = [d["weight"] for _, _, d in G.edges(data=True)]
        vmin, vmax = 0.0, 1.0
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        edge_cmap = plt.colormaps['viridis_r'] # type: ignore
        edge_colors = [edge_cmap(norm(w)) for w in edge_weights]
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=[2 + 4 * w for w in edge_weights], arrows=False, ax=ax) # type: ignore

        if weights:
            edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_weight='bold', label_pos=0.5, ax=ax)

        if colorbar:
            sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=norm) # type: ignore
            sm.set_array([])
            cbar = ax.figure.colorbar(sm, ax=ax, shrink=0.7, pad=0.03)
            cbar.set_label("Edge Weight", fontsize=10)

        if legend:cls._graph_legend(ax)

        if title:
            ax.set_title(title if isinstance(title,str) else "Graph over time")

    @classmethod
    @ensure_ax
    def draw_graph_mini(cls,
        graph: Graph,
        legend: bool = True,
        ts_freq: int | None = None,
        title: bool | str = True,
        ax: Axes | None = None,
    ) -> None:
        assert ax is not None

        if not graph.layers:
            raise ValueError("Graph has no layers to draw.")

        ax.set_axis_off()
        G = nx.DiGraph()
        origin_nodes = set()

        for ts, blob_coll in graph.layers.items():
            for blob in blob_coll.blobs:
                G.add_node((ts, blob.idx), layer=ts)

        for ec in graph.edges.values():
            for edge in ec:
                if edge == OriginEdge(edge.origin):
                    origin_nodes.add((edge.origin.ts, edge.origin.idx))
                    continue
                G.add_edge((edge.origin.ts, edge.origin.idx), (edge.dest.ts, edge.dest.idx), weight=edge.weight)


        layer_nodes = {}
        for node, data in G.nodes(data=True):
            layer = data['layer']
            layer_nodes.setdefault(layer, []).append(node)


        x_spacing, y_spacing = 200, 150
        pos = cls._compute_node_positions(layer_nodes, x_spacing, y_spacing)

        # Constant edge color and smaller width
        edge_weights = [d["weight"] for _, _, d in G.edges(data=True)]
        edge_color = 'black'
        edge_width = 1

        node_colors = [
            cls._get_node_color(node, origin_nodes, G)
            for node in G.nodes()
        ]

        nx.draw_networkx_nodes(G, pos, node_size=50, node_color=node_colors, ax=ax) # type: ignore
        nx.draw_networkx_edges(
            G, pos,
            edge_color=edge_color,
            width=edge_width,
            arrows=False,
            ax=ax
        )

        if ts_freq is not None:
            for ts in layer_nodes:
                if ts % ts_freq != 0:
                    continue
                nodes = layer_nodes[ts]
                xs = [pos[n][0] for n in nodes]
                ys = [pos[n][1] for n in nodes]
                x_center = sum(xs) / len(xs)
                y_min, y_max = min(ys), max(ys)

                # Draw vertical dotted line
                ax.plot(
                    [x_center, x_center], [y_min - y_spacing * 0.25, y_max + y_spacing * 0.25],
                    linestyle=':', color='gray', linewidth=1, zorder=0
                )

                # Add layer timestamp label
                ax.text(
                    x_center, y_max + y_spacing * 0.3,
                    f"ts={ts}",
                    ha='center', va='bottom',
                    fontsize=8, fontweight='bold', color='gray'
                )

        if legend:
            cls._graph_legend(ax)

        if title:
            ax.set_title(title if isinstance(title, str) else "Graph over time")

    ### TRACKING ###
    @classmethod
    @ensure_ax
    def draw_trail_2d(
        cls,
        graph: Graph,
        bc_start: Blob_collection,
        max_chains: int = -1,
        trail_centroid: bool = True,
        trail_kalman: bool = True,
        legend: bool = True,
        title: bool | str = True,
        ax: Axes | None = None,
    ) -> None:
        assert ax is not None

        # Per ogni blob in bc_start, ottieni la sua catena e tracciala
        for start_blob in bc_start:
            for chain in graph.get_superblob_chains(start_blob.idx, start_blob.ts, max_chains):
                chain_len = len(chain)
                if chain_len < 2: continue  #too shorrt

                prev_blob = None
                for i, blob in enumerate(chain):
                    if prev_blob is not None:
                        alpha = (i - 1) / (chain_len - 2) if chain_len > 2 else 1.0

                        # centroid trail
                        if trail_centroid:
                            x0, y0 = prev_blob.centroid[::-1]
                            x1, y1 = blob.centroid[::-1]
                            ax.plot([x0, x1], [y0, y1], color=cls.TRAIL_CENTROID, alpha=alpha, label='_nolegend_')

                        # kalman trail
                        if trail_kalman:
                            x0k, y0k = prev_blob.state[:2].flatten()
                            x1k, y1k = blob.state[:2].flatten()
                            ax.plot([x0k, x1k], [y0k, y1k], color=cls.TRAIL_KALMAN, alpha=alpha, label='_nolegend_')

                    prev_blob = blob

        if legend:
            legend_elements = [
                Patch(facecolor=cls.TRAIL_CENTROID, edgecolor='black', label="Centroid Trail") if trail_centroid else None,
                Patch(facecolor=cls.TRAIL_KALMAN, edgecolor='black', label="Kalman Trail") if trail_kalman else None,
            ]
            ax.legend(
                handles=[e for e in legend_elements if e],
                loc='upper right',
                frameon=True,
                facecolor='white',
                edgecolor='black'
            )

        if title:
            ax.set_title(title if isinstance(title, str) else "Blob Traces")
