from ..Detection.motion_detector import MotionDetector

from ..Matching.edge_collection import EdgesCollection
from ..Matching.edge import Edge,OriginEdge
from ..Matching.matcher import Matcher

from ..Splitting.blob_collection import Blob_collection
from ..Splitting.blob import Blob,VoidBlob
from ..Splitting.cc_splitter import CCSplitter


from ..frame_collection import FramesCollection
from ..Graph.graph import Graph
from ..misc import stringify_params

class TreeViewer:
    @classmethod
    def render(cls, obj, indent=0, max_depth=None, show_time=True):
        return cls._dispatch(obj, indent, max_depth, show_time)

    @classmethod
    def _dispatch(cls, obj, indent, max_depth, show_time):
        prefix = "\t" * indent
        if max_depth is not None and max_depth <= 0:
            return f"{prefix}{repr(obj)}"

        if isinstance(obj, FramesCollection):
            return cls._FramesCollection(obj, indent, max_depth, show_time)
        elif isinstance(obj, CCSplitter):
            return cls._ccsplitter(obj, indent, max_depth, show_time)
        elif isinstance(obj, Blob_collection):
            return cls._blob_collection(obj, indent, max_depth, show_time)
        elif isinstance(obj, Blob):
            return cls._blob(obj, indent, show_time)
        elif isinstance(obj, MotionDetector):
            return cls._motion(obj, indent, show_time)
        elif isinstance(obj, Edge):
            return cls._edge(obj, indent, show_time)
        elif isinstance(obj, Graph):
            return cls._graph(obj, indent, max_depth, show_time)
        elif isinstance(obj, EdgesCollection):
            return cls._edges_collection(obj, indent, max_depth, show_time)
        elif isinstance(obj, Matcher):
            return cls._matcher(obj, indent, max_depth, show_time)
        else:
            return f"{prefix}{repr(obj)}"

    @classmethod
    def _FramesCollection(cls, fc, indent, max_depth, show_time):
        prefix = "\t" * indent
        lines = [
            f"{prefix}FramesCollection(name={fc.name!r})",
            f"{prefix}\t- frames_remaining={len(fc.next_frames)}",
            f"{prefix}\t- current_bg_shape={getattr(fc.curr_bg, 'shape', 'N/A')}",
            f"{prefix}\t- detector_params={stringify_params(fc.detector_params)}",
            f"{prefix}\t- splitter_params={stringify_params(fc.splitter_params)}",
            f"{prefix}\t- current_timestamp={fc.get_timestamp()}",
            f"{prefix}\t- total_history={len(fc.history)}",
            f"{prefix}\t- History:"
        ]

        next_depth = None if max_depth is None else max_depth - 1
        #show history
        for h in fc.history: lines.append(cls._dispatch(h, indent + 2, next_depth, show_time))
        
        lines.append(cls._dispatch(fc.graph, indent + 1 , next_depth, show_time))

        return "\n".join(lines)


    @classmethod
    def _ccsplitter(cls, splitter, indent, max_depth, show_time):
        prefix = "\t" * indent
        ts_part = f"(ts={splitter.ts})" if show_time else ""
        next_indent = indent + 1
        next_depth = None if max_depth is None else max_depth - 1
        return (
            f"{prefix}CCSplitter{ts_part}\n"
            f"{cls._dispatch(splitter.detector, next_indent, next_depth, show_time)}\n"
            f"{cls._dispatch(splitter.blobs, next_indent, next_depth, show_time)}"
        )

    @classmethod
    def _blob_collection(cls, bc, indent, max_depth, show_time):
        prefix = "\t" * indent
        ts_part = f"ts={bc.ts}, " if show_time else ""
        next_indent = indent + 1
        next_depth = None if max_depth is None else max_depth - 1
        if max_depth is not None and max_depth <= 0:
            return f"{prefix}{repr(bc)}"
        sorted_blobs = sorted(bc, key=lambda b: b.idx)
        blob_lines = [cls._dispatch(b, next_indent, next_depth, show_time) for b in sorted_blobs]
        return f"{prefix}Blob_collection({ts_part}n_blobs={len(bc.blobs)})\n" + "\n".join(blob_lines) + "\n"

    @classmethod
    def _blob(cls, b, indent, show_time):
        prefix = "\t" * indent
        if isinstance(b,VoidBlob):return f"{prefix}{repr(b)}"

        ts_line = f"{prefix}Blob(ts={b.ts})\n" if show_time else f"{prefix}Blob\n"
        st = b.state.reshape(1, b.PHI.shape[0])
        return (
            ts_line +
            f"{prefix}\tidx={b.idx} centroid=({b.centroid[0]:.1f}, {b.centroid[1]:.1f} weights={b.weight}) \n"
            f"{prefix}\tstate=({st[0][1]:.1f}, {st[0][0]:.1f}) state_vel=({st[0][3]:.1f}, {st[0][2]:.1f})"
        )
    @classmethod
    def _motion(cls, m, indent, show_time):
        prefix = "\t" * indent
        return f"{prefix}MotionDetector" + (f"(ts={m.ts})" if show_time else "")
    
    @classmethod
    def _edge(cls, edge, indent, show_time):
        prefix = "\t" * indent
        if isinstance(edge,OriginEdge):return f"{prefix}{repr(edge)}"
        return f"{prefix}Edge({repr(edge.origin)} -> {repr(edge.dest)}, similarity={edge.weight:.2f})"
    
    @classmethod
    def _graph(cls, graph, indent, max_depth, show_time):
        prefix = "\t" * indent
        lines = [f"{prefix}Graph"]

        next_indent = indent + 2
        next_depth = None if max_depth is None else max_depth - 1

        # Layers
        lines.append(f"{prefix}\tLayers")
        for ts in sorted(graph.layers):
            bc = graph.layers[ts]
            lines.append(cls._dispatch(bc, next_indent, next_depth, show_time))

        # Edges
        lines.append(f"{prefix}\tEdges")
        for ts in sorted(graph.edges):
            ec = graph.edges[ts]
            lines.append(cls._dispatch(ec, next_indent, next_depth, show_time))

        return "\n".join(lines)
    
    @classmethod
    def _edges_collection(cls, ec, indent, max_depth, show_time):
        prefix = "\t" * indent
        ts_part = f"(ts={ec.ts})" if show_time else ""
        next_indent = indent + 1
        next_depth = None if max_depth is None else max_depth - 1

        lines = [f"{prefix}EdgesCollection{ts_part}, n_edges={len(ec.edges)}"]
        if max_depth is not None and max_depth <= 0:
            return lines[0]

        for edge in sorted(ec.edges, key=lambda e: (e.origin.idx, e.dest.idx)):
            lines.append(cls._dispatch(edge, next_indent, next_depth, show_time))

        return "\n".join(lines) +"\n"


    @classmethod
    def _matcher(cls, matcher, indent, max_depth, show_time):
        prefix = "\t" * indent
        lines = [f"{prefix}Matcher(ts={matcher.ts})"]

        # Show edges collection detail
        next_indent = indent + 1
        next_depth = None if max_depth is None else max_depth - 1
        lines.append(cls._dispatch(matcher.curr_CC, next_indent, next_depth, show_time))
        lines.append(cls._dispatch(matcher.edges, next_indent, next_depth, show_time))

        return "\n".join(lines)

