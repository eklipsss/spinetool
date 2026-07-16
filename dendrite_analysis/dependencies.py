from sklearn.cluster import DBSCAN
import trimesh
import pyvista as pv
from scipy.spatial.distance import pdist, squareform
from scipy.stats import f_oneway, ks_2samp, shapiro, mannwhitneyu, brunnermunzel, kruskal
from scipy.spatial.distance import cdist
import pandas as pd
import json
from esda.moran import Moran
from esda import G
from libpysal.weights import DistanceBand, W
import math
import numpy as np
from typing import Any, List, Tuple, Dict
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import seaborn as sns
import meshplot as mp
import networkx as nx
# import community as community_louvain

from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
from CGAL.CGAL_Polygon_mesh_processing import volume, does_self_intersect
from CGAL.CGAL_Kernel import Vector_3
from CGAL.CGAL_Polygon_mesh_processing import Polylines
from CGAL.CGAL_Surface_mesh_skeletonization import surface_mesh_skeletonization

from spine_segmentation import point_2_list
from spine_analysis.mesh.utils import MeshDataset, LineSet
from spine_analysis.shape_metric.float_metric import (
    ConvexHullRatioSpineMetric,
    ConvexHullVolumeSpineMetric,
    VolumeSpineMetric,
)
from spine_analysis.shape_metric.junction_metric import (
    AreaSpineMetric,
    AverageDistanceSpineMetric,
    CenterSpineMetric,
    CVDSpineMetric,
    JunctionAreaSpineMetric,
    JunctionCenterSpineMetric,
    LengthAreaRatioSpineMetric,
    LengthSpineMetric,
    LengthVolumeRatioSpineMetric,
    OpenAngleSpineMetric,
)
from spine_analysis.shape_metric.utils import _point_2_vec
