from .dependencies import *
from .config import *
from .metrics import *
from .spine import Spine
from .surface_distances import (
    calculate_mesh_graph_distance_matrix,
    calculate_spine_distance_matrices,
    centerline_length_from_mesh,
    polyhedron_to_trimesh,
    save_distance_method_comparison,
)
from spine_analysis.shape_metric.utils import get_dendrite_skeleton


def _fallback_dendrite_length_any_mesh(dendr_mesh: Any) -> float:
    length = centerline_length_from_mesh(dendr_mesh)
    print(f"  length (mesh-graph centerline fallback) = {length:.2f}")
    return length


def _polyline_length(points: Any) -> float:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 3:
        return 0.0
    points = points[:, :3]
    finite_mask = np.isfinite(points).all(axis=1)
    points = points[finite_mask]
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _skeleton_length_from_object(skeleton: Any) -> float:
    if skeleton is None:
        return 0.0

    if isinstance(skeleton, np.ndarray) and skeleton.dtype == object:
        if skeleton.shape == ():
            return _skeleton_length_from_object(skeleton.item())
        return float(sum(_skeleton_length_from_object(item) for item in skeleton.tolist()))

    if isinstance(skeleton, dict):
        for points_key in ("points", "vertices", "nodes"):
            if points_key in skeleton and "edges" in skeleton:
                points = np.asarray(skeleton[points_key], dtype=float)
                edges = np.asarray(skeleton["edges"], dtype=int)
                if points.ndim == 2 and points.shape[1] >= 3 and edges.ndim == 2 and edges.shape[1] >= 2:
                    valid_edges = edges[:, :2]
                    valid_edges = valid_edges[
                        (valid_edges >= 0).all(axis=1)
                        & (valid_edges < len(points)).all(axis=1)
                    ]
                    if len(valid_edges) > 0:
                        segments = points[valid_edges[:, 1], :3] - points[valid_edges[:, 0], :3]
                        return float(np.linalg.norm(segments, axis=1).sum())
        for key in ("skeleton", "branches", "polylines", "paths", "lines"):
            if key in skeleton:
                length = _skeleton_length_from_object(skeleton[key])
                if length > 0:
                    return length
        return float(sum(_skeleton_length_from_object(value) for value in skeleton.values()))

    if isinstance(skeleton, (list, tuple)):
        try:
            numeric = np.asarray(skeleton, dtype=float)
            if numeric.ndim >= 2:
                return _skeleton_length_from_object(numeric)
        except Exception:
            pass
        return float(sum(_skeleton_length_from_object(item) for item in skeleton))

    try:
        array = np.asarray(skeleton, dtype=float)
    except Exception:
        return 0.0

    if array.ndim == 2 and array.shape[1] >= 3:
        return _polyline_length(array)

    if array.ndim == 3 and array.shape[-1] >= 3:
        if array.shape[1] == 2:
            segments = array[:, 1, :3] - array[:, 0, :3]
            return float(np.linalg.norm(segments, axis=1).sum())
        return float(sum(_polyline_length(polyline) for polyline in array))

    return 0.0


def _registered_skeleton_length(dendr_mesh: Any) -> float:
    skeleton = get_dendrite_skeleton(dendr_mesh)
    length = _skeleton_length_from_object(skeleton)
    if np.isfinite(length) and length > 0:
        print(f"  dendr_len (registered branch_skeleton.npy) = {length:.2f}")
        return float(length)
    return 0.0


def _fallback_dendrite_volume_any_mesh(dendr_mesh: Any) -> float:
    tm = polyhedron_to_trimesh(dendr_mesh)
    raw_volume = getattr(tm, "volume", None)
    if raw_volume is not None and np.isfinite(raw_volume) and abs(float(raw_volume)) > 0:
        return abs(float(raw_volume))

    try:
        hull_volume = float(tm.convex_hull.volume)
        if np.isfinite(hull_volume) and hull_volume > 0:
            import warnings
            warnings.warn(
                "Dendrite volume is unavailable for this mesh; using convex-hull volume "
                "as an approximation.",
                stacklevel=2,
            )
            return hull_volume
    except Exception:
        pass

    vertices = np.asarray(tm.vertices, dtype=float)
    if len(vertices) < 3:
        return 0.0
    length = max(_fallback_dendrite_length_any_mesh(tm), 1e-12)
    center = vertices.mean(axis=0)
    _, _, vh = np.linalg.svd(vertices - center, full_matrices=False)
    axis = vh[0]
    axial = (vertices - center) @ axis
    projected = np.outer(axial, axis)
    radial_distances = np.linalg.norm((vertices - center) - projected, axis=1)
    radius = float(np.median(radial_distances))
    return float(math.pi * radius * radius * length)

class Dendrite:
    name: str
    mesh: Any
    spines: List[Spine] = []
    spine_meshes: MeshDataset

    volume: float
    length: float
    radius: float
    dr: float
    volume_around_dendr: float

    center_coords: List[float] = []
    center_coords_c: List[float] = []
    dists: List[float] = []  
    dists_c: List[float] = []  
    distance_matrix: Any
    distance_matrix_c: Any

    nndist: float
    nndist_norm: float
    nndist_c: float
    nndist_norm_c: float

    pair_distance_profile_r_values: List[float] = []
    pair_distance_profile_values: List[float] = []
    pair_distance_profile_entropy: float
    pair_distance_profile_r_values_c: List[float] = []
    pair_distance_profile_values_c: List[float] = []
    pair_distance_profile_entropy_c: float

    # Legacy global Volume Moran fields are kept commented out: the current
    # pipeline writes autocorr_<metric>_* records instead.
    # moran_I: float
    # moran_z: float
    # moran_p: float
    # moran_I_c: float
    # moran_z_c: float
    # moran_p_c: float

    moran_dict_c: Dict[str, float] = {}

    getis_ord_G: float
    getis_ord_z: float
    getis_ord_p: float
    getis_ord_G_c: float
    getis_ord_z_c: float
    getis_ord_p_c: float

    dbscan_labels: Any = []
    dbscan_eps: float = 0
    dbscan_min_samples: int = 0
    # Legacy statistic/p-value from the old DBSCAN parameter search.
    # dbscan_statistic: float = 0
    # dbscan_p_value: float = 0
    dbscan_noise: float = 0

    dbscan_labels_c: Any = []
    dbscan_eps_c: float = 0
    dbscan_min_samples_c: int = 0
    # Legacy cylindrical statistic/p-value from the old DBSCAN parameter search.
    # dbscan_statistic_c: float = 0
    # dbscan_p_value_c: float = 0
    dbscan_noise_c: float = 0

    db_count_class: Dict[str, List[int]]
    db_count_cluster: Dict[str, List[int]]
    db_count_class_c: Dict[str, List[int]]
    db_count_cluster_c: Dict[str, List[int]]

    db_matrix_class: np.ndarray = np.zeros((len(classes_list), len(classes_list)))
    db_matrix_class_c: np.ndarray = np.zeros((len(classes_list), len(classes_list)))
    db_matrix_cluster: np.ndarray = np.zeros((len(clusters_list), len(clusters_list)))
    db_matrix_cluster_c: np.ndarray = np.zeros((len(clusters_list), len(clusters_list)))

    n_count_class: Dict[str, List[int]]
    n_count_cluster: Dict[str, List[int]]
    n_count_class_c: Dict[str, List[int]]
    n_count_cluster_c: Dict[str, List[int]]

    n_matrix_class: np.ndarray = np.zeros((len(classes_list), len(classes_list)))    
    n_matrix_class_c: np.ndarray = np.zeros((len(classes_list), len(classes_list)))
    n_matrix_cluster: np.ndarray = np.zeros((len(clusters_list), len(clusters_list)))
    n_matrix_cluster_c: np.ndarray = np.zeros((len(clusters_list), len(clusters_list)))

    g_cluster_sizes: List[int]
    g_mean_cluster_size: float
    g_characteristic_extent: float
    g_average_clustering: float
    g_modularity: float

    n_count_c: int

    cylindr_flag: bool = True

    def __init__(self, dendr_name: str, dendrite_meshes: MeshDataset = None, spine_meshes: MeshDataset = None) -> None:
        print('Dendrite init')

        self.spines = []
        self.center_coords = []
        self.center_coords_c = []
        
        self.dists = []  
        self.dists_c = []  
        self.pair_distance_profile_r_values = []
        self.pair_distance_profile_values = []
        self.r_values = self.pair_distance_profile_r_values
        self.pcf_values = self.pair_distance_profile_values
        self.pair_distance_profile_r_values_c = []
        self.pair_distance_profile_values_c = []
        self.r_values_c = self.pair_distance_profile_r_values_c
        self.pcf_values_c = self.pair_distance_profile_values_c
        self.dbscan_labels = []
        self.dbscan_labels_c = []

        self.moran_dict_c = {}

        # self.n_count_class = {'Undefined' : [], 'Stubby' : [], 'Mushroom' : [],  'Thin' : [], 'Filopodia' : []}
        # self.n_count_cluster = { 0 : [], 1 : [], 2 : [], 3 : [], 4 : [], 5 : [], 6: [] } 
        # self.n_count_class_c = {'Undefined' : [], 'Stubby' : [], 'Mushroom' : [],  'Thin' : [], 'Filopodia' : []}
        # self.n_count_cluster_c = { 0 : [], 1 : [], 2 : [], 3 : [], 4 : [], 5 : [], 6: [] } 
        self.n_count_class = {'Stubby' : [], 'Mushroom' : [],  'Thin' : [], 'Filopodia' : []}
        self.n_count_cluster = {1 : [], 2 : [], 3 : [], 4 : [], 5 : [], 6: [] } 
        self.n_count_class_c = {'Stubby' : [], 'Mushroom' : [],  'Thin' : [], 'Filopodia' : []}
        self.n_count_cluster_c = {1 : [], 2 : [], 3 : [], 4 : [], 5 : [], 6: [] } 

        self.n_matrix_class = np.zeros((len(classes_list), len(classes_list)))    
        self.n_matrix_class_c = np.zeros((len(classes_list), len(classes_list))) 
        self.n_matrix_cluster = np.zeros((len(clusters_list), len(clusters_list))) 
        self.n_matrix_cluster_c = np.zeros((len(clusters_list), len(clusters_list))) 

        self.g_cluster_sizes = []
        self.g_mean_cluster_size = 0
        self.g_characteristic_extent = 0
        self.g_average_clustering = 0
        self.g_modularity = 0

        self.name = dendr_name

        if dendrite_meshes is not None:
            self.calculate_init_metrics(dendrite_meshes) # вычисление метрик 
        else:
            self.load_init_metrics() # загрузка метрик из файла

        self.center_coords: List[float] = []
        self.center_coords_c: List[float] = []

        if spine_meshes is not None:
            self.spine_meshes = spine_meshes
            self.calculate_and_create_spines()
        else:
            self.load_and_create_spines()

        self.save_spine_coords() # сохранение координат шипиков
        self.save_spine_metrics() # сохранение метрик
        
        # self.distance_matrix = squareform(pdist(self.center_coords))
        # self.distance_matrix_c = squareform(pdist(self.center_coords_c, lambda u, v: cylindrical_distance(u, v)))

        if dendrite_meshes is not None:
            spines_len = [spine.metrics['Length'] for spine in self.spines]
            self.dr = max(spines_len)/2 if spines_len else 0
            self.volume_around_dendr = math.pi*self.length*((self.radius + self.dr)**2 - self.radius**2)

        # if self.cylindr_flag:
        #     self.dists_c = calculate_all_dists(self.center_coords_c, 1)
        #     self.dists_c = [d/self.length for d in self.dists_c]

    def load_init_metrics(self) -> None:
        # with open('metrics/9009/dendr_metrics.json', 'r') as f:
        # with open('metrics/wt_old_st/dendr_metrics.json', 'r') as f:
        with open('input/dendr_metrics.json', 'r') as f:
            loaded_dict = json.load(f)

        self.volume = loaded_dict[self.name]['Volume']
        self.length = loaded_dict[self.name]['Length']
        self.radius = loaded_dict[self.name]['Radius']
        self.dr = loaded_dict[self.name]['Dr']
        self.volume_around_dendr = loaded_dict[self.name]['Volume_around_dendr']

    def calculate_init_metrics(self, dendrite_meshes: MeshDataset = None) -> None:
        for (dendr_name, dendr_mesh) in  dendrite_meshes.items():
            self.name = dendr_name
            print(f'Dendrite {dendr_name}')
            self.mesh = dendr_mesh
            try:
                if isinstance(dendr_mesh, Polyhedron_3) and get_dendrite_skeleton(dendr_mesh) is None:
                    raw_vol = volume(dendr_mesh)
                else:
                    raw_vol = _fallback_dendrite_volume_any_mesh(dendr_mesh)
            except Exception as exc:
                import warnings
                warnings.warn(
                    f"CGAL/trimesh volume calculation unavailable for '{dendr_name}' "
                    f"({exc}); using fallback volume approximation.",
                    stacklevel=2,
                )
                raw_vol = _fallback_dendrite_volume_any_mesh(dendr_mesh)
            if raw_vol < 0:
                import warnings
                warnings.warn(
                    f"Volume for '{dendr_name}' is negative ({raw_vol:.2f}) — mesh may not be "
                    "closed or face normals may be inverted. Using abs(volume) as approximation.",
                    stacklevel=2,
                )
            self.volume = abs(raw_vol)
            print(f'  dendr_volume = {self.volume:.2f}')

            registered_length = _registered_skeleton_length(dendr_mesh)
            if registered_length > 0:
                self.length = registered_length
            else:
                try:
                    if not isinstance(dendr_mesh, Polyhedron_3):
                        raise TypeError(
                            "CGAL skeletonization requires Polyhedron_3; "
                            f"got {type(dendr_mesh).__name__}"
                        )
                    sceleton_vecs, skeleton_line_set = get_sceleton_vecs(dendr_mesh)  # векторы скелета дендрита (без шипиков)
                    self.length = calculate_LengthDendriteMetric(sceleton_vecs, skeleton_line_set)
                except Exception as exc:
                    import warnings
                    warnings.warn(
                        f"Skeleton-based length calculation unavailable for '{dendr_name}' "
                        f"({exc}); falling back to a PCA-based length approximation.",
                        stacklevel=2,
                    )
                    self.length = _fallback_dendrite_length_any_mesh(dendr_mesh)
            if self.length <= 0:
                import warnings
                warnings.warn(
                    f"Length for '{dendr_name}' is non-positive ({self.length}); "
                    "using 1.0 to avoid division by zero in radius calculation.",
                    stacklevel=2,
                )
                self.length = 1.0
            self.radius = math.sqrt(self.volume/(math.pi*self.length))
        # print(f'  dendr_len = {self.length:.2f}')
        # print(f'  dendr_radius = {self.radius:.2f}')

    def calculate_and_create_spines(self) -> None:
        junction_center_klass = spine_metric_classes['JunctionCenterSpineMetric']
        center_klass = spine_metric_classes['CenterSpineMetric']

        junction_center_vecs = []
        junction_center_coords = [] 

        for (spine_name, spine_mesh) in self.spine_meshes.items():
            print(f"  junction/center metrics: {spine_name}", flush=True)
            # середина области крепления шипика
            junction_center_vec = junction_center_klass(spine_mesh)._value
            junction_center_coord = (junction_center_vec.x(), junction_center_vec.y(), junction_center_vec.z())  # tuple, тк неизменяемый
            junction_center_coords.append(junction_center_coord)
            junction_center_vecs.append(junction_center_vec)
            
            # середина шипика
            center_vec = center_klass(spine_mesh)._value
            center_coord = (center_vec.x(), center_vec.y(), center_vec.z())  # tuple, тк неизменяемый
            self.center_coords.append(center_coord)
        
        spine_meshes_list = list(self.spine_meshes.items())

        save_coords[self.name] = {}

        if len(junction_center_vecs) >= 3:
            two_dim_points = get_2dim_coordinates(junction_center_vecs)
            self.center_coords_c = get_cylindr_coord_from_2dim(two_dim_points, self.radius)

            for i in range(len(self.spine_meshes)):
                save_coords[self.name][spine_meshes_list[i][0]] = {'junction_center_coord': junction_center_coords[i], 'center_coord': self.center_coords[i],'center_coord_c': self.center_coords_c[i] }

                self.spines.append(Spine(spine_meshes_list[i][0], junction_center_coords[i], self.center_coords[i], self.center_coords_c[i], spine_meshes_list[i][1]))
                
        else:
            print('else ')
            self.cylindr_flag = False
            for i in range(len(self.spine_meshes)):
                save_coords[self.name][spine_meshes_list[i][0]] = {'junction_center_coord': junction_center_coords[i], 'center_coord': self.center_coords[i],'center_coord_c': False }

                self.spines.append(Spine(spine_meshes_list[i][0], junction_center_coords[i], self.center_coords[i], False, spine_meshes_list[i][1]))

    def load_and_create_spines(self) -> None:
        junction_center_coords = [] 

        # with open('metrics/9009/spine_coords.json', 'r') as f:
        # with open('metrics/wt_old_st/spine_coords.json', 'r') as f:
        with open('input/spine_coords.json', 'r') as f:
            spine_coords_dict = json.load(f)

        for i, spine_name in enumerate(spine_coords_dict[self.name].keys()):
            self.center_coords.append(spine_coords_dict[self.name][spine_name]['center_coord'])
            self.center_coords_c.append(spine_coords_dict[self.name][spine_name]['center_coord_c'])
            if self.center_coords_c[i] == False:
                self.cylindr_flag = False
            junction_center_coords.append(spine_coords_dict[self.name][spine_name]['junction_center_coord'])
            self.spines.append(Spine(spine_name, junction_center_coords[i], self.center_coords[i], self.center_coords_c[i]))

    def add_spine_class(self) -> None:
        for s in self.spines:
            s.add_spine_class()

    def add_spine_cluster(self) -> None:
        for s in self.spines:
            s.add_spine_cluster()

    def load_grouping_metrics(self) -> None:
        metric_dict_for_autocorr = {
            "Volume": [spine.metrics['Volume'] for spine in self.spines],
        }
                
        # with open('metrics/grouping_dendr_metrics.json', 'r') as f:
        with open('input/grouping_dendr_metrics.json', 'r') as f:
            loaded_dict = json.load(f)

        self.nndist = loaded_dict[self.name]['NNdist']
        self.nndist_norm = loaded_dict[self.name]['NNdist_norm']
        self.pair_distance_profile_r_values = loaded_dict[self.name].get(
            'PairDistanceProfile_r_values',
            loaded_dict[self.name].get('r_values', []),
        )
        self.pair_distance_profile_values = loaded_dict[self.name].get(
            'PairDistanceProfile_values',
            loaded_dict[self.name].get('PCF_values', []),
        )
        self.pair_distance_profile_entropy = loaded_dict[self.name].get(
            'PairDistanceProfile_entropy',
            loaded_dict[self.name].get('Entropy', 0.0),
        )
        self.r_values = self.pair_distance_profile_r_values
        self.pcf_values = self.pair_distance_profile_values
        self.entropy = self.pair_distance_profile_entropy

        # Legacy global Volume Moran fields are no longer loaded by the current
        # pipeline. Use spatial_morphology_permutation.csv instead.
        # self.moran_I = loaded_dict[self.name]['Moran_I']
        # self.moran_z = loaded_dict[self.name]['Moran_zI']
        # self.moran_p = loaded_dict[self.name]['Moran_p']

        # self.getis_ord_G = loaded_dict[self.name]['Getis_Ord_G']
        # self.getis_ord_z = loaded_dict[self.name]['Getis_Ord_zG']
        # self.getis_ord_p = loaded_dict[self.name]['Getis_Ord_p']

        # self.getis_ord_G, self.getis_ord_z, self.getis_ord_p = calculate_Getis_Ord_G(self.center_coords, metric_dict_for_autocorr, False, 7, self.name)['Volume']


        if self.cylindr_flag:
            # Старые файлы могли хранить отдельные cylindrical-метрики с суффиксом _c.
            # В новых файлах они не сохраняются: mesh_graph-метрики лежат в обычных полях.
            self.nndist_c = loaded_dict[self.name].get('NNdist_c', self.nndist)
            self.nndist_norm_c = loaded_dict[self.name].get('NNdist_norm_c', self.nndist_norm)
            self.pair_distance_profile_r_values_c = loaded_dict[self.name].get(
                'PairDistanceProfile_r_values_c',
                loaded_dict[self.name].get('r_values_c', self.pair_distance_profile_r_values),
            )
            self.pair_distance_profile_values_c = loaded_dict[self.name].get(
                'PairDistanceProfile_values_c',
                loaded_dict[self.name].get('PCF_values_c', self.pair_distance_profile_values),
            )
            self.pair_distance_profile_entropy_c = loaded_dict[self.name].get(
                'PairDistanceProfile_entropy_c',
                loaded_dict[self.name].get('Entropy_c', self.pair_distance_profile_entropy),
            )
            self.r_values_c = self.pair_distance_profile_r_values_c
            self.pcf_values_c = self.pair_distance_profile_values_c
            self.entropy_c = self.pair_distance_profile_entropy_c

            # self.moran_I_c = loaded_dict[self.name].get('Moran_I_c', self.moran_I)
            # self.moran_z_c = loaded_dict[self.name].get('Moran_zI_c', self.moran_z)
            # self.moran_p_c = loaded_dict[self.name].get('Moran_p_c', self.moran_p)

            # self.getis_ord_G_c = loaded_dict[self.name]['Getis_Ord_G_c']
            # self.getis_ord_z_c = loaded_dict[self.name]['Getis_Ord_zG_c']
            # self.getis_ord_p_c = loaded_dict[self.name]['Getis_Ord_p_c']

            # self.getis_ord_G_c, self.getis_ord_z_c, self.getis_ord_p_c = calculate_Getis_Ord_G(self.center_coords_c, metric_dict_for_autocorr, True, 7, self.name)['Volume']

    def load_cluster_metrics(self) -> None:
        with open('input/cluster_dendr_metrics.json', 'r') as f:
            loaded_dict = json.load(f)

        self.dbscan_eps = loaded_dict[self.name]['DBscan_eps']
        self.dbscan_min_samples = loaded_dict[self.name]['DBscan_min_samples']
        # Legacy statistic/p-value from the old DBSCAN parameter search are no
        # longer loaded by the current pipeline.
        # self.dbscan_statistic = loaded_dict[self.name]['DBscan_statistic']
        # self.dbscan_p_value = loaded_dict[self.name]['DBscan_p_value']
        self.dbscan_noise = loaded_dict[self.name]['DBscan_noise']

        if self.cylindr_flag:
            # Старые файлы могли хранить отдельные cylindrical-метрики с суффиксом _c.
            # В новых файлах они не сохраняются: mesh_graph-метрики лежат в обычных полях.
            self.dbscan_eps_c = loaded_dict[self.name].get('DBscan_eps_c', self.dbscan_eps)
            self.dbscan_min_samples_c = loaded_dict[self.name].get('DBscan_min_samples_c', self.dbscan_min_samples)
            # self.dbscan_statistic_c = loaded_dict[self.name].get('DBscan_statistic_c', self.dbscan_statistic)
            # self.dbscan_p_value_c = loaded_dict[self.name].get('DBscan_p_value_c', self.dbscan_p_value)
            self.dbscan_noise_c = loaded_dict[self.name].get('DBscan_noise_c', self.dbscan_noise)

    def get_spine_distance_points(self) -> List[Tuple[float, float, float]]:
        return [s.junction_center_coord for s in self.spines]

    def get_mesh_graph_distance_matrix(self) -> np.ndarray:
        if hasattr(self, "mesh_graph_distance_matrix"):
            return self.mesh_graph_distance_matrix
        if not hasattr(self, "mesh"):
            raise ValueError(
                "Mesh-graph distances require Dendrite.mesh. "
                "Create Dendrite with dendrite_meshes."
            )
        points = self.get_spine_distance_points()
        if len(points) == 0:
            self.mesh_graph_distance_matrix = np.zeros((0, 0), dtype=float)
            return self.mesh_graph_distance_matrix
        result = calculate_mesh_graph_distance_matrix(self.mesh, points)
        self.mesh_graph_distance_matrix = np.asarray(result.distance_matrix, dtype=float)
        return self.mesh_graph_distance_matrix

    def _calculate_pair_distance_profile_metrics(self) -> None:
        mesh_graph_distance_matrix = self.get_mesh_graph_distance_matrix()
        distance_points = self.get_spine_distance_points()
        self.nndist, self.nndist_norm = calculate_NNDist(
            distance_points,
            self.volume_around_dendr,
            0,
            distance_matrix=mesh_graph_distance_matrix,
        )
        self.pair_distance_profile_r_values, self.pair_distance_profile_values = calculate_pair_distance_profile(
            distance_points,
            self.volume_around_dendr,
            self.radius,
            self.dr,
            self.length,
            False,
            distance_matrix=mesh_graph_distance_matrix,
        )
        self.pair_distance_profile_entropy = calculate_Shannon_entropy(self.pair_distance_profile_values)
        # Deprecated aliases kept so older plotting/comparison code keeps working.
        self.r_values = self.pair_distance_profile_r_values
        self.pcf_values = self.pair_distance_profile_values
        self.entropy = self.pair_distance_profile_entropy

        if self.cylindr_flag:
            # Backwards-compatible *_c fields now mirror mesh_graph-based metrics.
            self.nndist_c, self.nndist_norm_c = self.nndist, self.nndist_norm
            self.pair_distance_profile_r_values_c = self.pair_distance_profile_r_values
            self.pair_distance_profile_values_c = self.pair_distance_profile_values
            self.pair_distance_profile_entropy_c = self.pair_distance_profile_entropy
            self.r_values_c = self.pair_distance_profile_r_values_c
            self.pcf_values_c = self.pair_distance_profile_values_c
            self.entropy_c = self.pair_distance_profile_entropy_c

    def _calculate_volume_moran_metric(self) -> None:
        # Legacy no-op. The current pipeline computes Moran's I for every
        # available morphology metric in calculate_spatial_autocorrelation_analysis()
        # and stores it as autocorr_<metric>_moran_i.
        return

    def calculate_grouping_metrics(self) -> None:
        self.calculate_density_distribution_analysis()
        self.calculate_spatial_autocorrelation_analysis(permutation_count=0)
        
    def calculate_cluster_metrics(self) -> None:
        spine_metrics_dict_for_dbscan = {}
        for s in self.spines:
            spine_metrics_dict_for_dbscan[(s.center_coord[0], s.center_coord[1], s.center_coord[2])] = { 'Volume' : s.metrics['Volume'] }
        points = [[s.center_coord[0], s.center_coord[1], s.center_coord[2]] for s in self.spines]
        mesh_graph_distance_matrix = self.get_mesh_graph_distance_matrix()

        dbscan_labels, dbscan_eps, dbscan_min_samples, _legacy_statistic, _legacy_p_value = dbscan(
            points,
            spine_metrics_dict_for_dbscan,
            0,
            'graphics/dbscan/' + self.name,
            distance_matrix=mesh_graph_distance_matrix,
            distance_label="mesh_graph",
        )
        self.dbscan_labels = dbscan_labels
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples

        points = np.array([s.center_coord for s in self.spines])

        class_member_mask = (self.dbscan_labels == -1)
        filtered_points = points[class_member_mask]
        self.dbscan_noise = len(filtered_points)/len(points)

        if self.cylindr_flag:
            # Backwards-compatible *_c fields now mirror mesh_graph-based DBSCAN.
            self.dbscan_labels_c = self.dbscan_labels
            self.dbscan_eps_c = self.dbscan_eps
            self.dbscan_min_samples_c = self.dbscan_min_samples
            # Legacy statistic/p-value from the old DBSCAN parameter search are
            # intentionally not updated in the current k-distance pipeline.
            # self.dbscan_statistic_c = self.dbscan_statistic
            # self.dbscan_p_value_c = self.dbscan_p_value
            self.dbscan_noise_c = self.dbscan_noise

    def cluster_analysis(self) -> None:
        if self.dbscan_labels.size != 0 :
            spine_vec_w_name = {}
            for s in self.spines:
                key = (s.center_coord[0], s.center_coord[1], s.center_coord[2])
                spine_vec_w_name[key] = s.name

            points = np.array([s.center_coord for s in self.spines])

            self.db_count_class, self.db_matrix_class = class_analysis(points, self.dbscan_labels, spine_vec_w_name)
            self.db_count_cluster, self.db_matrix_cluster = cluster_analysis(points, self.dbscan_labels, spine_vec_w_name)

            if self.cylindr_flag:
                spine_vec_w_name_c = {}
                for s in self.spines:
                    key = (s.center_coord_c[0], s.center_coord_c[1], s.center_coord_c[2])
                    spine_vec_w_name_c[key] = s.name

                points_c = np.array([s.center_coord_c for s in self.spines])

                self.db_count_class_c, self.db_matrix_class_c = class_analysis(points_c, self.dbscan_labels_c, spine_vec_w_name_c)
                self.db_count_cluster_c, self.db_matrix_cluster_c = cluster_analysis(points_c, self.dbscan_labels_c, spine_vec_w_name_c)

    def neighborhood_analysis(self, input_eps: float = 0) -> None: 
        counts_c = []

        for i, spine in enumerate(self.spines):
            if input_eps == 0:
                if self.dbscan_eps != 0:
                    eps = self.dbscan_eps
                else:
                    eps = 2.5
            else:
                eps = input_eps

            distances = self.get_mesh_graph_distance_matrix()[i]
            neighbors_ind = np.where(distances <= eps)[0]
            neighbors_spines = [self.spines[i] for i in neighbors_ind.tolist()]
            
            print(f"Точка {i} ({spine.name} - ик {spine.center_coord}): соседи {neighbors_ind}")

            for type in self.n_count_class.keys():
                self.n_count_class[type].append(0)
            for type in self.n_count_cluster.keys():
                self.n_count_cluster[type].append(0)

            if self.cylindr_flag:
                if input_eps == 0:
                    if self.dbscan_eps_c != 0:
                        eps = self.dbscan_eps_c
                    else:
                        eps = 2.5

                neighbors_ind_c = np.where(distances <= eps)[0].tolist()
                neighbors_spines_c = [self.spines[i] for i in neighbors_ind_c]
                counts_c.append(len(neighbors_spines_c))
                print(f"Точка {i} ({spine.name} - цк {spine.center_coord_c}): соседи {neighbors_spines_c}")

                for type in self.n_count_class.keys():
                    self.n_count_class_c[type].append(0)

                for type in self.n_count_cluster.keys():
                    self.n_count_cluster_c[type].append(0)
                
                self.n_count_class_c, self.n_count_cluster_c = self.count_by_type(i, self.n_count_class_c, self.n_count_cluster_c,  neighbors_spines_c)
                self.n_matrix_class_c, self.n_matrix_cluster_c = self.corr_matrix(self.n_matrix_class_c, self.n_matrix_cluster_c,  neighbors_spines_c)

            self.n_count_class, self.n_count_cluster = self.count_by_type(i, self.n_count_class, self.n_count_cluster, neighbors_spines)
            self.n_matrix_class, self.n_matrix_cluster = self.corr_matrix(self.n_matrix_class, self.n_matrix_cluster, neighbors_spines)

            self.n_count_c = np.mean(counts_c)

    def count_by_type(self, i: int, count_class: Dict[str, List[int]], count_cluster: Dict[int, List[int]], neighbors: List[Spine]):        
        for n_spine in neighbors:
            if n_spine.class_type != 'Undefined':
                count_class[n_spine.class_type][i] += 1
            if n_spine.cluster_type != 0:
                count_cluster[n_spine.cluster_type][i] += 1
        return count_class, count_cluster

    def corr_matrix(self, matrix_class, matrix_cluster, neighbors: List[Spine]):  
        class_type_w_index = {'Undefined' : 0, 'Stubby' : 1, 'Mushroom' : 2, 'Thin' : 3, 'Filopodia' : 4}
        # class_type_w_index = {'Stubby' : 1, 'Mushroom' : 2, 'Thin' : 3, 'Filopodia' : 4}

        class_types = np.array([class_type_w_index[s.class_type] for s in neighbors])   
        cluster_types = np.array([s.cluster_type for s in neighbors])    

        for i, type in enumerate(set(class_types)):
            for j, n_type in enumerate(set(class_types)): 
                if type == 0 or n_type == 0:
                    continue
                if i == j and np.count_nonzero(class_types == type) >= 2:
                    matrix_class[type-1, type-1] += 1
                else:
                    matrix_class[type-1, n_type-1] += 1

        for i, type in enumerate(set(cluster_types)):
            for j, n_type in enumerate(set(cluster_types)): 
                if type == 0 or n_type == 0:
                    continue
                if i == j and np.count_nonzero(cluster_types == type) >= 2:
                    matrix_cluster[type-1, type-1] += 1
                else:
                    matrix_cluster[type-1, n_type-1] += 1

        return matrix_class, matrix_cluster

    def graph_analysis(self) -> None:
        # Graph analysis now uses mesh_graph distances along the dendrite mesh.
        distances = self.get_mesh_graph_distance_matrix()

        # Создаем граф
        G = nx.Graph()
        n = distances.shape[0]
        G.add_nodes_from(range(n))

        # Добавляем ребра с весами, обратными mesh_graph-расстоянию по сетке дендрита.
        for i in range(n):
            for j in range(i + 1, n):
                d = distances[i][j]
                if np.isfinite(d) and d > 0:
                    w = 1 / d
                    G.add_edge(i, j, weight=w)

        if G.number_of_edges() == 0:
            self.g_average_clustering = 0
            self.g_cluster_sizes = []
            self.g_mean_cluster_size = 0
            self.g_characteristic_extent = 0
            self.g_modularity = 0
            return

        # Вычисляем средний коэффициент группировки
        self.g_average_clustering = nx.average_clustering(G, weight='weight')

        # Находим сообщества. Если установлен python-louvain, используем Louvain;
        # иначе fallback на greedy modularity из networkx.
        if hasattr(community_louvain, "best_partition"):
            partition = community_louvain.best_partition(G, weight='weight')
            modularity_partition = None
        else:
            community_sets = list(nx.algorithms.community.greedy_modularity_communities(G, weight='weight'))
            partition = {}
            for comm_id, nodes in enumerate(community_sets):
                for node in nodes:
                    partition[node] = comm_id
            modularity_partition = community_sets

        communities = {}
        for node, comm_id in partition.items():
            if comm_id not in communities:
                communities[comm_id] = []
            communities[comm_id].append(node)

        # Количество точек в каждом кластере
        self.g_cluster_sizes = [len(comm) for comm in communities.values()]
        self.g_mean_cluster_size = np.mean(self.g_cluster_sizes)

        # Характерная протяженность кластера (среднее mesh_graph-расстояние между точками внутри кластера)
        self.g_characteristic_extent = 0
        valid_community_count = 0
        for comm_id, nodes in communities.items():
            if len(nodes) < 2:
                continue
            total_distance = 0
            count = 0
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    d = distances[nodes[i]][nodes[j]]
                    if np.isfinite(d):
                        total_distance += d
                        count += 1
            if count > 0:
                self.g_characteristic_extent += total_distance / count
                valid_community_count += 1
        if valid_community_count > 0:
            self.g_characteristic_extent /= valid_community_count

        # Модульность разбиения
        if hasattr(community_louvain, "modularity"):
            self.g_modularity = community_louvain.modularity(partition, G, weight='weight')
        else:
            if modularity_partition is None:
                modularity_partition = [set(nodes) for nodes in communities.values()]
            self.g_modularity = nx.algorithms.community.modularity(
                G,
                modularity_partition,
                weight='weight',
            )

    @staticmethod
    def _cliffs_delta(values_a, values_b) -> float:
        a = np.asarray(values_a, dtype=float)
        b = np.asarray(values_b, dtype=float)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) == 0 or len(b) == 0:
            return float("nan")
        greater = 0
        less = 0
        for value in a:
            greater += int(np.sum(value > b))
            less += int(np.sum(value < b))
        return float((greater - less) / (len(a) * len(b)))

    @staticmethod
    def _safe_stat_test(test_fn, values_a, values_b):
        a = np.asarray(values_a, dtype=float)
        b = np.asarray(values_b, dtype=float)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) < 2 or len(b) < 2:
            return float("nan"), float("nan")
        try:
            result = test_fn(a, b)
            return float(result.statistic), float(result.pvalue)
        except Exception:
            return float("nan"), float("nan")

    @staticmethod
    def _safe_spearman(values_a, values_b):
        try:
            from scipy.stats import spearmanr
            a = np.asarray(values_a, dtype=float)
            b = np.asarray(values_b, dtype=float)
            mask = np.isfinite(a) & np.isfinite(b)
            if np.count_nonzero(mask) < 3:
                return float("nan"), float("nan")
            result = spearmanr(a[mask], b[mask])
            return float(result.statistic), float(result.pvalue)
        except Exception:
            return float("nan"), float("nan")

    @staticmethod
    def _moran_i_from_weights(values: np.ndarray, weights: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        mask = np.isfinite(values)
        if np.count_nonzero(mask) < 3:
            return float("nan")
        w = np.asarray(weights, dtype=float).copy()
        w[~np.isfinite(w)] = 0
        w[~mask, :] = 0
        w[:, ~mask] = 0
        w_sum = float(w.sum())
        if w_sum <= 0:
            return float("nan")
        x = values - np.nanmean(values)
        x[~mask] = 0
        denom = float(np.sum(x[mask] ** 2))
        if denom <= 0:
            return float("nan")
        n = int(np.count_nonzero(mask))
        return float((n / w_sum) * ((w * np.outer(x, x)).sum() / denom))

    @staticmethod
    def _geary_c_from_weights(values: np.ndarray, weights: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        mask = np.isfinite(values)
        if np.count_nonzero(mask) < 3:
            return float("nan")
        w = np.asarray(weights, dtype=float).copy()
        w[~np.isfinite(w)] = 0
        w[~mask, :] = 0
        w[:, ~mask] = 0
        w_sum = float(w.sum())
        if w_sum <= 0:
            return float("nan")
        x = values.copy()
        x[~mask] = np.nanmean(values)
        denom = float(np.nansum((values - np.nanmean(values)) ** 2))
        if denom <= 0:
            return float("nan")
        n = int(np.count_nonzero(mask))
        diff2 = (x[:, None] - x[None, :]) ** 2
        return float(((n - 1) / (2 * w_sum)) * np.nansum(w * diff2) / denom)

    @staticmethod
    def _permutation_p_value(observed: float, permuted: List[float]) -> float:
        permuted = np.asarray(permuted, dtype=float)
        permuted = permuted[np.isfinite(permuted)]
        if not np.isfinite(observed) or len(permuted) == 0:
            return float("nan")
        return float((np.sum(np.abs(permuted) >= abs(observed)) + 1) / (len(permuted) + 1))

    def _default_spatial_radius(self, distances: np.ndarray, min_samples: int = 3, percentile: float = 75) -> float:
        n = distances.shape[0]
        if n < 2:
            return 0.0
        rank = min(min_samples, n - 1)
        clean = np.asarray(distances, dtype=float).copy()
        clean[~np.isfinite(clean)] = np.inf
        np.fill_diagonal(clean, 0.0)
        kth = np.sort(clean, axis=1)[:, rank]
        kth = kth[np.isfinite(kth) & (kth > 0)]
        if len(kth) == 0:
            pair_distances = clean[np.triu_indices_from(clean, k=1)]
            pair_distances = pair_distances[np.isfinite(pair_distances) & (pair_distances > 0)]
            return float(np.percentile(pair_distances, percentile)) if len(pair_distances) else 0.0
        return float(np.percentile(kth, percentile))

    def _set_empty_spatial_analysis_results(self) -> None:
        self.spatial_morphology_summary = {"dendrite": self.name, "n_spines": 0}
        self.spatial_morphology_spine_records = []
        self.spatial_morphology_cluster_records = []
        self.spatial_morphology_test_records = []
        self.spatial_morphology_permutation_records = []
        self.local_neighborhood_summary = {
            "dendrite": self.name,
            "n_spines": 0,
            "local_radius": np.nan,
            "close_pair_fraction": np.nan,
        }
        self.local_neighborhood_spine_records = []

    def _build_spatial_analysis_context(self, local_radius: float = None) -> Dict[str, Any]:
        distances = np.asarray(self.get_mesh_graph_distance_matrix(), dtype=float)
        n = len(self.spines)

        if local_radius is None:
            local_radius = float(getattr(self, "dbscan_eps", 0) or 0)
        if local_radius <= 0 or not np.isfinite(local_radius):
            local_radius = self._default_spatial_radius(distances, min_samples=3, percentile=75)

        labels = np.asarray(getattr(self, "dbscan_labels", []))
        if labels.shape[0] != n:
            labels = np.full(n, -1, dtype=int)

        clean = distances.copy()
        clean[~np.isfinite(clean)] = np.inf
        np.fill_diagonal(clean, 0.0)
        sorted_distances = np.sort(clean, axis=1)

        def kth_distance(rank: int) -> np.ndarray:
            if n <= 1:
                return np.full(n, np.nan)
            index = min(rank, n - 1)
            values = sorted_distances[:, index]
            values[~np.isfinite(values)] = np.nan
            return values

        nn_distance = kth_distance(1)
        second_distance = kth_distance(2)
        third_distance = kth_distance(3)
        fifth_distance = kth_distance(5)

        adjacency = (distances <= local_radius) & np.isfinite(distances) & (distances > 0)
        neighbor_counts = adjacency.sum(axis=1).astype(int)

        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if adjacency[i, j]:
                    graph.add_edge(i, j, weight=float(distances[i, j]))
        degree = np.array([graph.degree(i) for i in range(n)], dtype=float)
        local_clustering_dict = nx.clustering(graph) if graph.number_of_edges() else {i: 0 for i in range(n)}
        local_clustering = np.array([local_clustering_dict.get(i, 0) for i in range(n)], dtype=float)
        component_sizes = []
        if graph.number_of_nodes():
            component_sizes = [len(component) for component in nx.connected_components(graph)]

        morphology_metrics = sorted({
            metric_name
            for spine in self.spines
            for metric_name, value in spine.metrics.items()
            if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value))
        })
        preferred = [metric for metric in ("Volume", "Length") if metric in morphology_metrics]
        morphology_metrics = preferred + [metric for metric in morphology_metrics if metric not in preferred]

        metric_values = {
            metric: np.asarray([float(spine.metrics.get(metric, np.nan)) for spine in self.spines], dtype=float)
            for metric in morphology_metrics
        }

        pair_distances = distances[np.triu_indices_from(distances, k=1)] if distances.size else np.array([])
        pair_distances = pair_distances[np.isfinite(pair_distances) & (pair_distances > 0)]

        return {
            "distances": distances,
            "n": n,
            "local_radius": local_radius,
            "labels": labels,
            "clean": clean,
            "nn_distance": nn_distance,
            "second_distance": second_distance,
            "third_distance": third_distance,
            "fifth_distance": fifth_distance,
            "adjacency": adjacency,
            "neighbor_counts": neighbor_counts,
            "graph": graph,
            "degree": degree,
            "local_clustering": local_clustering,
            "component_sizes": component_sizes,
            "morphology_metrics": morphology_metrics,
            "metric_values": metric_values,
            "pair_distances": pair_distances,
        }

    def calculate_density_distribution_analysis(self, local_radius: float = None) -> None:
        context = self._build_spatial_analysis_context(local_radius=local_radius)
        n = context["n"]
        if n == 0:
            self._set_empty_spatial_analysis_results()
            self._spatial_analysis_context = context
            return
        self._calculate_pair_distance_profile_metrics()

        labels = context["labels"]
        nn_distance = context["nn_distance"]
        second_distance = context["second_distance"]
        third_distance = context["third_distance"]
        fifth_distance = context["fifth_distance"]
        local_radius = context["local_radius"]
        neighbor_counts = context["neighbor_counts"]
        degree = context["degree"]
        local_clustering = context["local_clustering"]
        metric_values = context["metric_values"]
        pair_distances = context["pair_distances"]
        graph = context["graph"]
        component_sizes = context["component_sizes"]

        spine_records = []
        for i, spine in enumerate(self.spines):
            record = {
                "dendrite": self.name,
                "spine_name": spine.name,
                "dbscan_label": int(labels[i]),
                "is_clustered": bool(labels[i] != -1),
                "nearest_neighbor_distance": float(nn_distance[i]) if np.isfinite(nn_distance[i]) else np.nan,
                "second_neighbor_distance": float(second_distance[i]) if np.isfinite(second_distance[i]) else np.nan,
                "third_neighbor_distance": float(third_distance[i]) if np.isfinite(third_distance[i]) else np.nan,
                "fifth_neighbor_distance": float(fifth_distance[i]) if np.isfinite(fifth_distance[i]) else np.nan,
                "local_radius": float(local_radius),
                "local_neighbor_count": int(neighbor_counts[i]),
                "graph_degree": float(degree[i]),
                "local_clustering": float(local_clustering[i]),
            }
            for metric, values in metric_values.items():
                own_value = values[i]
                record[metric] = float(own_value) if np.isfinite(own_value) else np.nan
            spine_records.append(record)

        finite_nn_distance = nn_distance[np.isfinite(nn_distance)]
        nn_mean = float(np.mean(finite_nn_distance)) if len(finite_nn_distance) else np.nan
        nn_std = float(np.std(finite_nn_distance, ddof=1)) if len(finite_nn_distance) > 1 else 0.0
        pair_distance_mean = float(np.mean(pair_distances)) if len(pair_distances) else np.nan
        pair_distance_std = float(np.std(pair_distances, ddof=1)) if len(pair_distances) > 1 else 0.0

        summary = {
            "dendrite": self.name,
            "n_spines": int(n),
            "local_radius": float(local_radius),
            "dbscan_eps": float(getattr(self, "dbscan_eps", 0) or 0),
            "dbscan_min_samples": int(getattr(self, "dbscan_min_samples", 0) or 0),
            "dbscan_noise_fraction": float(np.mean(labels == -1)) if len(labels) else np.nan,
            "nearest_neighbor_mean": nn_mean,
            "nearest_neighbor_median": float(np.median(finite_nn_distance)) if len(finite_nn_distance) else np.nan,
            "nearest_neighbor_std": nn_std,
            "nearest_neighbor_cv": (
                float(nn_std / nn_mean)
                if np.isfinite(nn_mean) and nn_mean > 0 and len(finite_nn_distance) > 1
                else np.nan
            ),
            "nearest_neighbor_q10": float(np.percentile(finite_nn_distance, 10)) if len(finite_nn_distance) else np.nan,
            "nearest_neighbor_q25": float(np.percentile(finite_nn_distance, 25)) if len(finite_nn_distance) else np.nan,
            "nearest_neighbor_q75": float(np.percentile(finite_nn_distance, 75)) if len(finite_nn_distance) else np.nan,
            "nearest_neighbor_q90": float(np.percentile(finite_nn_distance, 90)) if len(finite_nn_distance) else np.nan,
            "local_neighbor_count_mean": float(np.mean(neighbor_counts)),
            "local_neighbor_count_max": int(np.max(neighbor_counts)) if len(neighbor_counts) else 0,
            "local_isolated_fraction": float(np.mean(neighbor_counts == 0)),
            "threshold_graph_edges": int(graph.number_of_edges()),
            "threshold_graph_components": int(len(component_sizes)),
            "threshold_graph_largest_component": int(max(component_sizes)) if component_sizes else 0,
            "mean_pair_distance": pair_distance_mean,
            "median_pair_distance": float(np.median(pair_distances)) if len(pair_distances) else np.nan,
            "pair_distance_std": pair_distance_std,
            "pair_distance_cv": (
                float(pair_distance_std / pair_distance_mean)
                if np.isfinite(pair_distance_mean) and pair_distance_mean > 0 and len(pair_distances) > 1
                else np.nan
            ),
            "pair_distance_q10": float(np.percentile(pair_distances, 10)) if len(pair_distances) else np.nan,
            "pair_distance_q25": float(np.percentile(pair_distances, 25)) if len(pair_distances) else np.nan,
            "pair_distance_q75": float(np.percentile(pair_distances, 75)) if len(pair_distances) else np.nan,
            "pair_distance_q90": float(np.percentile(pair_distances, 90)) if len(pair_distances) else np.nan,
        }

        self.spatial_morphology_summary = summary
        self.spatial_morphology_spine_records = spine_records
        self.spatial_morphology_cluster_records = []
        self.spatial_morphology_test_records = []
        self.spatial_morphology_permutation_records = []
        self._spatial_analysis_context = context

    def calculate_local_neighborhood_analysis(self, local_radius: float = None) -> None:
        if (
            not hasattr(self, "_spatial_analysis_context")
            or local_radius is not None
            or not hasattr(self, "spatial_morphology_spine_records")
        ):
            self.calculate_density_distribution_analysis(local_radius=local_radius)

        context = self._spatial_analysis_context
        n = context["n"]
        if n == 0:
            self._set_empty_spatial_analysis_results()
            return

        adjacency = context["adjacency"]
        neighbor_counts = context["neighbor_counts"]
        metric_values = context["metric_values"]
        local_radius = context["local_radius"]
        graph = context["graph"]
        component_sizes = context["component_sizes"]
        pair_distances = context["pair_distances"]

        local_records = []
        for i, record in enumerate(self.spatial_morphology_spine_records):
            neighbors = np.where(adjacency[i])[0]
            local_record = {
                "dendrite": self.name,
                "spine_name": record.get("spine_name", self.spines[i].name),
                "local_radius": float(local_radius),
                "local_neighbor_count": int(neighbor_counts[i]),
                "graph_degree": float(context["degree"][i]),
                "local_clustering": float(context["local_clustering"][i]),
            }
            for metric, values in metric_values.items():
                own_value = values[i]
                if len(neighbors):
                    neighbor_values = values[neighbors]
                    neighbor_values = neighbor_values[np.isfinite(neighbor_values)]
                    neighbor_mean = float(np.mean(neighbor_values)) if len(neighbor_values) else np.nan
                else:
                    neighbor_mean = np.nan
                neighbor_abs_diff = (
                    float(abs(own_value - neighbor_mean))
                    if np.isfinite(own_value) and np.isfinite(neighbor_mean)
                    else np.nan
                )
                record[f"neighbor_mean_{metric}"] = neighbor_mean
                record[f"neighbor_abs_diff_{metric}"] = neighbor_abs_diff
                local_record[f"neighbor_mean_{metric}"] = neighbor_mean
                local_record[f"neighbor_abs_diff_{metric}"] = neighbor_abs_diff
            local_records.append(local_record)

        self.local_neighborhood_summary = {
            "dendrite": self.name,
            "n_spines": int(n),
            "local_radius": float(local_radius),
            "local_neighbor_count_mean": float(np.mean(neighbor_counts)),
            "local_neighbor_count_max": int(np.max(neighbor_counts)) if len(neighbor_counts) else 0,
            "local_isolated_fraction": float(np.mean(neighbor_counts == 0)),
            "threshold_graph_edges": int(graph.number_of_edges()),
            "threshold_graph_components": int(len(component_sizes)),
            "threshold_graph_largest_component": int(max(component_sizes)) if component_sizes else 0,
            "close_pair_fraction": float(np.mean(pair_distances <= local_radius)) if len(pair_distances) else np.nan,
        }
        self.local_neighborhood_spine_records = local_records

    def calculate_spatial_autocorrelation_analysis(
        self,
        local_radius: float = None,
        permutation_count: int = 199,
        random_state: int = 42,
    ) -> None:
        if (
            not hasattr(self, "_spatial_analysis_context")
            or local_radius is not None
            or not hasattr(self, "spatial_morphology_spine_records")
        ):
            self.calculate_density_distribution_analysis(local_radius=local_radius)

        context = self._spatial_analysis_context
        n = context["n"]
        if n == 0:
            self._set_empty_spatial_analysis_results()
            return

        self.calculate_local_neighborhood_analysis(local_radius=local_radius)
        context = self._spatial_analysis_context

        adjacency = context["adjacency"]
        neighbor_counts = context["neighbor_counts"]
        metric_values = context["metric_values"]
        local_radius = context["local_radius"]

        weights = adjacency.astype(float)
        permutation_records = []
        rng = np.random.default_rng(random_state)
        for metric, values in metric_values.items():
            moran_i = self._moran_i_from_weights(values, weights)
            geary_c = self._geary_c_from_weights(values, weights)
            neighbor_means = np.asarray([
                np.nanmean(values[np.where(adjacency[i])[0]]) if neighbor_counts[i] > 0 else np.nan
                for i in range(n)
            ], dtype=float)
            neighbor_corr, neighbor_corr_p = self._safe_spearman(values, neighbor_means)

            perm_moran = []
            perm_geary = []
            perm_neighbor_corr = []
            valid_values = values.copy()
            for _ in range(max(0, int(permutation_count))):
                permuted = rng.permutation(valid_values)
                perm_moran.append(self._moran_i_from_weights(permuted, weights))
                perm_geary.append(self._geary_c_from_weights(permuted, weights))
                perm_neighbor_means = np.asarray([
                    np.nanmean(permuted[np.where(adjacency[i])[0]]) if neighbor_counts[i] > 0 else np.nan
                    for i in range(n)
                ], dtype=float)
                perm_corr, _ = self._safe_spearman(permuted, perm_neighbor_means)
                perm_neighbor_corr.append(perm_corr)

            permutation_records.append({
                "dendrite": self.name,
                "metric": metric,
                "local_radius": float(local_radius),
                "moran_i": moran_i,
                "moran_i_permutation_p": self._permutation_p_value(moran_i, perm_moran),
                "geary_c": geary_c,
                "geary_c_permutation_p": self._permutation_p_value(geary_c - 1, [value - 1 for value in perm_geary]),
                "neighbor_spearman_r": neighbor_corr,
                "neighbor_spearman_p": neighbor_corr_p,
                "neighbor_spearman_permutation_p": self._permutation_p_value(neighbor_corr, perm_neighbor_corr),
                "n_permutations": int(max(0, int(permutation_count))),
            })

        self.spatial_morphology_permutation_records = permutation_records

    def calculate_spatial_cluster_analysis(self, local_radius: float = None) -> None:
        if (
            not hasattr(self, "_spatial_analysis_context")
            or local_radius is not None
            or not hasattr(self, "spatial_morphology_summary")
        ):
            self.calculate_density_distribution_analysis(local_radius=local_radius)

        context = self._spatial_analysis_context
        n = context["n"]
        if n == 0:
            self._set_empty_spatial_analysis_results()
            return

        distances = context["distances"]
        labels = context["labels"]
        metric_values = context["metric_values"]

        cluster_records = []
        for label in sorted(set(labels.tolist()) - {-1}):
            indices = np.where(labels == label)[0]
            sub_distances = distances[np.ix_(indices, indices)]
            upper = sub_distances[np.triu_indices_from(sub_distances, k=1)]
            upper = upper[np.isfinite(upper) & (upper > 0)]
            diameter = float(np.max(upper)) if len(upper) else 0.0
            record = {
                "dendrite": self.name,
                "cluster_label": int(label),
                "cluster_size": int(len(indices)),
                "cluster_diameter": diameter,
                "mean_intra_cluster_distance": float(np.mean(upper)) if len(upper) else 0.0,
                "median_intra_cluster_distance": float(np.median(upper)) if len(upper) else 0.0,
                "cluster_linear_density": float(len(indices) / diameter) if diameter > 0 else np.nan,
            }
            for metric, values in metric_values.items():
                cluster_values = values[indices]
                cluster_values = cluster_values[np.isfinite(cluster_values)]
                record[f"mean_{metric}"] = float(np.mean(cluster_values)) if len(cluster_values) else np.nan
                record[f"median_{metric}"] = float(np.median(cluster_values)) if len(cluster_values) else np.nan
                record[f"std_{metric}"] = float(np.std(cluster_values, ddof=1)) if len(cluster_values) > 1 else 0.0
            cluster_records.append(record)

        test_records = []
        clustered_mask = labels != -1
        isolated_mask = labels == -1
        for metric, values in metric_values.items():
            clustered_values = values[clustered_mask]
            isolated_values = values[isolated_mask]
            clustered_finite = clustered_values[np.isfinite(clustered_values)]
            isolated_finite = isolated_values[np.isfinite(isolated_values)]
            clustered_mean = float(np.mean(clustered_finite)) if len(clustered_finite) else np.nan
            isolated_mean = float(np.mean(isolated_finite)) if len(isolated_finite) else np.nan
            clustered_std = float(np.std(clustered_finite, ddof=1)) if len(clustered_finite) > 1 else 0.0
            isolated_std = float(np.std(isolated_finite, ddof=1)) if len(isolated_finite) > 1 else 0.0
            u_stat, u_p = self._safe_stat_test(mannwhitneyu, clustered_values, isolated_values)
            bm_stat, bm_p = self._safe_stat_test(brunnermunzel, clustered_values, isolated_values)
            test_records.append({
                "dendrite": self.name,
                "analysis": "clustered_vs_isolated",
                "metric": metric,
                "n_clustered": int(len(clustered_finite)),
                "n_isolated": int(len(isolated_finite)),
                "clustered_mean": clustered_mean,
                "isolated_mean": isolated_mean,
                "clustered_median": float(np.median(clustered_finite)) if len(clustered_finite) else np.nan,
                "isolated_median": float(np.median(isolated_finite)) if len(isolated_finite) else np.nan,
                "clustered_std": clustered_std,
                "isolated_std": isolated_std,
                "clustered_cv": (
                    float(clustered_std / clustered_mean)
                    if np.isfinite(clustered_mean) and clustered_mean > 0 and len(clustered_finite) > 1
                    else np.nan
                ),
                "isolated_cv": (
                    float(isolated_std / isolated_mean)
                    if np.isfinite(isolated_mean) and isolated_mean > 0 and len(isolated_finite) > 1
                    else np.nan
                ),
                "mannwhitney_statistic": u_stat,
                "mannwhitney_p": u_p,
                "brunnermunzel_statistic": bm_stat,
                "brunnermunzel_p": bm_p,
                "cliffs_delta": self._cliffs_delta(clustered_values, isolated_values),
            })

        cluster_sizes = [record["cluster_size"] for record in cluster_records]
        self.spatial_morphology_summary.update({
            "n_dbscan_clusters": int(len(cluster_records)),
            "mean_dbscan_cluster_size": float(np.mean(cluster_sizes)) if cluster_sizes else 0.0,
            "max_dbscan_cluster_size": int(max(cluster_sizes)) if cluster_sizes else 0,
        })
        self.spatial_morphology_cluster_records = cluster_records
        self.spatial_morphology_test_records = test_records

    def calculate_comprehensive_spatial_analysis(
        self,
        local_radius: float = None,
        permutation_count: int = 199,
        random_state: int = 42,
    ) -> None:
        self.calculate_density_distribution_analysis(local_radius=local_radius)
        self.calculate_local_neighborhood_analysis(local_radius=local_radius)
        self.calculate_spatial_autocorrelation_analysis(
            permutation_count=permutation_count,
            random_state=random_state,
        )
        self.calculate_spatial_cluster_analysis()

    def calculate_spatial_morphology_analysis(
        self,
        local_radius: float = None,
        permutation_count: int = 199,
        random_state: int = 42,
    ) -> None:
        self.calculate_comprehensive_spatial_analysis(
            local_radius=local_radius,
            permutation_count=permutation_count,
            random_state=random_state,
        )

    def save_spatial_morphology_analysis(self) -> None:
        if (
            not hasattr(self, "spatial_morphology_summary")
            or "n_dbscan_clusters" not in self.spatial_morphology_summary
        ):
            self.calculate_comprehensive_spatial_analysis()

        save_spatial_morphology_summary_records.append(self.spatial_morphology_summary)
        save_spatial_morphology_spine_records.extend(self.spatial_morphology_spine_records)
        save_spatial_morphology_cluster_records.extend(self.spatial_morphology_cluster_records)
        save_spatial_morphology_test_records.extend(self.spatial_morphology_test_records)
        save_spatial_morphology_permutation_records.extend(self.spatial_morphology_permutation_records)

        pd.DataFrame(save_spatial_morphology_summary_records).to_csv(
            output_path("spatial_morphology_summary.csv"),
            index=False,
        )
        pd.DataFrame(save_spatial_morphology_spine_records).to_csv(
            output_path("spatial_morphology_spines.csv"),
            index=False,
        )
        pd.DataFrame(save_spatial_morphology_cluster_records).to_csv(
            output_path("spatial_morphology_clusters.csv"),
            index=False,
        )
        pd.DataFrame(save_spatial_morphology_test_records).to_csv(
            output_path("spatial_morphology_tests.csv"),
            index=False,
        )
        pd.DataFrame(save_spatial_morphology_permutation_records).to_csv(
            output_path("spatial_morphology_permutation.csv"),
            index=False,
        )

    @staticmethod
    def _is_scalar_metric_value(value: Any) -> bool:
        return value is None or isinstance(
            value,
            (
                str,
                bool,
                int,
                float,
                np.integer,
                np.floating,
                np.bool_,
            ),
        )

    @staticmethod
    def _to_json_compatible(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return Dendrite._to_json_compatible(value.tolist())
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, dict):
            return {
                str(key): Dendrite._to_json_compatible(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [Dendrite._to_json_compatible(item) for item in value]
        return value

    @staticmethod
    def _as_metric_scalar(value: Any) -> Any:
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()
        return value

    @staticmethod
    def _add_prefixed_metrics(record: Dict[str, Any], prefix: str, metrics: Dict[str, Any], skip_keys=None) -> None:
        skip_keys = set(skip_keys or [])
        for key, value in metrics.items():
            if key in skip_keys:
                continue
            column_name = f"{prefix}_{key}"
            if Dendrite._is_scalar_metric_value(value):
                record[column_name] = Dendrite._as_metric_scalar(value)
            else:
                record[f"{column_name}_json"] = json.dumps(
                    Dendrite._to_json_compatible(value),
                    ensure_ascii=False,
                )

    def build_structural_organization_vector(self) -> Dict[str, Any]:
        if not hasattr(self, "nndist") or not hasattr(self, "pair_distance_profile_values"):
            self.calculate_density_distribution_analysis()
        if not hasattr(self, "local_neighborhood_summary"):
            self.calculate_local_neighborhood_analysis()
        if not getattr(self, "spatial_morphology_permutation_records", None):
            self.calculate_spatial_autocorrelation_analysis(permutation_count=0)

        dbscan_labels = np.asarray(getattr(self, "dbscan_labels", []))
        if dbscan_labels.shape[0] != len(self.spines):
            self.calculate_cluster_metrics()

        if (
            not hasattr(self, "spatial_morphology_summary")
            or "n_dbscan_clusters" not in self.spatial_morphology_summary
        ):
            self.calculate_comprehensive_spatial_analysis()

        record = {
            "Name": self.name,
            "Type": self.name[:2],
            "n_spines": int(len(self.spines)),
            "dendrite_Volume": self.volume,
            "dendrite_Length": self.length,
            "dendrite_Radius": self.radius,
            "density_pair_distance_profile_entropy": self.pair_distance_profile_entropy,
            "graph_average_clustering": getattr(self, "g_average_clustering", np.nan),
            "graph_mean_cluster_size": getattr(self, "g_mean_cluster_size", np.nan),
            "graph_characteristic_extent": getattr(self, "g_characteristic_extent", np.nan),
            "graph_modularity": getattr(self, "g_modularity", np.nan),
        }

        self._add_prefixed_metrics(
            record,
            "spatial",
            self.spatial_morphology_summary,
            skip_keys={
                "dendrite",
                "n_spines",
                "local_radius",
                "local_neighbor_count_mean",
                "local_neighbor_count_max",
                "local_isolated_fraction",
                "threshold_graph_edges",
                "threshold_graph_components",
                "threshold_graph_largest_component",
            },
        )

        self._add_prefixed_metrics(
            record,
            "neighborhood",
            getattr(self, "local_neighborhood_summary", {}),
            skip_keys={"dendrite", "n_spines"},
        )

        for permutation_record in getattr(self, "spatial_morphology_permutation_records", []):
            metric = permutation_record.get("metric", "unknown")
            metric_prefix = f"autocorr_{metric}"
            for key, value in permutation_record.items():
                if key in {"dendrite", "metric"}:
                    continue
                if self._is_scalar_metric_value(value):
                    record[f"{metric_prefix}_{key}"] = self._as_metric_scalar(value)
                else:
                    record[f"{metric_prefix}_{key}_json"] = json.dumps(
                        self._to_json_compatible(value),
                        ensure_ascii=False,
                    )

        for test_record in getattr(self, "spatial_morphology_test_records", []):
            metric = test_record.get("metric", "unknown")
            analysis = test_record.get("analysis", "test")
            test_prefix = f"test_{analysis}_{metric}"
            for key, value in test_record.items():
                if key in {"dendrite", "analysis", "metric", "n_clustered", "n_isolated"}:
                    continue
                if self._is_scalar_metric_value(value):
                    record[f"{test_prefix}_{key}"] = self._as_metric_scalar(value)
                else:
                    record[f"{test_prefix}_{key}_json"] = json.dumps(
                        self._to_json_compatible(value),
                        ensure_ascii=False,
                    )

        return record

    def save_structural_organization_vector(self) -> None:
        record = self.build_structural_organization_vector()

        save_structural_organization_vector_records[:] = [
            saved_record
            for saved_record in save_structural_organization_vector_records
            if saved_record.get("Name") != self.name
        ]
        save_structural_organization_vector_records.append(record)

        pd.DataFrame(save_structural_organization_vector_records).to_csv(
            output_path("dendrite_structural_organization_vectors.csv"),
            index=False,
        )

    def calculate_spine_distance_matrices(self, methods=None, output_dir=None, pair_for_path=None):
        if methods is None:
            methods = ("mesh_graph",)

        if not hasattr(self, "mesh"):
            raise ValueError("Surface distance methods require Dendrite.mesh. Create Dendrite with dendrite_meshes, not only from saved input JSON.")

        attachment_points = [s.junction_center_coord for s in self.spines]
        cylindrical_points = None
        if self.cylindr_flag:
            cylindrical_points = [s.center_coord_c for s in self.spines]

        if output_dir is not None:
            return save_distance_method_comparison(
                dendrite_mesh=self.mesh,
                attachment_points=attachment_points,
                cylindrical_points=cylindrical_points,
                output_dir=output_dir,
                methods=methods,
                radius=self.radius,
                pair_for_path=pair_for_path,
            )

        return calculate_spine_distance_matrices(
            dendrite_mesh=self.mesh,
            attachment_points=attachment_points,
            cylindrical_points=cylindrical_points,
            methods=methods,
            radius=self.radius,
            pair_for_path=pair_for_path,
        )

    def save_spine_coords(self) -> None:
        # with open('spine_coords.json', 'w') as f:
        # with open('metrics/9009/spine_coords.json', 'w') as f:
        # with open('metrics/wt_old_st/spine_coords.json', 'w') as f:
        with open(output_path('spine_coords.json'), 'w') as f:
            json.dump(save_coords, f)

    def save_spine_metrics(self) -> None:
        for s in self.spines:
            save_spine_metric_dict[s.name] = s.metrics

        # with open('spine_metrics.json', 'w') as f:
        # with open('metrics/9009/spine_metrics.json', 'w') as f:
        # with open('metrics/wt_old_st/spine_metrics.json', 'w') as f:
        with open(output_path('spine_metrics.json'), 'w') as f:
            json.dump(save_spine_metric_dict, f)

    def save_init_metrics(self) -> None:
        save_dendr_metric_dict[self.name] = {'Volume' : self.volume, 
                                             'Length' : self.length, 
                                             'Radius' : self.radius, 
                                             'Dr' : self.dr, 
                                             'Volume_around_dendr' : self.volume_around_dendr}
        
        with open(output_path('dendr_metrics.json'), 'w') as f:
            json.dump(save_dendr_metric_dict, f)

    def save_grouping_metrics(self):
        save_grouping_dendr_metric_dict[self.name] = {}
        save_grouping_dendr_metric_dict[self.name]['NNdist'] = self.nndist
        save_grouping_dendr_metric_dict[self.name]['NNdist_norm'] = self.nndist_norm

        save_grouping_dendr_metric_dict[self.name]['PairDistanceProfile_values'] = np.asarray(
            self.pair_distance_profile_values
        ).tolist()
        save_grouping_dendr_metric_dict[self.name]['PairDistanceProfile_r_values'] = np.asarray(
            self.pair_distance_profile_r_values
        ).tolist()
        save_grouping_dendr_metric_dict[self.name]['PairDistanceProfile_entropy'] = self.pair_distance_profile_entropy

        # Legacy global Volume Moran fields are no longer written by the
        # current pipeline. Use spatial_morphology_permutation.csv and
        # autocorr_<metric>_* columns in dendrite_structural_organization_vectors.csv.
        # save_grouping_dendr_metric_dict[self.name]['Moran_I'] = self.moran_I
        # save_grouping_dendr_metric_dict[self.name]['Moran_zI'] = self.moran_z
        # save_grouping_dendr_metric_dict[self.name]['Moran_p'] = self.moran_p

        # save_grouping_dendr_metric_dict[self.name]['Getis_Ord_G'] = self.getis_ord_G
        # save_grouping_dendr_metric_dict[self.name]['Getis_Ord_zG'] = self.getis_ord_z
        # save_grouping_dendr_metric_dict[self.name]['Getis_Ord_p'] = self.getis_ord_p

        # with open('metrics/grouping_dendr_metrics.json', 'w') as f:
        # with open('metrics/9009/grouping_dendr_metrics.json', 'w') as f:
        # with open('metrics/wt_old_st/grouping_dendr_metrics.json', 'w') as f:
        with open(output_path('grouping_dendr_metrics.json'), 'w') as f:
            json.dump(save_grouping_dendr_metric_dict, f)

    def save_cluster_metrics(self):
        save_cluster_dendr_metric_dict[self.name] = {}
        save_cluster_dendr_metric_dict[self.name]['DBscan_eps'] = self.dbscan_eps
        save_cluster_dendr_metric_dict[self.name]['DBscan_min_samples'] = self.dbscan_min_samples
        # Legacy statistic/p-value from the old DBSCAN parameter search are no
        # longer meaningful after switching to the k-distance eps rule.
        # save_cluster_dendr_metric_dict[self.name]['DBscan_statistic'] = self.dbscan_statistic
        # save_cluster_dendr_metric_dict[self.name]['DBscan_p_value'] = self.dbscan_p_value
        save_cluster_dendr_metric_dict[self.name]['DBscan_noise'] = self.dbscan_noise

        # with open('metrics/grouping_dendr_metrics.json', 'w') as f:
        # with open('metrics/9009/grouping_dendr_metrics.json', 'w') as f:
        # with open('metrics/wt_old_st/grouping_dendr_metrics.json', 'w') as f:
        with open(output_path('cluster_dendr_metrics.json'), 'w') as f:
            json.dump(save_cluster_dendr_metric_dict, f)

    def save_graph_metrics(self):
        save_graph_dendr_metric_dict[self.name] = {}
        # save_graph_dendr_metric_dict[self.name]['DBscan_eps'] = self.dbscan_eps
        # save_graph_dendr_metric_dict[self.name]['DBscan_min_samples'] = self.dbscan_min_samples
        # save_graph_dendr_metric_dict[self.name]['DBscan_statistic'] = self.dbscan_statistic
        # save_graph_dendr_metric_dict[self.name]['DBscan_p_value'] = self.dbscan_p_value
        # save_graph_dendr_metric_dict[self.name]['DBscan_noise'] = self.dbscan_noise

        save_graph_dendr_metric_dict[self.name]['g_cluster_sizes'] = self.g_cluster_sizes
        save_graph_dendr_metric_dict[self.name]['g_mean_cluster_size'] = self.g_mean_cluster_size
        save_graph_dendr_metric_dict[self.name]['g_characteristic_extent'] = self.g_characteristic_extent
        save_graph_dendr_metric_dict[self.name]['g_average_clustering'] = self.g_average_clustering
        save_graph_dendr_metric_dict[self.name]['g_modularity'] = self.g_modularity

        # with open('metrics/grouping_dendr_metrics.json', 'w') as f:
        # with open('metrics/9009/grouping_dendr_metrics.json', 'w') as f:
        # with open('metrics/wt_old_st/grouping_dendr_metrics.json', 'w') as f:
        with open(output_path('graph_dendr_metrics.json'), 'w') as f:
            json.dump(save_graph_dendr_metric_dict, f)

    def save_dendr_metrics(self) -> None:
        if self.cylindr_flag:
            dendrite = {"Name": self.name, "Type": self.name[:2], "NNdist": self.nndist, "PairDistanceProfile_entropy": self.pair_distance_profile_entropy, 
                        # Legacy global Volume Moran fields are excluded from current aggregate metrics.
                        # "Moran_I": self.moran_I, "Moran_zI": self.moran_z, "Moran_p": self.moran_p,
                        # "Getis_Ord_G": self.getis_ord_G_c, "Getis_Ord_zG": self.getis_ord_z_c, "Getis_Ord_p": self.getis_ord_p_c,
                        "Eps": self.dbscan_eps, "Min_samples": self.dbscan_min_samples,
                        # "DBSCAN_statistic": self.dbscan_statistic, "DBSCAN_p_value": self.dbscan_p_value,
                        "Noise": self.dbscan_noise,
                        "g_mean_cluster_size": self.g_mean_cluster_size, "g_characteristic_extent": self.g_characteristic_extent, 
                        "g_average_clustering": self.g_average_clustering, "g_modularity": self.g_modularity}
            
            for k, m in enumerate([self.db_matrix_class, self.db_matrix_cluster, self.n_matrix_class, self.n_matrix_cluster]):
                matrix = m / m.max()
                rows, cols = matrix.shape
                matrix_dict = {}
                
                for i in range(rows):
                    for j in range(cols):
                        if k == 0:
                            key = f"DBSCAN_class_{i+1}{j+1}"
                        elif k == 1:
                            key = f"DBSCAN_cluster_{i+1}{j+1}"
                        elif k == 2:
                            key = f"N_class_{i+1}{j+1}"
                        elif k == 3:
                            key = f"N_cluster_{i+1}{j+1}"
                        matrix_dict[key] = matrix[i, j]

                dendrite.update(matrix_dict)

            save_all_dendr_metric_dict.append(dendrite)
  
    def save_dendr_metrics_without_class_cluster(self) -> None:
        if self.cylindr_flag:
                    
            dendrite = {"Name": self.name, "Type": self.name[:2], "NNdist": self.nndist, "PairDistanceProfile_entropy": self.pair_distance_profile_entropy, 
                        # Legacy global Volume Moran fields are excluded from current aggregate metrics.
                        # "Moran_I": self.moran_I, "Moran_zI": self.moran_z, "Moran_p": self.moran_p,
                        # "Getis_Ord_G": self.getis_ord_G_c, "Getis_Ord_zG": self.getis_ord_z_c, "Getis_Ord_p": self.getis_ord_p_c,
                        "Eps": self.dbscan_eps, "Min_samples": self.dbscan_min_samples,
                        # "DBSCAN_statistic": self.dbscan_statistic, "DBSCAN_p_value": self.dbscan_p_value,
                        "Noise": self.dbscan_noise,
                        "g_mean_cluster_size": self.g_mean_cluster_size, "g_characteristic_extent": self.g_characteristic_extent, 
                        "g_average_clustering": self.g_average_clustering, "g_modularity": self.g_modularity}

            save_all_dendr_metric_dict.append(dendrite)


# Backward-compatible semantic alias: historically this class was named
# `Dendrite`, but it represents one dendritic branch/fragment with its spines.
DendriteBranch = Dendrite
