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


def _fallback_dendrite_length_any_mesh(dendr_mesh: Any) -> float:
    length = centerline_length_from_mesh(dendr_mesh)
    print(f"  length (mesh-graph centerline fallback) = {length:.2f}")
    return length


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

    r_values: List[float] = []
    pcf_values: List[float] = []
    entropy: float
    r_values_c: List[float] = []
    pcf_values_c: List[float] = []
    entropy_c: float

    moran_I: float
    moran_z: float
    moran_p: float
    moran_I_c: float
    moran_z_c: float
    moran_p_c: float

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
    dbscan_statistic: float = 0
    dbscan_p_value: float = 0
    dbscan_noise: float = 0

    dbscan_labels_c: Any = []
    dbscan_eps_c: float = 0
    dbscan_min_samples_c: int = 0
    dbscan_statistic_c: float = 0
    dbscan_p_value_c: float = 0
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
        self.r_values = []
        self.pcf_values = []
        self.r_values_c = []
        self.pcf_values_c = []
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
                if isinstance(dendr_mesh, Polyhedron_3):
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
        self.r_values = loaded_dict[self.name]['r_values']
        self.pcf_values = loaded_dict[self.name]['PCF_values']
        self.entropy = loaded_dict[self.name]['Entropy']

        self.moran_I = loaded_dict[self.name]['Moran_I']
        self.moran_z = loaded_dict[self.name]['Moran_zI']
        self.moran_p = loaded_dict[self.name]['Moran_p']

        # self.getis_ord_G = loaded_dict[self.name]['Getis_Ord_G']
        # self.getis_ord_z = loaded_dict[self.name]['Getis_Ord_zG']
        # self.getis_ord_p = loaded_dict[self.name]['Getis_Ord_p']

        # self.getis_ord_G, self.getis_ord_z, self.getis_ord_p = calculate_Getis_Ord_G(self.center_coords, metric_dict_for_autocorr, False, 7, self.name)['Volume']


        if self.cylindr_flag:
            # Старые файлы могли хранить отдельные cylindrical-метрики с суффиксом _c.
            # В новых файлах они не сохраняются: mesh_graph-метрики лежат в обычных полях.
            self.nndist_c = loaded_dict[self.name].get('NNdist_c', self.nndist)
            self.nndist_norm_c = loaded_dict[self.name].get('NNdist_norm_c', self.nndist_norm)
            self.r_values_c = loaded_dict[self.name].get('r_values_c', self.r_values)
            self.pcf_values_c = loaded_dict[self.name].get('PCF_values_c', self.pcf_values)
            self.entropy_c = loaded_dict[self.name].get('Entropy_c', self.entropy)

            self.moran_I_c = loaded_dict[self.name].get('Moran_I_c', self.moran_I)
            self.moran_z_c = loaded_dict[self.name].get('Moran_zI_c', self.moran_z)
            self.moran_p_c = loaded_dict[self.name].get('Moran_p_c', self.moran_p)

            # self.getis_ord_G_c = loaded_dict[self.name]['Getis_Ord_G_c']
            # self.getis_ord_z_c = loaded_dict[self.name]['Getis_Ord_zG_c']
            # self.getis_ord_p_c = loaded_dict[self.name]['Getis_Ord_p_c']

            # self.getis_ord_G_c, self.getis_ord_z_c, self.getis_ord_p_c = calculate_Getis_Ord_G(self.center_coords_c, metric_dict_for_autocorr, True, 7, self.name)['Volume']

    def load_cluster_metrics(self) -> None:
        with open('input/cluster_dendr_metrics.json', 'r') as f:
            loaded_dict = json.load(f)

        self.dbscan_eps = loaded_dict[self.name]['DBscan_eps']
        self.dbscan_min_samples = loaded_dict[self.name]['DBscan_min_samples']
        self.dbscan_statistic = loaded_dict[self.name]['DBscan_statistic']
        self.dbscan_p_value = loaded_dict[self.name]['DBscan_p_value']
        self.dbscan_noise = loaded_dict[self.name]['DBscan_noise']

        if self.cylindr_flag:
            # Старые файлы могли хранить отдельные cylindrical-метрики с суффиксом _c.
            # В новых файлах они не сохраняются: mesh_graph-метрики лежат в обычных полях.
            self.dbscan_eps_c = loaded_dict[self.name].get('DBscan_eps_c', self.dbscan_eps)
            self.dbscan_min_samples_c = loaded_dict[self.name].get('DBscan_min_samples_c', self.dbscan_min_samples)
            self.dbscan_statistic_c = loaded_dict[self.name].get('DBscan_statistic_c', self.dbscan_statistic)
            self.dbscan_p_value_c = loaded_dict[self.name].get('DBscan_p_value_c', self.dbscan_p_value)
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

    def calculate_grouping_metrics(self) -> None:
        # metric_dict_for_autocorr = { 'Volume': [spine.metrics['Volume'] for spine in self.spines] } 
        metric_dict_for_autocorr = {
            "Volume": [spine.metrics['Volume'] for spine in self.spines],
        }

        mesh_graph_distance_matrix = self.get_mesh_graph_distance_matrix()
        distance_points = self.get_spine_distance_points()
        self.nndist, self.nndist_norm = calculate_NNDist(
            distance_points,
            self.volume_around_dendr,
            0,
            distance_matrix=mesh_graph_distance_matrix,
        )
        self.r_values, self.pcf_values = calculate_PCF(
            distance_points,
            self.volume_around_dendr,
            self.radius,
            self.dr,
            self.length,
            False,
            distance_matrix=mesh_graph_distance_matrix,
        )
        self.entropy = calculate_Shannon_entropy(self.pcf_values)
        self.moran_I, self.moran_z, self.moran_p = calculate_Morans_I(
            distance_points,
            metric_dict_for_autocorr,
            False,
            threshold_distance=5,
            distance_matrix=mesh_graph_distance_matrix,
            distance_label="mesh_graph",
        )['Volume']
        # self.getis_ord_G, self.getis_ord_z, self.getis_ord_p = calculate_Getis_Ord_G(self.center_coords, metric_dict_for_autocorr, False, threshold_distance = 5)['Volume']
        
        if self.cylindr_flag:
            # Backwards-compatible *_c fields now mirror mesh_graph-based metrics.
            self.nndist_c, self.nndist_norm_c = self.nndist, self.nndist_norm
            self.r_values_c, self.pcf_values_c = self.r_values, self.pcf_values
            self.entropy_c = self.entropy
            self.moran_I_c, self.moran_z_c, self.moran_p_c = self.moran_I, self.moran_z, self.moran_p
            # self.getis_ord_G_c, self.getis_ord_z_c, self.getis_ord_p_c = calculate_Getis_Ord_G(self.center_coords_c, metric_dict_for_autocorr, True, threshold_distance = 5)['Volume']
        
    def calculate_cluster_metrics(self) -> None:
        spine_metrics_dict_for_dbscan = {}
        for s in self.spines:
            spine_metrics_dict_for_dbscan[(s.center_coord[0], s.center_coord[1], s.center_coord[2])] = { 'Volume' : s.metrics['Volume'] }
        points = [[s.center_coord[0], s.center_coord[1], s.center_coord[2]] for s in self.spines]
        mesh_graph_distance_matrix = self.get_mesh_graph_distance_matrix()

        self.dbscan_labels, self.dbscan_eps, self.dbscan_min_samples, self.dbscan_statistic, self.dbscan_p_value = dbscan(
            points,
            spine_metrics_dict_for_dbscan,
            0,
            'graphics/dbscan/' + self.name,
            distance_matrix=mesh_graph_distance_matrix,
            distance_label="mesh_graph",
        )

        points = np.array([s.center_coord for s in self.spines])

        class_member_mask = (self.dbscan_labels == -1)
        filtered_points = points[class_member_mask]
        self.dbscan_noise = len(filtered_points)/len(points)

        if self.cylindr_flag:
            # Backwards-compatible *_c fields now mirror mesh_graph-based DBSCAN.
            self.dbscan_labels_c = self.dbscan_labels
            self.dbscan_eps_c = self.dbscan_eps
            self.dbscan_min_samples_c = self.dbscan_min_samples
            self.dbscan_statistic_c = self.dbscan_statistic
            self.dbscan_p_value_c = self.dbscan_p_value
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

        # Находим сообщества (алгоритм Лувена)
        partition = community_louvain.best_partition(G, weight='weight')
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
        self.g_modularity = community_louvain.modularity(partition, G, weight='weight')

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

        save_grouping_dendr_metric_dict[self.name]['PCF_values'] = self.pcf_values.tolist()
        save_grouping_dendr_metric_dict[self.name]['r_values'] = self.r_values.tolist()
        # save_grouping_dendr_metric_dict[self.name]['PCF_values'] = self.pcf_values
        # save_grouping_dendr_metric_dict[self.name]['r_values'] = self.r_values
        save_grouping_dendr_metric_dict[self.name]['Entropy'] = self.entropy

        save_grouping_dendr_metric_dict[self.name]['Moran_I'] = self.moran_I
        save_grouping_dendr_metric_dict[self.name]['Moran_zI'] = self.moran_z
        save_grouping_dendr_metric_dict[self.name]['Moran_p'] = self.moran_p

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
        save_cluster_dendr_metric_dict[self.name]['DBscan_statistic'] = self.dbscan_statistic
        save_cluster_dendr_metric_dict[self.name]['DBscan_p_value'] = self.dbscan_p_value
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
            dendrite = {"Name": self.name, "Type": self.name[:2], "NNdist": self.nndist, "PCF_entrophy": self.entropy, 
                        "Moran_I": self.moran_I, "Moran_zI": self.moran_z, "Moran_p": self.moran_p,
                        # "Getis_Ord_G": self.getis_ord_G_c, "Getis_Ord_zG": self.getis_ord_z_c, "Getis_Ord_p": self.getis_ord_p_c,
                        "Eps": self.dbscan_eps, "Min_samples": self.dbscan_min_samples,
                        "DBSCAN_statistic": self.dbscan_statistic, "DBSCAN_p_value": self.dbscan_p_value, "Noise": self.dbscan_noise,
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
                    
            dendrite = {"Name": self.name, "Type": self.name[:2], "NNdist": self.nndist, "PCF_entrophy": self.entropy, 
                        "Moran_I": self.moran_I, "Moran_zI": self.moran_z, "Moran_p": self.moran_p,
                        # "Getis_Ord_G": self.getis_ord_G_c, "Getis_Ord_zG": self.getis_ord_z_c, "Getis_Ord_p": self.getis_ord_p_c,
                        "Eps": self.dbscan_eps, "Min_samples": self.dbscan_min_samples,
                        "DBSCAN_statistic": self.dbscan_statistic, "DBSCAN_p_value": self.dbscan_p_value, "Noise": self.dbscan_noise,
                        "g_mean_cluster_size": self.g_mean_cluster_size, "g_characteristic_extent": self.g_characteristic_extent, 
                        "g_average_clustering": self.g_average_clustering, "g_modularity": self.g_modularity}

            save_all_dendr_metric_dict.append(dendrite)
