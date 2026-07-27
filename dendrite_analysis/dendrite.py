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
    """Оценивает длину дендрита fallback-методом для произвольного mesh.

    Входные данные: mesh дендрита.
    Выходные данные: численная оценка длины дендрита.
    """
    length = centerline_length_from_mesh(dendr_mesh)
    print(f"  length (mesh-graph centerline fallback) = {length:.2f}")
    return length


def _polyline_length(points: Any) -> float:
    """Вычисляет длину полилинии по последовательности точек. 
    Фильтрует невалидные точки и суммирует длины соседних сегментов.

    Входные данные: массив координат.
    Выходные данные: длина полилинии или `0.0`.
    """
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
    """Извлекает и суммирует длину skeleton-объекта.

    Входные данные: skeleton в формате массива, словаря, списка или вложенной
    объектной структуры.
    Выходные данные: суммарная длина skeleton или `0.0`.
    """
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
    """Возвращает длину зарегистрированного skeleton для mesh дендрита.

    Входные данные: mesh дендрита.
    Выходные данные: длина skeleton или `0.0`, если skeleton не зарегистрирован.
    """
    skeleton = get_dendrite_skeleton(dendr_mesh)
    length = _skeleton_length_from_object(skeleton)
    if np.isfinite(length) and length > 0:
        print(f"  dendr_len (registered branch_skeleton.npy) = {length:.2f}")
        return float(length)
    return 0.0


def _fallback_dendrite_volume_any_mesh(dendr_mesh: Any) -> float:
    """Оценивает объём дендрита fallback-методом для произвольного mesh.
    Для закрытого trimesh использует mesh volume, иначе оценивает
    объём через цилиндрическую аппроксимацию по PCA-оси.

    Входные данные: mesh дендрита.
    Выходные данные: численная оценка объёма.
    """
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
    """Модель одной дендритной ветви с набором шипиков.
    Хранит геометрию ветви, объекты шипиков, матрицу расстояний по
    сетке дендрита и результаты пространственного анализа.

    Входные данные: имя ветви, mesh дендрита и словарь mesh-объектов шипиков.
    Выходные данные: объект состояния; расчётные таблицы и итоговый вектор
    признаков формируются отдельными методами расчёта и сохранения.
    """

    name: str
    mesh: Any
    spines: List[Spine]
    spine_meshes: MeshDataset

    volume: float
    length: float
    radius: float

    center_coords: List[float]
    dists: List[float]
    distance_matrix: Any

    pair_distance_profile_r_values: List[float]
    pair_distance_profile_values: List[float]
    pair_distance_profile_entropy: float

    dbscan_labels: Any
    dbscan_eps: float = 0
    dbscan_min_samples: int = 0
    dbscan_noise: float = 0

    g_cluster_sizes: List[int]
    g_mean_cluster_size: float
    g_characteristic_extent: float
    g_average_clustering: float
    g_modularity: float

    def __init__(
        self,
        dendr_name: str,
        dendrite_meshes: MeshDataset = None,
        spine_meshes: MeshDataset = None,
        save_spine_data_on_init: bool = True,
    ) -> None:
        """Инициализирует объект дендритной ветви.
        Cоздаёт внутренние контейнеры, рассчитывает базовые метрики
        дендрита и создаёт объекты `Spine`.

        Входные данные: 
        `dendr_name` — имя ветви; 
        `dendrite_meshes` — словарь
        mesh-объектов дендрита, обычно из одного элемента; 
        `spine_meshes` — словарь mesh-объектов шипиков; 
        `save_spine_data_on_init` — флаг явного сохранения координат 
        и морфологических метрик шипиков при инициализации.

        Выходные данные: заполненный объект `Dendrite`; файлы записываются
        только при `save_spine_data_on_init=True` или при явном вызове `save_*`.
        """
        print('Dendrite init')

        self.spines = []
        self.center_coords = []
        
        self.dists = []  
        self.pair_distance_profile_r_values = []
        self.pair_distance_profile_values = []
        self.dbscan_labels = []

        self.g_cluster_sizes = []
        self.g_mean_cluster_size = 0
        self.g_characteristic_extent = 0
        self.g_average_clustering = 0
        self.g_modularity = 0

        self.name = dendr_name

        if dendrite_meshes is None:
            raise ValueError("Dendrite requires dendrite_meshes; loading old precomputed dendrite JSON is disabled.")
        if spine_meshes is None:
            raise ValueError("Dendrite requires spine_meshes; loading old precomputed spine JSON is disabled.")

        self.calculate_init_metrics(dendrite_meshes)
        self.spine_meshes = spine_meshes
        self.calculate_and_create_spines()

        if save_spine_data_on_init:
            self.save_spine_coords() # сохранение координат шипиков
            self.save_spine_metrics() # сохранение метрик

    def calculate_init_metrics(self, dendrite_meshes: MeshDataset = None) -> None:
        """Вычисляет базовые геометрические метрики дендрита.
        Выбирает меш текущей ветви, вычисляет объём, 
        длину по скелету или fallback-методу и радиус цилиндрической
        аппроксимации из объёма и длины.

        Входные данные: словарь мешей дендрита `dendrite_meshes`.
        Выходные данные: поля `mesh`, `volume`, `length`, `radius`.
        """
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
        """Создаёт объекты шипиков и определяет их основные координаты.
        Вычисляет точку крепления и геометрический центр каждого
        шипика, сохраняет координаты во внутренний словарь `save_coords`,
        создаёт объекты `Spine`.

        Входные данные: `self.spine_meshes`, где ключ — имя шипика, значение —
        его меш.
        Выходные данные: заполненные `self.center_coords` и `self.spines`.
        """
        junction_center_klass = spine_metric_classes['JunctionCenterSpineMetric']
        center_klass = spine_metric_classes['CenterSpineMetric']

        junction_center_coords = [] 

        for (spine_name, spine_mesh) in self.spine_meshes.items():
            print(f"  junction/center metrics: {spine_name}", flush=True)
            # середина области крепления шипика
            junction_center_vec = junction_center_klass(spine_mesh)._value
            junction_center_coord = (junction_center_vec.x(), junction_center_vec.y(), junction_center_vec.z())  # tuple, тк неизменяемый
            junction_center_coords.append(junction_center_coord)
            
            # середина шипика
            center_vec = center_klass(spine_mesh)._value
            center_coord = (center_vec.x(), center_vec.y(), center_vec.z())  # tuple, тк неизменяемый
            self.center_coords.append(center_coord)
        
        spine_meshes_list = list(self.spine_meshes.items())

        save_coords[self.name] = {}

        for i in range(len(self.spine_meshes)):
            save_coords[self.name][spine_meshes_list[i][0]] = {
                'junction_center_coord': junction_center_coords[i],
                'center_coord': self.center_coords[i],
            }
            self.spines.append(
                Spine(
                    spine_meshes_list[i][0],
                    junction_center_coords[i],
                    self.center_coords[i],
                    spine_meshes_list[i][1],
                )
            )

    def add_spine_class(self) -> None:
        """Загружает или назначает морфологический класс каждому шипику.

        Входные данные: объекты `Spine` в `self.spines`.
        Выходные данные: обновлённое поле класса внутри объектов `Spine`.
        """
        for s in self.spines:
            s.add_spine_class()

    def add_spine_cluster(self) -> None:
        """Загружает или назначает кластер каждому шипику.

        Входные данные: объекты `Spine` в `self.spines`.
        Выходные данные: обновлённое поле кластера внутри объектов `Spine`.
        """
        for s in self.spines:
            s.add_spine_cluster()

    def get_spine_distance_points(self) -> List[Tuple[float, float, float]]:
        """Возвращает координаты точек крепления шипиков к дендриту, 
        используемые для расчёта расстояний между шипиками.

        Входные данные: объекты `Spine` в `self.spines`.
        Выходные данные: список трёхмерных координат точек крепления.
        """
        return [s.junction_center_coord for s in self.spines]

    def get_mesh_graph_distance_matrix(self) -> np.ndarray:
        """Вычисляет или возвращает кэшированную матрицу расстояний по мешу.
        Проецирует точки крепления на вершины меша и считает кратчайшие
        пути по рёбрам сетки дендрита.

        Входные данные: меш дендрита `self.mesh` и точки крепления шипиков.
        Выходные данные: квадратная матрица `mesh_graph` расстояний между
        шипиками.
        """
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
        """Вычисляет ненормированный профиль попарных расстояний.
        Считает среднее число соседей на шипик в последовательных
        интервалах расстояний и энтропию полученного профиля.

        Входные данные: матрица `mesh_graph`-расстояний между точками крепления.
        Выходные данные: `pair_distance_profile_r_values`,
        `pair_distance_profile_values`, `pair_distance_profile_entropy`.
        """
        mesh_graph_distance_matrix = self.get_mesh_graph_distance_matrix()
        distance_points = self.get_spine_distance_points()
        self.pair_distance_profile_r_values, self.pair_distance_profile_values = calculate_pair_distance_profile(
            distance_points,
            distance_matrix=mesh_graph_distance_matrix,
        )
        self.pair_distance_profile_entropy = calculate_Shannon_entropy(self.pair_distance_profile_values)

    def calculate_cluster_metrics(self) -> None:
        """Выполняет DBSCAN-кластеризацию шипиков по расстояниям вдоль меша.
        Подбирает `eps` по k-distance эвристике при необходимости и
        запускает DBSCAN с предвычисленной матрицей расстояний.

        Входные данные: точки крепления шипиков, морфологические метрики
        шипиков и матрица `mesh_graph`-расстояний.
        Выходные данные: `dbscan_labels`, `dbscan_eps`,
        `dbscan_min_samples`, `dbscan_noise`.
        """
        spine_metrics_dict_for_dbscan = {}
        for s in self.spines:
            spine_metrics_dict_for_dbscan[(s.center_coord[0], s.center_coord[1], s.center_coord[2])] = { 'Volume' : s.metrics['Volume'] }
        points = [[s.center_coord[0], s.center_coord[1], s.center_coord[2]] for s in self.spines]
        if len(points) == 0:
            self.dbscan_labels = np.array([], dtype=int)
            self.dbscan_eps = 0
            self.dbscan_min_samples = 0
            self.dbscan_noise = np.nan
            return
        mesh_graph_distance_matrix = self.get_mesh_graph_distance_matrix()

        dbscan_labels, dbscan_eps, dbscan_min_samples = dbscan(
            points,
            spine_metrics_dict_for_dbscan,
            'graphics/dbscan/' + self.name,
            distance_matrix=mesh_graph_distance_matrix,
            distance_label="mesh_graph",
        )
        self.dbscan_labels = dbscan_labels
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        if len(self.dbscan_labels) != len(points):
            self.dbscan_noise = np.nan
            return

        points = np.array([s.center_coord for s in self.spines])

        class_member_mask = (self.dbscan_labels == -1)
        filtered_points = points[class_member_mask]
        self.dbscan_noise = len(filtered_points)/len(points)

    def graph_analysis(self) -> None:
        """Строит полный взвешенный граф шипиков с весами `1 / d`,
        считает коэффициент кластеризации, сообщества и модульность.

        Входные данные: матрица `mesh_graph`-расстояний между шипиками.
        Выходные данные: `g_average_clustering`, `g_cluster_sizes`,
        `g_mean_cluster_size`, `g_characteristic_extent`, `g_modularity`.
        """
        distances = self.get_mesh_graph_distance_matrix()

        G = nx.Graph()
        n = distances.shape[0]
        G.add_nodes_from(range(n))

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

        # средний коэффициент группировки
        self.g_average_clustering = nx.average_clustering(G, weight='weight')

        # поиск сообществсли 
        # если установлен python-louvain, используем Louvain
        # иначе fallback на greedy modularity из networkx
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

        # количество точек в каждом кластере
        self.g_cluster_sizes = [len(comm) for comm in communities.values()]
        self.g_mean_cluster_size = np.mean(self.g_cluster_sizes)

        # протяженность кластера (среднее mesh_graph-расстояние между точками внутри кластера)
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

        # модульность разбиения
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
        """Вычисляет размер эффекта Cliff's delta для двух выборок.
        Сравнивает все попарные значения из двух групп и оценивает
        преимущественное направление различий.

        Входные данные: две числовые выборки.
        Выходные данные: число от -1 до 1 или `nan` при недостатке данных.
        """
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
        """Безопасно применяет статистический тест к двум выборкам.
        Фильтрует нечисловые значения, проверяет минимальный размер
        выборок и подавляет исключения теста.

        Входные данные: функция теста и две числовые выборки.
        Выходные данные: пара `(statistic, p_value)` или `(nan, nan)`.
        """
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
        """Вычисляет ранговую корреляцию Спирмена с проверкой входных данных.

        Входные данные: два числовых массива одинаковой длины.
        Выходные данные: пара `(rho, p_value)` или `(nan, nan)`.
        """
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
        """Вычисляет глобальный индекс Морана для заданной матрицы весов.
        Оценивает пространственную автокорреляцию значений признака
        между соседними шипиками

        Входные данные: числовой признак шипиков и матрица пространственных
        весов.
        Выходные данные: значение Moran's I или `nan`.
        """
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
        """Вычисляет глобальный коэффициент Гири для заданной матрицы весов.
        Оценивает локальные различия значений признака между
        соседними шипиками

        Входные данные: числовой признак шипиков и матрица пространственных
        весов.
        Выходные данные: значение Geary's C или `nan`.
        """
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
        """Вычисляет перестановочное p-значение для наблюдаемой статистики.
        Считает долю перестановок с абсолютным отклонением не меньше
        наблюдаемого с поправкой `+1`

        Входные данные: наблюдаемое значение статистики и список значений,
        полученных на перестановках.
        Выходные данные: перестановочное p-значение или `nan`.
        """
        permuted = np.asarray(permuted, dtype=float)
        permuted = permuted[np.isfinite(permuted)]
        if not np.isfinite(observed) or len(permuted) == 0:
            return float("nan")
        return float((np.sum(np.abs(permuted) >= abs(observed)) + 1) / (len(permuted) + 1))

    def _default_spatial_radius(self, distances: np.ndarray, min_samples: int = 3, percentile: float = 75) -> float:
        """Определяет локальный радиус анализа по k-distance эвристике.
        Берёт расстояние до `min_samples`-го ближайшего соседа для
        каждого шипика и возвращает заданный перцентиль этих расстояний

        Входные данные: матрица расстояний, `min_samples` и перцентиль.
        Выходные данные: радиус локальной окрестности.
        """
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
        """Заполняет пустые результаты пространственного анализа.

        Входные данные: состояние дендрита без валидных шипиков.
        Выходные данные: инициализированные поля результатов анализа.
        """
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
        """Формирует общий контекст для пространственных этапов анализа.
        Получает матрицу расстояний, DBSCAN-метки, расстояния до
        ближайших соседей, threshold-граф окрестностей и числовые
        морфологические признаки шипиков.

        Входные данные: опциональный радиус локальной окрестности.
        Выходные данные: словарь контекста, используемый последующими этапами.
        """
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
        """Вычисляет метрики плотности и распределения расстояний между шипиками.
        Считает статистики расстояний до ближайшего соседа,
        статистики всех попарных расстояний, threshold-граф и профиль попарных
        расстояний.

        Входные данные: опциональный радиус локальной окрестности; геометрия
        дендрита и шипиков берётся из состояния объекта.
        Выходные данные: `spatial_morphology_summary`,
        `spatial_morphology_spine_records` и кэш `_spatial_analysis_context`.
        """
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
        """Вычисляет характеристики локальных окрестностей шипиков.
        Для каждого шипика находит соседей в пределах радиуса,
        считает число соседей, graph degree, local clustering и средние
        морфологические признаки соседей.

        Входные данные: опциональный радиус локальной окрестности.
        Выходные данные: `local_neighborhood_summary` и
        `local_neighborhood_spine_records`.
        """
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
        """Вычисляет пространственную автокорреляцию морфологических признаков.
        Для каждого числового морфологического признака считает
        Moran's I, Geary's C, корреляцию Спирмена между значением шипика и
        средним значением у соседей, а также перестановочные p-значения.

        Входные данные: радиус локальной окрестности, число перестановок и
        seed генератора случайных чисел.
        Выходные данные: список `spatial_morphology_permutation_records`.
        """
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
        """Описывает DBSCAN-кластеры и сравнивает кластеризованные шипики с шумом.
        Вычисляет размеры и протяжённость кластеров, внутрикластерные
        расстояния, морфологические summaries по кластерам и непараметрические
        тесты clustered vs isolated.

        Входные данные: DBSCAN-метки, матрица расстояний и морфологические
        признаки шипиков.
        Выходные данные: `spatial_morphology_cluster_records`,
        `spatial_morphology_test_records` и кластерные поля в summary.
        """
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
        """Запускает полный актуальный пространственно-морфологический анализ.
        Последовательно выполняет анализ плотности, локальных
        окрестностей, автокорреляции и DBSCAN-кластеров.

        Входные данные: радиус локальной окрестности, число перестановок и
        seed генератора случайных чисел.
        Выходные данные: все summary-таблицы пространственного анализа.
        """
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
        """Совместимый alias для полного пространственно-морфологического анализа.

        Входные данные: радиус локальной окрестности, число перестановок и
        seed генератора случайных чисел.
        Действие: вызывает `calculate_comprehensive_spatial_analysis`.
        Выходные данные: все summary-таблицы пространственного анализа.
        """
        self.calculate_comprehensive_spatial_analysis(
            local_radius=local_radius,
            permutation_count=permutation_count,
            random_state=random_state,
        )

    def save_spatial_morphology_analysis(self) -> None:
        """Сохраняет подробные таблицы пространственно-морфологического анализа.

        Входные данные: рассчитанные поля пространственного анализа текущего
        дендрита.
        Выходные данные: CSV-файлы в текущей output-директории.
        """
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
        """Проверяет, можно ли значение сохранить как scalar-поле таблицы
        (проверяет принадлежность к scalar-типам Python/NumPy).

        Входные данные: произвольное значение метрики.
        Выходные данные: `True` для scalar-значений, иначе `False`.
        """
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
        """Преобразует значение метрики к JSON-совместимому виду
        (рекурсивно преобразует NumPy-типы в Python-типы и заменяет
        не конечные float-значения на `None`).

        Входные данные: произвольное значение, включая NumPy-типы, массивы,
        словари и списки.
        Выходные данные: JSON-совместимое значение.
        """
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
        """Преобразует NumPy scalar к стандартному Python scalar.

        Входные данные: scalar-значение Python или NumPy.
        Действие: вызывает `.item()` для NumPy scalar.
        Выходные данные: значение, пригодное для записи в таблицу.
        """
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()
        return value

    @staticmethod
    def _add_prefixed_metrics(record: Dict[str, Any], prefix: str, metrics: Dict[str, Any], skip_keys=None) -> None:
        """Добавляет словарь метрик в итоговую запись с префиксом.

        Входные данные: целевой словарь, префикс, словарь метрик и набор
        ключей, которые нужно пропустить.
        Действие: scalar-значения записывает отдельными колонками, сложные
        значения сериализует в JSON-поля.
        Выходные данные: обновлённый словарь `record`.
        """
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
        """Формирует единый вектор признаков дендритной ветви
        (объединяет scalar-метрики в один словарь, исключая подробные
        записи, предназначенные для отдельных таблиц).

        Входные данные: рассчитанные базовые, плотностные, окрестностные,
        автокорреляционные, кластерные и графовые метрики.
        Выходные данные: словарь признаков одной дендритной ветви.
        """
        if not hasattr(self, "spatial_morphology_summary") or not hasattr(self, "pair_distance_profile_values"):
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

    def save_structural_organization_vector(self) -> Dict[str, Any]:
        """Сохраняет единый вектор признаков дендритной ветви.

        Входные данные: результат `build_structural_organization_vector()`.
        Выходные данные: сохранённый словарь признаков текущего дендрита.
        """
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
        return record

    def calculate_spine_distance_matrices(self, methods=None, output_dir=None, pair_for_path=None):
        """Вычисляет диагностические матрицы расстояний между шипиками.
        Считает расстояния между точками крепления шипиков по
        выбранным методам; по умолчанию используется `mesh_graph`

        Входные данные: список методов расстояний, директория сохранения и
        опциональная пара шипиков для визуализации пути.
        Выходные данные: словарь результатов расстояний или сохранённые
        диагностические файлы при указанном `output_dir`.
        """
        if methods is None:
            methods = ("mesh_graph",)

        if not hasattr(self, "mesh"):
            raise ValueError("Surface distance methods require Dendrite.mesh. Create Dendrite with dendrite_meshes, not only from saved input JSON.")

        attachment_points = [s.junction_center_coord for s in self.spines]

        if output_dir is not None:
            return save_distance_method_comparison(
                dendrite_mesh=self.mesh,
                attachment_points=attachment_points,
                output_dir=output_dir,
                methods=methods,
                radius=self.radius,
                pair_for_path=pair_for_path,
            )

        return calculate_spine_distance_matrices(
            dendrite_mesh=self.mesh,
            attachment_points=attachment_points,
            methods=methods,
            radius=self.radius,
            pair_for_path=pair_for_path,
        )

    def save_spine_coords(self) -> None:
        """Сохраняет координаты центров и точек крепления шипиков в
        JSON-файл.

        Входные данные: глобальный накопитель `save_coords`.
        Выходные данные: файл `spine_coords.json` в output-директории.
        """
        # with open('spine_coords.json', 'w') as f:
        # with open('metrics/9009/spine_coords.json', 'w') as f:
        # with open('metrics/wt_old_st/spine_coords.json', 'w') as f:
        with open(output_path('spine_coords.json'), 'w') as f:
            json.dump(save_coords, f)

    def save_spine_metrics(self) -> None:
        """Сохраняет морфологические метрики шипиков в JSON.

        Входные данные: объекты `Spine` текущего дендрита и их словари метрик.
        Выходные данные: файл `spine_metrics.json` в output-директории.
        """
        for s in self.spines:
            save_spine_metric_dict[s.name] = s.metrics

        # with open('spine_metrics.json', 'w') as f:
        # with open('metrics/9009/spine_metrics.json', 'w') as f:
        # with open('metrics/wt_old_st/spine_metrics.json', 'w') as f:
        with open(output_path('spine_metrics.json'), 'w') as f:
            json.dump(save_spine_metric_dict, f)

    def save_init_metrics(self) -> None:
        """Сохраняет базовые метрики дендрита.

        Входные данные: поля `volume`, `length`, `radius`.
        Действие: записывает базовые геометрические метрики текущего дендрита
        в JSON-накопитель.
        Выходные данные: файл `dendr_metrics.json` в output-директории.
        """
        save_dendr_metric_dict[self.name] = {'Volume' : self.volume, 
                                             'Length' : self.length, 
                                             'Radius' : self.radius}
        
        with open(output_path('dendr_metrics.json'), 'w') as f:
            json.dump(save_dendr_metric_dict, f)

# Semantic alias
DendriteBranch = Dendrite
