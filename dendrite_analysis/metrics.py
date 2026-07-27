from .dependencies import *
from spine_analysis.shape_metric.utils import _point_2_vec
from pathlib import Path


def make_viewer(width: int = 600, height: int = 600) -> mp.Viewer:
    """Создаёт meshplot-viewer заданного размера.

    Входные данные: ширина и высота окна.
    Выходные данные: viewer для визуализации mesh/линий.
    """
    return mp.Viewer({"width": width, "height": height})


def polylines_to_line_set(polylines: Polylines) -> LineSet:
    """Преобразует набор CGAL-полилиний в список отрезков.

    Входные данные: объект `Polylines`.
    Выходные данные: список отрезков `(point_start, point_end)`.
    """
    output = []
    for line in polylines:
        for i in range(len(line) - 1):
            output.append((line[i], line[i + 1]))
    return output


def _add_line_set_to_viewer(viewer: mp.Viewer, lines: LineSet) -> None:
    """Добавляет набор линий красным цветом в meshplot-viewer.

    Входные данные: viewer и список отрезков.
    Выходные данные: отсутствуют.
    """
    viewer.add_lines(
        np.array([point_2_list(line[0]) for line in lines]),
        np.array([point_2_list(line[1]) for line in lines]),
        shading={"line_color": "red"},
    )


def get_dendr_vecs(dendr_mesh: Polyhedron_3) -> List[Vector_3]:
    """Извлекает вершины дендритного mesh как CGAL-векторы `Vector_3`.

    Входные данные: `Polyhedron_3` mesh дендрита.
    Выходные данные: список `Vector_3`.
    """
    dendr_vecs = []
    for vert in dendr_mesh.vertices():
        point = vert.point()
        dendr_vecs.append(Vector_3(point.x(), point.y(), point.z()))
    return dendr_vecs


def get_sceleton_vecs(dendr_mesh: Polyhedron_3) -> Tuple[List[Vector_3], LineSet]:
    """Строит CGAL-skeleton дендрита и возвращает его точки.

    Входные данные: закрытый несамопересекающийся `Polyhedron_3`.
    Выходные данные: список skeleton-векторов и список skeleton-отрезков.
    """
    if not bool(dendr_mesh.is_closed()):
        raise RuntimeError(
            "surface_mesh_skeletonization requires a closed (watertight) mesh; "
            "the input mesh has open boundaries."
        )
    if bool(does_self_intersect(dendr_mesh)):
        raise RuntimeError(
            "surface_mesh_skeletonization requires a non-self-intersecting mesh; "
            "the input mesh has self-intersecting faces."
        )

    skeleton_polylines = Polylines()
    correspondence_polylines = Polylines()
    surface_mesh_skeletonization(dendr_mesh, skeleton_polylines, correspondence_polylines)

    skeleton_line_set = polylines_to_line_set(skeleton_polylines)
    sceleton_vecs = []
    for i, line in enumerate(skeleton_line_set):
        sceleton_vecs.append(_point_2_vec(line[0]))
        if i == len(skeleton_line_set) - 1:
            sceleton_vecs.append(_point_2_vec(line[1]))

    return sceleton_vecs, skeleton_line_set


def fallback_dendrite_length(dendr_mesh: Polyhedron_3) -> float:
    """Оценивает длину дендрита через PCA при недоступном skeleton.
    Извлекает вершины, строит главную PCA-ось и измеряет размах
    проекций вершин на эту ось.

    Входные данные: `Polyhedron_3` mesh дендрита.
    Выходные данные: численная оценка длины.
    """
    from spine_analysis.mesh.utils import _mesh_to_v_f

    vertices, _ = _mesh_to_v_f(dendr_mesh)
    center = vertices.mean(axis=0)
    _, _, vh = np.linalg.svd(vertices - center, full_matrices=False)
    axis = vh[0]
    projections = (vertices - center) @ axis
    length = float(projections.max() - projections.min())
    print(f"  length (PCA fallback, no skeleton) = {length:.2f}")
    return length


def calculate_RadiusDendriteMetric(
    dendr_vecs: List[Vector_3],
    sceleton_vecs: List[Vector_3],
) -> float:
    """Вычисляет радиус дендрита по расстояниям skeleton-точек до поверхности.
    Для каждой skeleton-точки ищет ближайшую вершину поверхности и
    усредняет эти расстояния.

    Входные данные: векторы вершин mesh и векторы skeleton.
    Выходные данные: средний радиус дендрита.
    """
    min_dists_list = []
    for sceleton_vec in sceleton_vecs:
        min_dist = math.sqrt((sceleton_vec - dendr_vecs[0]).squared_length())
        for dendr_vec in dendr_vecs:
            dist = math.sqrt((sceleton_vec - dendr_vec).squared_length())
            if dist < min_dist:
                min_dist = dist
        min_dists_list.append(min_dist)

    radius = float(np.mean(min_dists_list))
    print(f"  radius = {radius:.2f}")
    return radius


def calculate_LengthDendriteMetric(
    sceleton_vecs: List[Vector_3],
    skeleton_line_set: LineSet,
) -> float:
    """Вычисляет длину дендрита по skeleton-точкам.
    Ищет максимальное евклидово расстояние между skeleton-точками.

    Входные данные: список skeleton-векторов и skeleton-line-set.
    Выходные данные: оценка длины дендрита.
    """
    max_dist = 0.0
    for i in range(len(sceleton_vecs)):
        for j in range(i + 1, len(sceleton_vecs)):
            dist = math.sqrt((sceleton_vecs[i] - sceleton_vecs[j]).squared_length())
            if dist > max_dist:
                max_dist = dist

    length = float(max_dist)
    print(f"  length = {length:.2f}")
    return length


def calculate_distance_matrix(points, distance_matrix=None):
    """Возвращает матрицу расстояний между точками.
    Если матрица передана, приводит её к `numpy.ndarray`, 
    иначе считает евклидовы попарные расстояния.

    Входные данные: массив точек и опциональная заранее вычисленная матрица
    расстояний.
    Выходные данные: квадратная матрица расстояний.
    """
    if distance_matrix is not None:
        return np.asarray(distance_matrix, dtype=float)
    return squareform(pdist(points))


def calculate_pair_distance_profile(points, dr: float = 0.5, distance_matrix=None):
    """Вычисляет ненормированный профиль попарных расстояний.
    Для каждого интервала `[r, r + dr)` считает среднее число
    соседних шипиков на один шипик.

    Входные данные: точки, ширина бина и опциональная матрица расстояний.
    Выходные данные: массив радиусов и массив значений профиля.
    """
    dist_list = calculate_distance_matrix(points, distance_matrix=distance_matrix)
    if dist_list.size == 0:
        return np.array([]), np.array([])

    finite_distances = dist_list[np.isfinite(dist_list)]
    if len(finite_distances) == 0:
        return np.array([]), np.array([])

    r_values = np.arange(0, np.amax(finite_distances), dr)
    pair_distance_profile_values = np.zeros_like(r_values)
    n = len(points)

    for k, r in enumerate(r_values):
        count = 0
        for i, _point in enumerate(points):
            for j, _neighbour_point in enumerate(points):
                if i != j and r <= dist_list[i][j] < r + dr:
                    count += 1
        pair_distance_profile_values[k] = count / n

    return r_values, pair_distance_profile_values


def calculate_Shannon_entropy(values: List[float]) -> float:
    """Вычисляет энтропию Шеннона для неотрицательного профиля.
    Нормирует значения на сумму и считает `-sum(p_i * log(p_i))`,
    при нулевой сумме возвращает `0.0`.

    Входные данные: массив значений профиля.
    Выходные данные: значение энтропии.
    """
    values = np.asarray(values, dtype=float)
    total = np.sum(values)
    if not np.isfinite(total) or total <= 0:
        return 0.0
    probabilities = values / total
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log(probabilities)))


def dbscan(
    points,
    spine_metrics_dict,
    save_path=None,
    eps=None,
    min_samples=None,
    distance_matrix=None,
    distance_label=None,
):
    """Запускает DBSCAN-кластеризацию шипиков.
    Использует переданную матрицу расстояний или считает её,
    подбирает параметры через k-distance эвристику при необходимости,
    выполняет DBSCAN с `metric="precomputed"` и сохраняет визуализацию.

    Входные данные: координаты точек, словарь морфологических метрик шипиков,
    путь сохранения графика, параметры `eps`/`min_samples`, опциональная
    предвычисленная матрица расстояний и подпись типа расстояний.
    Выходные данные: метки кластеров, `eps` и `min_samples`.
    """
    print("\n-----------------------------------------------------")
    if distance_label is not None:
        print(f"Кластеризация DBSCAN для расстояний: {distance_label}")
    else:
        print("Кластеризация DBSCAN для классических трехмерных координат")
    print("-----------------------------------------------------")

    points = np.asarray(points)
    distance_matrix = calculate_distance_matrix(points, distance_matrix=distance_matrix)
    if not np.isfinite(distance_matrix).all():
        finite = distance_matrix[np.isfinite(distance_matrix)]
        fill_value = float(finite.max() * 10.0) if len(finite) else 1e12
        distance_matrix = np.nan_to_num(
            distance_matrix,
            nan=fill_value,
            posinf=fill_value,
            neginf=fill_value,
        )

    if eps is None or min_samples is None:
        print("   Автоматический подбор параметров DBSCAN по k-distance heuristic...")
        eps, min_samples = param_from_k_distance(
            distance_matrix,
            target_min_samples=3,
            percentile=75,
        )
        print(f"   ---> Выбранные параметры: eps={eps:.4f}, min_samples={min_samples}")
        if eps == 0:
            return np.array([]), 0, 0

    model = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = model.fit_predict(distance_matrix)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Количество кластеров: {n_clusters}")
    visualization_dbscan_clusters(points, labels, save_path)

    return labels, eps, min_samples


def _finite_upper_distances(distance_matrix):
    """Возвращает конечные положительные расстояния верхнего треугольника.
    Извлекает элементы выше диагонали и фильтрует `nan`, `inf` и нули.

    Входные данные: квадратная матрица расстояний.
    Выходные данные: одномерный массив расстояний.
    """
    if distance_matrix.size == 0:
        return np.array([], dtype=float)
    dists = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    dists = dists[np.isfinite(dists) & (dists > 0)]
    return dists.astype(float)


def param_from_k_distance(distance_matrix, target_min_samples=3, percentile=75):
    """Подбирает параметры DBSCAN по k-distance эвристике.
    Берёт расстояние до `min_samples`-го ближайшего соседа для
    каждой точки и возвращает заданный перцентиль этих расстояний как `eps`.

    Входные данные: матрица расстояний, целевое `min_samples` и перцентиль.
    Выходные данные: `(eps, min_samples)`.
    """
    n_samples = distance_matrix.shape[0]
    if n_samples < 2:
        return 0, 0

    min_samples = min(target_min_samples, n_samples)
    neighbor_rank = min_samples
    if neighbor_rank >= n_samples:
        neighbor_rank = n_samples - 1

    clean_distances = np.asarray(distance_matrix, dtype=float).copy()
    clean_distances[~np.isfinite(clean_distances)] = np.inf
    np.fill_diagonal(clean_distances, 0.0)

    sorted_distances = np.sort(clean_distances, axis=1)
    kth_neighbor_distances = sorted_distances[:, neighbor_rank]
    kth_neighbor_distances = kth_neighbor_distances[
        np.isfinite(kth_neighbor_distances) & (kth_neighbor_distances > 0)
    ]

    if len(kth_neighbor_distances) == 0:
        finite_distances = _finite_upper_distances(distance_matrix)
        eps = float(np.percentile(finite_distances, percentile)) if len(finite_distances) else 0.0
    else:
        eps = float(np.percentile(kth_neighbor_distances, percentile))

    if not np.isfinite(eps) or eps <= 0:
        return 0, 0

    print(
        f"   k-distance: min_samples={min_samples}, "
        f"neighbor_rank={neighbor_rank}, percentile={percentile}, eps={eps:.4f}"
    )
    return eps, min_samples


def visualization_dbscan_clusters(points, labels, save_path=None):
    """Визуализирует DBSCAN-кластеры в 3D.

    Входные данные: координаты точек, метки DBSCAN и путь сохранения.
    Выходные данные: отображённый график; при `save_path` сохраняется файл.
    """
    plt.rcParams.update({"font.size": 14})

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")

    palette = [
        [0.7, 0.0, 0.7],
        [0.0, 0.8, 0.8],
        [1.0, 0.65, 0.0],
        [0.5, 0.3, 0.9],
        [0.9, 0.5, 0.5],
        [0.2, 0.8, 0.2],
        [1.0, 0.0, 0.0],
        [0.0, 0.6, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.8, 0.0],
    ]

    for k in set(labels):
        if k == -1:
            color = [0.3, 0.3, 0.3]
            label = "Шум"
        else:
            color = palette[k % len(palette)]
            label = f"Кластер {k}"

        class_member_mask = labels == k
        class_points = points[class_member_mask]
        ax.scatter(
            class_points[:, 0],
            class_points[:, 1],
            class_points[:, 2],
            c=[color],
            label=label,
            s=100,
        )

    ax.set_xlabel("X", fontsize=14)
    ax.set_ylabel("Y", fontsize=14)
    ax.set_zlabel("Z", fontsize=14)
    plt.legend(fontsize=12)

    if save_path is not None:
        save_path = save_path.replace(".off", "")
        save_path = save_path.replace("Ab_output\\", "")
        save_path = save_path.replace("Wt_output\\", "")
        save_path = save_path.replace("\\pure_surface_mesh", "")
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"График сохранен в {save_path}")

    plt.show()
