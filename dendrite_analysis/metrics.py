from .dependencies import *
from .config import *
from spine_analysis.shape_metric.utils import _point_2_vec

def make_viewer(width: int = 600, height: int = 600) -> mp.Viewer:
    return mp.Viewer({"width": width, "height": height})


def polylines_to_line_set(polylines: Polylines) -> LineSet:
    output = []
    for line in polylines:
        for i in range(len(line) - 1):
            output.append((line[i], line[i + 1]))
    return output


def _add_line_set_to_viewer(viewer: mp.Viewer, lines: LineSet) -> None:
    viewer.add_lines(np.array([point_2_list(line[0]) for line in lines]),
                     np.array([point_2_list(line[1]) for line in lines]),
                     shading={"line_color": "red"})


def get_dendr_vecs(dendr_mesh: Polyhedron_3) -> List[Vector_3]: 
    dendr_vertices = dendr_mesh.vertices()
    dendr_vecs = []

    for vert in dendr_vertices:
        point = vert.point()
        vec = Vector_3(point.x(), point.y(), point.z())
        dendr_vecs.append(vec)
    
    return dendr_vecs


def get_sceleton_vecs(dendr_mesh: Polyhedron_3) -> List[Vector_3]:
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
        vec = _point_2_vec(line[0])
        sceleton_vecs.append(vec)
        if i==len(skeleton_line_set)-1:
            vec = _point_2_vec(line[1])
            sceleton_vecs.append(vec)
            
    return sceleton_vecs, skeleton_line_set


def fallback_dendrite_length(dendr_mesh: Polyhedron_3) -> float:
    from spine_analysis.mesh.utils import _mesh_to_v_f

    vertices, _ = _mesh_to_v_f(dendr_mesh)
    center = vertices.mean(axis=0)
    _, _, vh = np.linalg.svd(vertices - center, full_matrices=False)
    axis = vh[0]
    projections = (vertices - center) @ axis
    length = float(projections.max() - projections.min())
    print(f"  length (PCA fallback, no skeleton) = {length:.2f}")
    return length


def calculate_RadiusDendriteMetric(dendr_vecs: List[Vector_3], sceleton_vecs: List[Vector_3]) -> float:
    min_dists_list = []
    
    for sceleton_vec in sceleton_vecs:
        min_dist = math.sqrt((sceleton_vec - dendr_vecs[0]).squared_length())
        for dendr_vec in dendr_vecs:
            dist = math.sqrt((sceleton_vec - dendr_vec).squared_length())
            if dist < min_dist:
                min_dist = dist
        min_dists_list.append(min_dist)
               
    radius = np.mean(min_dists_list)
    print(f"  radius = {radius:.2f}")

    return radius


def calculate_LengthDendriteMetric(sceleton_vecs: List[Vector_3], skeleton_line_set) -> float:
    max_dist = 0
    
    for i in range(len(sceleton_vecs)):
        for j in range(i+1, len(sceleton_vecs)):
            dist = math.sqrt((sceleton_vecs[i] - sceleton_vecs[j]).squared_length())
            if dist > max_dist:
                max_dist = dist
    
    # make viewers
    # w = 600
    # h = 600
    # skeleton_viewer = make_viewer(w, h)
    # _add_line_set_to_viewer(skeleton_viewer, skeleton_line_set)
    # display(skeleton_viewer._renderer)
    
    length = max_dist
    print(f"  length = {length:.2f}")
    
    return length


def calculate_distance_matrix(points, cylindr_flag):
    if cylindr_flag:
        distance_matrix = squareform(pdist(points, lambda u, v: cylindrical_distance(u, v)))
    else: 
        distance_matrix = squareform(pdist(points))
    return distance_matrix


def calculate_all_dists(points, cylindr_flag = False):
    distance_matrix = calculate_distance_matrix(points, cylindr_flag)
    distance_vector = distance_matrix.flatten()  # или arr.ravel()
    distance_vector = distance_vector[distance_vector != 0]

    return distance_vector


def calculate_NNDist(points, volume: float, cylind_flag: bool = True):
    min_dist_list = []
    n = len(points)
    
    for i in range(len(points)-1):
        min_dist = float('inf')
        for j in range(i+1, len(points)):
            point = points[i]
            neigbour_point = points[j]

            if cylind_flag:
                dist = cylindrical_distance(point, neigbour_point)
            else: 
                dist = math.sqrt((point[0] - neigbour_point[0])**2 + (point[1] - neigbour_point[1])**2 + (point[2] - neigbour_point[2])**2)

            if dist < min_dist:
                min_dist = dist

        min_dist_list.append(min_dist)

    mean_dist = np.mean(min_dist_list)
    print("  nndist = ", mean_dist)
    
    density = n / volume
    # mean_dist_random = 0.5 * (volume / N) ** (1/3)
    mean_dist_random = (3/(4*math.pi*density)) ** (1/3)
    
    mean_dist_normalized = mean_dist / mean_dist_random
    print("  nndist_norm = ", mean_dist_normalized)
    
    return mean_dist, mean_dist_normalized


def create_spherical_shell(inner_radius, outer_radius, center=(0, 0, 0)):
    outer_sphere = trimesh.creation.icosphere(radius=outer_radius, subdivisions=3)
    if inner_radius == 0.0:
        outer_sphere.apply_translation(center)
        return outer_sphere
    inner_sphere = trimesh.creation.icosphere(radius=inner_radius, subdivisions=3)
    outer_sphere.apply_translation(center)
    inner_sphere.apply_translation(center)
    spherical_shell = outer_sphere.difference(inner_sphere)

    return spherical_shell


def create_cylindrical_shell(inner_radius, outer_radius, height):
    outer_cylinder = trimesh.creation.cylinder(radius=outer_radius, height=height, sections=50)
    inner_cylinder = trimesh.creation.cylinder(radius=inner_radius, height=height, sections=50)
    cylindrical_shell = outer_cylinder.difference(inner_cylinder)

    return cylindrical_shell


def get_cylindrical_shell(r_cylindr, R_cylindr, height):
    cylindrical_shell = create_cylindrical_shell(r_cylindr, R_cylindr, height)
    pv_cylindrical_shell = get_pv_mesh(cylindrical_shell)
    
    return pv_cylindrical_shell


def get_pv_mesh(mesh_trimesh):
    vertices = mesh_trimesh.vertices
    faces = np.hstack([[3] + face.tolist() for face in mesh_trimesh.faces])
    pv_mesh = pv.PolyData(vertices, faces)

    return pv_mesh


def calculate_shell_volume(pv_cylindrical_shell, r_sph, R_sph, r_cylindr, R_cylindr, height):                                 
    sphere_center_offset = (r_cylindr + R_cylindr) / 2.0
    
    # Создаем сферическую и цилиндрическую оболочки
    # print('\n r_sph = ', r_sph, ', R_sph = ', R_sph)
    # print(' r_cylindr = ', r_cylindr, ', R_cylindr = ', R_cylindr, ', height = ', height)
    if R_sph <= (R_cylindr - r_cylindr)/2:
        return 4/3 * math.pi * (R_cylindr**3 - r_cylindr**3)
    
    if r_sph >= height/2:
        spherical_shell = create_spherical_shell(r_sph, R_sph, center=(0, sphere_center_offset, -height/2))
    else:
        spherical_shell = create_spherical_shell(r_sph, R_sph, center=(0, sphere_center_offset, 0))
    
    pv_spherical_shell = get_pv_mesh(spherical_shell)
    intersection_mesh = pv_spherical_shell.boolean_intersection(pv_cylindrical_shell)
    intersection_volume = intersection_mesh.volume
    
    return intersection_volume


def calculate_PCF(points, volume: float, r_dendr: float, dr_dendr: float, height: float, cylind_flag: bool = True, dr: float = 0.5):
    dist_list = np.zeros((len(points), len(points)))

    for i, point in enumerate(points):
        for j, neigbour_point in enumerate(points):
            if cylind_flag:
                dist = cylindrical_distance(point, neigbour_point)
            else: 
                dist = math.sqrt((point[0] - neigbour_point[0])**2 + (point[1] - neigbour_point[1])**2 + (point[2] - neigbour_point[2])**2)
            dist_list[i][j] = dist
            
    r_max = np.amax(dist_list)
        
    r_values = np.arange(0, r_max, dr)
    pcf_values = np.zeros_like(r_values)
    pcf_values_norm = np.zeros_like(r_values)
    
    n = len(points)
    density = n / volume
    
    r_dendr = round(r_dendr, 2)
    dr_dendr = round(dr_dendr, 2)
    height = round(height, 2)
    # print(' r_cylindr = ', r_dendr, ', R_cylindr = ', r_dendr+dr_dendr, ', height = ', height)
    
    # cylindrical_shell = get_cylindrical_shell(r_dendr, r_dendr+dr_dendr, height)
    
    for k, r in enumerate(r_values):
        # shell_volume = (4/3) * np.pi * ((r+dr)**3 - r**3)
        # shell_volume = calculate_shell_volume(cylindrical_shell, r, r+dr, r_dendr, r_dendr+dr_dendr, height)
        # print('--> norm_volume = ', shell_volume)
        count = 0
        # для каждой точки подсчитываем количество точек, находящихся от данной на расстоянии от r до r + dr
        for i, point in enumerate(points):
            for j, neigbour_point in enumerate(points):
                if (i != j) & (r <= dist_list[i][j]) & (dist_list[i][j] < r + dr):
                    count+=1
                    
        pcf_values[k] = count / n
        # pcf_values_norm[k] = count / (n * shell_volume * density)
                
    
    # plt.plot(r_values, pcf_values, label="PCF")
    # plt.xlabel('Расстояние r')
    # plt.ylabel('g(r)')
    # plt.title('Функция парной корреляции без деления на объем оболочки и плотность')
    # plt.show()
    
    # plt.plot(r_values, pcf_values_norm, label="PCF")
    # plt.xlabel('Расстояние r')
    # plt.ylabel('g(r)')
    # plt.title('Функция парной корреляции')
    # plt.show()
    
    return r_values, pcf_values     


def calculate_Shannon_entropy(pcf_values: List[float]) -> float:
    S = np.sum(pcf_values)
    probabilities = pcf_values / S
    entropy = -np.sum(probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0]))
    # print("  Энтропия Шеннона для PCF: ", entropy)
    
    return entropy


def cylindrical_distance(point1, point2):
    r, z1, phi1  = point1
    r, z2, phi2 = point2
    
    delta_phi = min(abs(phi2 - phi1), 2 * np.pi - abs(phi2 - phi1)) 
    delta_z = z2 - z1  
    distance = np.sqrt((r * delta_phi) ** 2 + delta_z ** 2)  

    return distance


def param_from_kruskal(points, spine_metrics_dict, distance_matrix, max_dist=5, eps_range=(0.1, 8), min_samples_range=(2,10)):
    best_statistic = 0
    best_p_value = 10

    best_params = (2.5,2,0,0)
    
    eps = round(eps_range[0],1)
    while eps <= eps_range[1]:
        for min_samples in range(min_samples_range[0], min_samples_range[1] + 1):
            db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
            labels = db.fit_predict(distance_matrix)

            unique_labels = set(labels) 
            unique_labels = unique_labels - {-1}
            
            if len(unique_labels) <= 1:
                continue
            
            cluster_metrics_val = {}  # словарь [кластер: [метрика: набор значений в кластере]]

            for k in unique_labels:
                class_member_mask = (labels == k)
                class_points = points[class_member_mask]

                cluster_metrics_val[k] = {}
                cluster_metrics_val[k]['Volume'] = []

                for point in class_points:
                    key = (point[0], point[1], point[2])   
                    cluster_metrics_val[k]['Volume'].append(spine_metrics_dict[key]['Volume']) 
                        
            metric_clusters = [cluster_metrics_val[k]['Volume'] for k in unique_labels]

            statistic, p_value = kruskal(*metric_clusters)

            if statistic > best_statistic or p_value < best_p_value:
                # print(f"  + Kruskal-Wallis H-test for Volume: statistic = {statistic:.2f}, p-value = {p_value:.3f}")
                best_statistic = statistic
                best_p_value = p_value
                best_params = (eps, min_samples, best_statistic, best_p_value)
            # else:
            #     print(f"  - Kruskal-Wallis H-test for Volume: statistic = {statistic:.2f}, p-value = {p_value:.3f}")
            
        eps += 0.1
    # print (f'best_statistic = {best_statistic}, best_p_value = {best_p_value}')
    return best_params


def dbscan(points, spine_metrics_dict, cylindr_coords_flag = 0, save_path = None, eps = None, min_samples = None):
    print('\n-----------------------------------------------------')
    if cylindr_coords_flag:
        print('Кластеризация DBSCAN для цилиндрических координат')
    else:
        print('Кластеризация DBSCAN для классических трехмерных координат')
    print('-----------------------------------------------------')

    points = np.array(points)

    if cylindr_coords_flag:
        distance_matrix = squareform(pdist(points, lambda u, v: cylindrical_distance(u, v)))
    else:
        distance_matrix = squareform(pdist(points))

    if eps is None or min_samples is None:
        print('   Автоматический подбор параметров DBSCAN с помощью теста Крускала-Уоллиса...')
        eps, min_samples, statistic, p_value = param_from_dbcv(points, spine_metrics_dict, distance_matrix)
        # eps, min_samples, statistic, p_value = param_from_kruskal(points, spine_metrics_dict, distance_matrix)
        print(f'   ---> Выбранные параметры: eps={eps:.1f}, min_samples={min_samples}')
        if eps == 0:
            return np.array([]),0,0,0,0


    db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = db.fit_predict(distance_matrix)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f'Количество кластеров: {n_clusters}')
    
    unique_labels = set(labels)

    visualization_dbscan_clusters(points, labels, cylindr_coords_flag, save_path)

    # count_by_type, adjacency_matrix = cluster_analysis(points, labels, n_clusters, metric_bounds_dict, spine_metrics_dict, spine_vec_w_name, cylindr_coords_flag)

    return labels, eps, min_samples, statistic, p_value


def param_from_dbcv(points, spine_metrics_dict, distance_matrix):
    n_samples = len(points)
    # Определение диапазонов параметров
    # min_samples_range = range(2, 10)
    min_samples_range = [2,3,4,5]
    dists = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    # eps_range = np.percentile(dists, np.linspace(5, 30, 6))
    eps = 1 

    best_score = -np.inf
    best_eps = None
    best_min_samples = None

    # for eps in eps_range:
    while eps <= 5:
        print('eps = ', eps)
        for min_samples in min_samples_range:
            db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
            labels = db.fit_predict(distance_matrix)
            
            # Проверка числа кластеров
            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
            if n_clusters < 1:
                continue
                
            # Расчет DBCV
            score = dbcv_score(distance_matrix, labels, min_samples)
            if score > best_score:
                best_score = score
                best_eps = eps
                best_min_samples = min_samples
        eps += 0.5

    if best_eps is None:
        return 0, 0, -1, 0
    return best_eps, best_min_samples, best_score, 0


def dbcv_score(distance_matrix, labels, min_samples):
    n = distance_matrix.shape[0]
    core_distances = np.zeros(n)
    # Вычисление core distances
    for i in range(n):
        sorted_dists = np.sort(distance_matrix[i])
        core_distances[i] = sorted_dists[min_samples] if min_samples < n else sorted_dists[-1]
    
    # Матрица взаимной достижимости
    mr = np.maximum(core_distances[:, None], np.maximum(core_distances[None, :], distance_matrix))
    
    clusters = set(labels) - {-1}
    if not clusters:
        return -1.0
        
    validity_indices = []
    cluster_weights = []
    
    for cluster in clusters:
        indices = np.where(labels == cluster)[0]
        cluster_size = len(indices)
        if cluster_size < 2:
            continue
            
        # Компактность кластера
        sub_mr = mr[indices][:, indices]
        mst = csgraph.minimum_spanning_tree(sub_mr)
        max_edge = mst.max() if mst.nnz > 0 else 0
        
        # Разделимость кластера
        other_indices = np.where(labels != cluster)[0]
        if not other_indices.size:
            min_separation = np.inf
        else:
            separation_matrix = mr[indices][:, other_indices]
            min_separation = np.min(separation_matrix)
        
        # Расчет индекса валидности
        numerator = min_separation - max_edge
        denominator = max(min_separation, max_edge)
        validity_index = numerator / denominator if denominator > 0 else 0
        validity_indices.append(validity_index)
        cluster_weights.append(cluster_size)
    
    if not validity_indices:
        return -1.0
        
    # Взвешенное усреднение
    weights = np.array(cluster_weights) / sum(cluster_weights)
    return np.dot(validity_indices, weights)


def visualization_dbscan_clusters(points, labels, cylindr_coords_flag, save_path = None):
    plt.rcParams.update({'font.size': 14})

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')

    unique_labels = set(labels)
    # colors = plt.cm.Pastel1(np.linspace(0, 1, len(unique_labels)))
    palette = [
            [0.7, 0.0, 0.7],     # Фиолетовый
            [0.0, 0.8, 0.8],     # Бирюзовый
            [1.0, 0.65, 0.0],    # Оранжевый
            [0.5, 0.3, 0.9],     # Лавандовый
            [0.9, 0.5, 0.5],     # Розовый
            [0.2, 0.8, 0.2],     # Салатовый
            [1.0, 0.0, 0.0],     # Красный
            [0.0, 0.6, 0.0],     # Зелёный
            [0.0, 0.0, 1.0],     # Синий
            [1.0, 0.8, 0.0],     # Жёлтый
        ]

    # for k, col in zip(unique_labels, colors):
    for k in unique_labels:
        if k == -1:
            # Серый цвет для шума
            col = [0.3, 0.3, 0.3]
        else:
            col = palette[k % len(palette)]

        class_member_mask = (labels == k)
        class_points = points[class_member_mask]

        if cylindr_coords_flag:
            for i in range(len(class_points)):
                x = class_points[i][0] * math.cos(class_points[i][2])
                y = class_points[i][0] * math.sin(class_points[i][2])
                z = class_points[i][1]
                class_points[i] = (x,y,z)

        if k != -1:
            ax.scatter(class_points[:, 0], class_points[:, 1], class_points[:, 2], c=[col], label=f'Кластер {k}', s=100)
        else:  
            ax.scatter(class_points[:, 0], class_points[:, 1], class_points[:, 2], c=[col], label=f'Шум', s=100)


    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    ax.set_zlabel('Z', fontsize=14)
    plt.legend(fontsize=12)

    save_path = save_path.replace('.off', '')
    save_path = save_path.replace('Ab_output\\', '')
    save_path = save_path.replace('Wt_output\\', '')
    save_path = save_path.replace('\\pure_surface_mesh', '')

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"График сохранен в {save_path}")

    plt.show()


def basic_metrics_analysis(points, labels, spine_metrics_dict):
    unique_labels = set(labels)

    # cluster_statistics = ['mean', 'median', 'variance']
    cluster_statistics = ['mean', 'variance']
    cluster_statistics_val = {}  # словарь [кластер: [метрика: [стат. парам: значение]]]
    cluster_metrics_val = {}  # словарь [кластер: [метрика: набор значений в кластере]]
    for_df = {}

    cluster_means = {}
    cluster_variances = {}
    noise_means = {}
    noise_variances = {}

    metrics = list(spine_metrics_dict.values())[0].keys()

    for metric in metrics:
        cluster_means[metric] = []
        cluster_variances[metric] = []

    for k in unique_labels:
        class_member_mask = (labels == k)
        class_points = points[class_member_mask]

        cluster_statistics_val[k] = {}
        cluster_metrics_val[k] = {}
        for_df[k] = {}

        print('\nКластер: ', k)

        for metric in metrics:
            cluster_metrics_val[k][metric] = []
            cluster_statistics_val[k][metric] = {}
            for_df[k][metric] = []


        for point in class_points:
            key = (point[0], point[1], point[2])        
                
            for metric in spine_metrics_dict[key].keys():
                cluster_metrics_val[k][metric].append(spine_metrics_dict[key][metric])

        for metric in metrics:
            cluster_statistics_val[k][metric]['mean'] = np.mean(cluster_metrics_val[k][metric])
            for_df[k][metric].append(np.mean(cluster_metrics_val[k][metric]))
            # cluster_statistics_val[k][metric]['median'] = np.median(cluster_metrics_val[k][metric])  
            cluster_statistics_val[k][metric]['variance'] = np.var(cluster_metrics_val[k][metric]) 
            for_df[k][metric].append(np.var(cluster_metrics_val[k][metric]))
            if k != -1:
                cluster_means[metric].append(cluster_statistics_val[k][metric]['mean'])
                cluster_variances[metric].append(cluster_statistics_val[k][metric]['variance'])
            else:
                noise_means[metric] = cluster_statistics_val[k][metric]['mean']
                noise_variances[metric] = cluster_statistics_val[k][metric]['variance']

            # print(f'  - {metric}:')
            # for stat in cluster_statistics:
            #     print(f'    -> {stat}: {cluster_statistics_val[k][metric][stat]:.2f}')

        df = pd.DataFrame(cluster_statistics_val[k])
        print(df.T)

    kruskal_statistic, kruskal_p_value = kruskal_test(unique_labels, cluster_metrics_val, metrics)

    return cluster_means, cluster_variances, noise_means, noise_variances, kruskal_statistic, kruskal_p_value


def kruskal_test(unique_labels, cluster_metrics_val, metrics):
    if not metrics:
        metrics = ["JunctionArea", "Length", "Area", "Volume", "ConvexHullVolume"]
    unique_labels = unique_labels - {-1}

    kruskal_statistic = {}
    kruskal_p_value = {}
    for_df = {}

    for metric in metrics:
        for_df[metric] = []

    print("Статистический тест Kruskal-Wallis H-test:")
    for metric in metrics:
        metric_clusters = [cluster_metrics_val[k][metric] for k in unique_labels]
        f_stat, p_val = f_oneway(*metric_clusters)
        kruskal_statistic[metric] = f_stat
        kruskal_p_value[metric] = p_val
        for_df[metric].append(f_stat)
        for_df[metric].append(p_val)
        # print(f"  - Kruskal-Wallis H-test for {metric}: F-statistic = {f_stat:.2f}, p-value = {p_val:.3f}")
    
    df = pd.DataFrame(for_df)
    print(df.T)

    return kruskal_statistic, kruskal_p_value


def size_spine_analysis(points, labels, n_clusters, size_metric_bounds_dict, spine_metrics_dict):
    unique_labels = set(labels)

    print("Кластерный анализ больших шипиков")
    sum_kol_big_spines_dict = {}
    kol_big_spines_in_cluster_dict = {}
    kol_2big_spines_in_cluster_dict = {}
    kol_cluster_w_big_spines = {}
    kol_cluster_w_2big_spines = {}

    percent_big_spines_dict = {}
    percent_big_spines_in_noise = {}
    
    for metric in size_metric_bounds_dict.keys():
            sum_kol_big_spines_dict[metric] = 0
            kol_big_spines_in_cluster_dict[metric] = 0
            kol_2big_spines_in_cluster_dict[metric] = 0
            kol_cluster_w_big_spines[metric] = 0
            kol_cluster_w_2big_spines[metric] = 0
            percent_big_spines_dict[metric] = []
            percent_big_spines_in_noise[metric] = -1

    # for k, col in zip(unique_labels, colors):
    for k in unique_labels:
        # if k == -1:
        #     # Черный цвет для шума
        #     col = [0, 0, 0, 1]

        class_member_mask = (labels == k)
        class_points = points[class_member_mask]

        for_df = {}

        # словарь [метрика: количество шипиков, которые считаются большими относительно данной метрики]
        kol_big_spines_dict = {}
        for metric in size_metric_bounds_dict.keys():
            for_df[metric] = []
            kol_big_spines_dict[metric] = 0

        kol_spines_in_klaster = len(class_points) 

        print('\nКластер: ', k)

        for point in class_points:
            key = (point[0], point[1], point[2])        

            # анализ больших шипиков в кластерах
            for metric in size_metric_bounds_dict.keys():
                if spine_metrics_dict[key][metric] > size_metric_bounds_dict[metric]:
                    kol_big_spines_dict[metric] += 1
                    
                    # print('    point: ', point, f', {metric} = {spine_metrics_dict[key][metric]:.3f}',
                    #       f' => {metric} > {metric}_bound = {size_metric_bounds_dict[metric]:.3f}')
    

        print('  Итого:')
        for metric in kol_big_spines_dict.keys():
            # print(f'  - k_{metric} = {kol_big_spines_dict[metric]}')
            for_df[metric].append(kol_big_spines_dict[metric])

            sum_kol_big_spines_dict[metric] += kol_big_spines_dict[metric]
            if k != -1:
                percent_big_spines_dict[metric].append(round(kol_big_spines_dict[metric]/kol_spines_in_klaster,2)*100)
                kol_big_spines_in_cluster_dict[metric] += kol_big_spines_dict[metric]
                if kol_big_spines_dict[metric] > 0:
                    kol_cluster_w_big_spines[metric] += 1
                if kol_big_spines_dict[metric] > 1:
                    kol_2big_spines_in_cluster_dict[metric] += kol_big_spines_dict[metric]
                    kol_cluster_w_2big_spines[metric] += 1
            elif kol_spines_in_klaster:
                percent_big_spines_in_noise[metric] = round(kol_big_spines_dict[metric]/kol_spines_in_klaster,2)*100

        df = pd.DataFrame(for_df)
        print(df)


    for metric in kol_big_spines_dict.keys():
        for_df[metric] = []
    

    print("  Процент больших шипиков, которые попали в кластер: ")
    for metric in kol_big_spines_dict.keys():
        percent = round(kol_big_spines_in_cluster_dict[metric]/sum_kol_big_spines_dict[metric],2)*100
        for_df[metric].append(percent)
        # print(f'  - {metric}: {percent}')


    print("  Процент больших шипиков, которые образовали кластер с другими большими шипиками: ")
    for metric in kol_big_spines_dict.keys():
        if kol_big_spines_in_cluster_dict[metric] != 0:
            percent = round(kol_2big_spines_in_cluster_dict[metric]/kol_big_spines_in_cluster_dict[metric],2)*100
            for_df[metric].append(percent)
            # print(f'  - {metric}: {percent}')
        else:
            for_df[metric].append(0)

    print("  Процент кластеров, которые содержат большие шипики")
    for metric in kol_big_spines_dict.keys():
        percent = round(kol_cluster_w_big_spines[metric]/n_clusters,2)*100
        for_df[metric].append(percent)
        # print(f'  - {metric}: {percent}')

    print("  Процент кластеров, которые содержат группу больших шипиков (>1)")
    for metric in kol_big_spines_dict.keys():
        if kol_cluster_w_big_spines[metric] != 0:
            percent = round(kol_cluster_w_2big_spines[metric]/kol_cluster_w_big_spines[metric],2)*100
            for_df[metric].append(percent)
            # print(f'  - {metric}: {percent}')
        else:
            for_df[metric].append(0)
    
    print("  Средний процент больших шипиков в кластере от общего числа шипиков в кластере")
    for metric in kol_big_spines_dict.keys():
        # print(f'  - {metric}: {round(np.mean(percent_big_spines_dict[metric]),1)}')
        for_df[metric].append(round(np.mean(percent_big_spines_dict[metric]),1))

    print("  Процент больших шипиков в шуме от общего числа шипиков в шуме")
    for metric in kol_big_spines_dict.keys():
        # print(f'  - {metric}: {percent_big_spines_in_noise[metric]}')
        for_df[metric].append(percent_big_spines_in_noise[metric])

    df = pd.DataFrame(for_df)
    print(df)


def class_analysis(points, labels, spine_vec_w_name):
    unique_labels = set(labels)  
    unique_labels.discard(-1)

    class_member_mask = (labels != -1)
    filtered_points = points[class_member_mask]

    class_type_w_index = {'Undefined' : 0, 'Stubby' : 1, 'Mushroom' : 2, 'Thin' : 3, 'Filopodia' : 4}

    df = pd.read_csv('input/classification.csv')

    class_type = []
    indexes = []

    for point in filtered_points :
        key = (point[0], point[1], point[2])
        spine_name = spine_vec_w_name[key]
        if spine_name in df['Path'].values:
            type = df[df["Path"]==spine_name]["Group"].values[0]
        else:
            type = 'Undefined'

        class_type.append(type)
        indexes.append(class_type_w_index[type])

    data = pd.DataFrame({
        'x': filtered_points[:, 0] ,
        'y': filtered_points[:, 1] ,
        'z': filtered_points[:, 1] ,
        'class': class_type,
        'cluster': labels[class_member_mask],
        'class_index' : indexes
    })
    
    adjacency_matrix = np.zeros((len(classes_list), len(classes_list)))  

    for cluster in data['cluster'].unique():
        cluster_data = data[data['cluster'] == cluster]
        classes_in_cluster = cluster_data['class_index']

        for k, i in enumerate(classes_in_cluster.unique()):
            if i == 0:
                continue
            if classes_in_cluster.value_counts().get(i, 0) >= 2:
                adjacency_matrix[i-1,i-1] += 1
            for m, j in enumerate(classes_in_cluster.unique()):
                if j == 0:
                    continue
                if k != m:
                    adjacency_matrix[i-1, j-1] += 1

    count_by_type = {'Stubby' : [],  'Mushroom' : [], 'Thin' : [], 'Filopodia' : []}
    
    for k in unique_labels:
        if k == -1:
            continue
        class_member_mask = (labels == k)
        class_points = points[class_member_mask]

        for type in count_by_type.keys():
            count_by_type[type].append(0)

        for point in class_points:
            key = (point[0], point[1], point[2])
            spine_name = spine_vec_w_name[key]
            
            if spine_name in df['Path'].values:
                type = df[df["Path"]==spine_name]["Group"].values[0]

                count_by_type[type][k] += 1
            else:
                type = 'Undefined'



    # отрисовка графиков

    # class_types = range(1, 7)
    # colors = plt.cm.tab10(np.linspace(0, 1, len(class_types)))  # Используем цветовую карту tab10

    # fig = plt.figure(figsize=(7, 5))
    # ax = fig.add_subplot(111, projection='3d')

    # class_type_points = {}
    # for class_type in class_types:
    #     class_type_points[class_type] = []

    # kol_type_in_cluster = {}

    # for k in unique_labels:
    #     kol_type_in_cluster[k] = {}

    # print('points: ', points)
    # for k in unique_labels:
    #     class_member_mask = (labels == k)
    #     print('class_member_mask: ', class_member_mask)

    #     class_points = points[class_member_mask]
    #     print('class_points: ', class_points)

    #     for class_type in class_types:
    #         kol_type_in_cluster[k][class_type] = 0

    #     for point in class_points:
    #         key = (point[0], point[1], point[2])        

    #         spine_name = spine_vec_w_name[key]
    #         # print('spine_name: ', spine_name)
    #         df = pd.read_csv('clusterization_df.csv')
    #         if spine_name in df['Path'].values:
    #             class_type = int(df[df["Path"]==spine_name]["Group"].values[0])
    #         else:
    #             class_type = 6
    #         class_type_points[class_type].append(point)
    #         print('spine_name: ', spine_name)
    #         kol_type_in_cluster[k][class_type] += 1 

    #         # ax.scatter(point[0], point[1], point[2], color=colors[class_type - 1], label=f'Class type {class_type}')

    #     print(f'Кластер {k}: ')
    #     # df = pd.DataFrame(kol_type_in_cluster[k])
    #     print(kol_type_in_cluster[k])

    # for class_type in class_types:
    #     if class_type_points[class_type]:  # Проверяем, что список не пустой
    #         one_class_type_points = np.array(class_type_points[class_type])
    #         one_class_type_points = np.array(class_type_points[class_type])
    #         ax.scatter(one_class_type_points[:, 0], one_class_type_points[:, 1], one_class_type_points[:, 2], color=colors[class_type - 1], label=f'Class type {class_type}')

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.legend()
    # plt.show()

    return count_by_type, adjacency_matrix


def cluster_analysis(points, labels, spine_vec_w_name):
    unique_labels = set(labels)  
    unique_labels.discard(-1)

    class_member_mask = (labels != -1)
    filtered_points  = points[class_member_mask]

    df = pd.read_csv('input/clusterization.csv')

    class_type = []
    indexes = []

    for point in filtered_points :
        key = (point[0], point[1], point[2])
        spine_name = spine_vec_w_name[key]
        if spine_name in df['Path'].values:
            type = int(df[df["Path"]==spine_name]["Group"].values[0])
        else:
            type = 0

        class_type.append(type)
        indexes.append(type)

    data = pd.DataFrame({
        'x': filtered_points[:, 0] ,
        'y': filtered_points[:, 1] ,
        'z': filtered_points[:, 1] ,
        'class': class_type,
        'cluster': labels[class_member_mask],
        'class_index' : indexes
    })

    adjacency_matrix = np.zeros((len(clusters_list), len(clusters_list)))  

    for cluster in data['cluster'].unique():
        cluster_data = data[data['cluster'] == cluster]
        classes_in_cluster = cluster_data['class_index']

        for k, i in enumerate(classes_in_cluster.unique()):
            if i == 0:
                continue
            if classes_in_cluster.value_counts().get(i, 0) >= 2:
                adjacency_matrix[i-1,i-1] += 1
            for m, j in enumerate(classes_in_cluster.unique()):
                if j == 0:
                    continue
                if k != m:
                    adjacency_matrix[i-1, j-1] += 1

    count_by_type = { 1 : [], 2 : [], 3 : [], 4 : [], 5 : [], 6: [] } 
    
    for k in unique_labels:
        if k == -1:
            continue
        class_member_mask = (labels == k)
        class_points = points[class_member_mask]

        for type in count_by_type.keys():
            count_by_type[type].append(0)

        for point in class_points:
            key = (point[0], point[1], point[2])
            spine_name = spine_vec_w_name[key]
            
            if spine_name in df['Path'].values:
                type = int(df[df["Path"]==spine_name]["Group"].values[0])

                count_by_type[type][k] += 1 
            else:
                type = 0

    return count_by_type, adjacency_matrix


def calculate_Morans_I(points, metrics_dict, cylindr_flag = 0, threshold_distance = 2.0, name = None):
    if cylindr_flag:
        print("  Значение коэффициента автокорреляции Moran's I для цилиндрических координат:")
        dist_matrix = squareform(pdist(points, lambda u, v: cylindrical_distance(u, v)))
        # print('--- cylindr_dist_matrix\n', dist_matrix)
    else:
        print("  Значение коэффициента автокорреляции Moran's I для исходных координат:")
        dist_matrix = squareform(pdist(points))
    # print(f"  --> threshold_distance = {threshold_distance} ")

    if threshold_distance:
        W = (dist_matrix < threshold_distance).astype(int)
        np.fill_diagonal(W, 0)
    else:
        W = np.zeros_like(dist_matrix)
        np.fill_diagonal(W, 0)  # диагональ остается нулевой
        
        # Заполняем веса: W[i,j] = 1 / dist_matrix[i,j], если dist_matrix[i,j] > 0
        mask = (dist_matrix > 0)  
        W[mask] = 1.0 / dist_matrix[mask]

    morans_dict = {}

    for metric_name in metrics_dict.keys():
        metrics = metrics_dict[metric_name]
        n = len(metrics)
        mean_metric = np.mean(metrics)

        if W.shape != (n, n):
            print(f"Матрица W должна быть размера ({n}, {n}), но имеет размер {W.shape}")

        num = sum(W[i, j] * (metrics[i] - mean_metric) * (metrics[j] - mean_metric) 
                for i in range(n) for j in range(n))
        denom = np.sum((metrics - mean_metric) ** 2)
        W_sum = np.sum(W)
        I = (n / W_sum) * (num / denom)

        morans_dict[metric_name] = I

        # print(f"  - {metric_name}: {I:.3f}")

        if metric_name == "Volume":
            # print("    morans_dict[Volume] = ", morans_dict['Volume'])

            if cylindr_flag:
                dist_matrix = cdist(points, points, metric=cylindrical_distance)
                W_n = DistanceBand.from_array(dist_matrix, threshold=threshold_distance, binary=True)
                # print('  dist_matrix\n', dist_matrix)
                dense_matrix = W_n.full()[0]
                # print('  cylindr_W_n(dense_matrix)\n', dense_matrix)
                
            else:
                W_n = DistanceBand(points, threshold=threshold_distance, binary=True)
                dense_matrix = W_n.full()[0]
                # print('  W_n(dense_matrix)\n', dense_matrix)

            metrics = np.asarray(metrics)

            np.random.seed(42)

            moran = Moran(metrics, W_n, permutations=9999)
        
            # morans_dict_new[metric_name] = {
            #     'I': moran.I,
            #     'z_score': moran.z_rand,
            #     'p_value': moran.p_rand
            # }

            moran_values = [
                # moran.I if not np.isnan(moran.I) else 0,
                I,
                moran.z_rand if not np.isnan(moran.z_rand) else 0,
                moran.p_rand if not np.isnan(moran.p_rand) else 0
            ]
            
            morans_dict[metric_name] = moran_values

            print(f"     - Moran's I: {moran.I:.3f}, z_score: {moran.z_rand:.4f}, p-value: {moran.p_rand:.4f}")
            
            metrics_diff = metrics - mean_metric
            sum_weights = W.sum(axis=1)
            sum_weights_nonzero = np.where(sum_weights == 0, 1, sum_weights)
            y = (W @ metrics_diff) / sum_weights_nonzero
            y[sum_weights == 0] = 0

            # plt.scatter(metrics_diff, y, alpha=0.8, color=edge_colors['Wt'], s=100)
            # plt.plot(metrics_diff, 
            #         np.poly1d(np.polyfit(metrics_diff, y, 1))(metrics_diff), 
            #         color=edge_colors['Ab'], 
            #         linewidth=4)  # Было 2, стало 4
            # plt.axhline(0, color='grey', linestyle='--', linewidth=1.5)
            # plt.axvline(0, color='grey', linestyle='--', linewidth=1.5)
            # plt.xlabel('Отклонение значения от среднего', fontsize=18)
            # plt.ylabel('Пространственный лаг', fontsize=18)
            # plt.title(f'Диаграмма рассеяния Морана ({metric_name})', fontsize=18)
            # plt.grid(True, alpha=0.6)

            # if name is not None:
            #     save_path = 'graphics/moran/' + name
            #     save_path = save_path.replace('.off', '')
            #     save_path = save_path.replace('\\', '_')
            #     save_path = save_path.replace('_pure_surface_mesh', '')
            #     plt.savefig(save_path, dpi=300)
            #     print(f"График сохранен в {save_path}")       

            # plt.show()

    return morans_dict


def calculate_Getis_Ord_G(points, metrics_dict, threshold_distance=2.0, cylindr_flag=False, name=None):
    getis_ord_results = {}

    if cylindr_flag:
        dist_matrix = cdist(points, points, metric=cylindrical_distance)
        w = DistanceBand.from_array(dist_matrix, threshold=threshold_distance, binary=True)
        print('cylindr_flag: \n', w.full()[0])
    else:
        w = DistanceBand(points, threshold=threshold_distance, binary=True)
        print('cylindr_flag: \n', w.full()[0])

    for metric_name, values in metrics_dict.items():
        values = np.asarray(values)
        print('values: ', values)
        print(len(values))

        if np.all(w.full() == 0):
            print(f"Нет соседей для метрики '{metric_name}', возвращаем нули.")
            getis_ord_results[metric_name] = {
                'G': 0.0,
                'z_score': 0.0,
                'p_value': 1.0
            }
            continue

        np.random.seed(42)
        g = G(values, w, permutations=9999)

        getis_ord_results[metric_name] = [
            g.G if not np.isnan(g.G) else 0,
            g.z_sim if not np.isnan(g.z_sim) else 0,
            g.p_sim if not np.isnan(g.p_sim) else 0
        ]

        print(f"Глобальный Getis-Ord для метрики '{metric_name}':")
        print(f"  G = {g.G:.4f}, z = {g.z_sim:.4f}, p = {g.p_sim:.4f}")

        if metric_name == 'Volume':
            plt.figure(figsize=(6, 4))
            color = '#d73027' if g.p_sim < 0.05 and g.G > 0 else \
                    '#4575b4' if g.p_sim < 0.05 and g.G < 0 else \
                    '#999999'
            plt.bar([0], [g.G], color=color, edgecolor='black')
            plt.axhline(0, color='gray', linestyle='--', linewidth=1)
            plt.xticks([0], ['G'])
            plt.ylabel('Getis-Ord G', fontsize=14)
            plt.title(f'Глобальный Getis-Ord G для {metric_name}', fontsize=16)
            plt.grid(True, alpha=0.4)

            if name is not None:
                save_path = 'graphics/getis_ord/' + name
                save_path = save_path.replace('.off', '')
                save_path = save_path.replace('\\', '_')
                save_path = save_path.replace('_pure_surface_mesh', '')
                plt.savefig(save_path, dpi=300)
                print(f"График сохранен {save_path}")

            plt.show()

    return getis_ord_results


def visualize_pca_result(points, projected_points, w, heights, angles):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', label='Исходные точки', alpha=0.6)
    cos_angles = []
    sin_angles = []
    for angle in angles:
        cos_angles.append(math.cos(angle))
        sin_angles.append(math.sin(angle))
    ax.scatter(projected_points[:, 0], projected_points[:, 1], projected_points[:, 2], 
               c='red', label='Проекции на главную ось', alpha=0.8)
    start_point = np.mean(points, axis=0) - w * np.max(np.abs(heights))
    end_point = start_point + 2 * w * np.max(np.abs(heights))  # Удлиним ось для отображения
    ax.plot([start_point[0], end_point[0]], 
            [start_point[1], end_point[1]], 
            [start_point[2], end_point[2]], 
            c='gray', label='Главная ось', linewidth=2)
    
    ax.scatter(0, 0, heights, c='yellow', label='Проекция на повернутую главную ось (z)')
    ax.scatter(cos_angles, sin_angles, heights, c='green', label='Распределение точек по цилиндру', alpha=0.6)

    ax.set_title('Визуализация точек и главной оси')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


def get_points_from_vecs(spine_vecs: List[Vector_3]):
    points = []
    for vec in spine_vecs:
        points.append([vec.x(), vec.y(), vec.z()])

    return points


def get_2dim_coordinates(spine_vecs: List[Vector_3]):
    points = get_points_from_vecs(spine_vecs)
    points = np.array(points)

    # Центрирование точек вокруг начала координат 
    points_centered = points - np.mean(points, axis=0)

    # Применяем PCA
    pca = PCA(n_components=3, random_state=42)
    pca.fit(points_centered)

    # Получаем первую главную компоненту, вдоль которой вытянута фигура
    w = pca.components_[0]

    # Вычисляем координату вдоль оси w для каждой точки (высота h)
    heights = points_centered @ w  # скалярное произведение для проекции
    projected_points = np.outer(heights, w) + np.mean(points, axis=0)

    # Получаем вторую и третью компоненты PCA (перпендикулярные к w)
    u = pca.components_[1]
    v = pca.components_[2]

    # Проекция точек на плоскость, перпендикулярную w, для угловой координаты
    x_proj = points_centered @ u
    y_proj = points_centered @ v

    # Вычисляем угол θ для каждой точки в плоскости, перпендикулярной w
    angles = np.arctan2(y_proj, x_proj)

    # Возвращаем координаты в цилиндрическом формате (высота, угол)
    cylindrical_coords = np.column_stack((heights, angles))

    # visualize_pca_result(points, projected_points, w, heights, angles)

    return cylindrical_coords


def visualize_pca_w_radius(points_cartesian):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_cartesian[:, 0], points_cartesian[:, 1], points_cartesian[:, 2],
               c='red', label='Точки на цилиндрической поверхности', alpha=0.9)
    ax.set_title('Точки на цилиндрической поверхности')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


def get_cylindr_coord_from_2dim(coords, r):
    points = []
    points_cartesian = []
    for coord in coords:
        tuple_p = (r, coord[0], coord[1])
        points.append(tuple_p)

        h, theta = coord  # высота и угол
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = h
        points_cartesian.append([x, y, z])
    
    points_cartesian = np.array(points_cartesian)
    # visualize_pca_w_radius(points_cartesian)

    return points


def metric_distr(metric_dict):
    for metric in metric_dict.keys():
        plt.figure(figsize=(8, 6))
        sns.histplot(metric_dict[metric], kde=True, bins=30, color='lightpink', alpha=0.7)
        plt.title(f"Distribution of {metric}")
        plt.xlabel(metric)
        plt.ylabel("Frequency")
        plt.show()
