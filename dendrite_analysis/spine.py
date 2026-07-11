from .dependencies import *
from .config import *

class Spine:
    name: str
    mesh: Polyhedron_3

    junction_center_coord: Tuple[float] # координаты центра области крепления шипика к дендриту - вспомогательная переменная, в вычислении метрик не участвует
    center_coord: Tuple[float] # координаты центра шипика к дендриту 
    center_coord_c: Tuple[float] # координаты центра области крепления шипика к дендриту в цилиндрических координатах (junction_center_coord в цк)

    metrics: Dict[str, float] = {}

    class_type: str
    cluster_type: int

    def __init__(self, name: str,  
                #  junction_center_coord: Tuple[float],
                #  center_coord: Tuple[float],
                 junction_center_coord: Any,
                 center_coord: Any,
                 center_coord_cylindr: Any, 
                 spine_mesh: Polyhedron_3 = None) -> None:
        print('Spine init')

        self.metrics = {}

        self.name = name
        if spine_mesh is not None:
            self.mesh = spine_mesh

        self.junction_center_coord = junction_center_coord
        self.center_coord = center_coord
        if center_coord_cylindr != False:
            self.center_coord_c = center_coord_cylindr

        # junction_center_klass = globals()['JunctionCenterSpineMetric'] 
        # center_klass = globals()['CenterSpineMetric'] 

        # # середина области крепления шипика
        # junction_center_vec = junction_center_klass(spine_mesh)._value
        # self.junction_center_coord = (junction_center_vec.x(), junction_center_vec.y(), junction_center_vec.z())  # tuple, тк неизменяемый
        
        # # середина шипика
        # center_vec = center_klass(spine_mesh)._value
        # self.center_coord = (center_vec.x(), center_vec.y(), center_vec.z())  # tuple, тк неизменяемый

        # two_dim_points = get_2dim_coordinates([self.junction_center_coord])
        # center_coord_cylindr = get_cylindr_coord_from_2dim(two_dim_points, dendr_radius)[0]

        if spine_mesh is not None:
            self.calculate_metrics() # вычисление метрик
        else:
            self.load_metrics() # загрузка из файла

        # self.add_spine_class()
        # self.add_spine_cluster()

    def load_metrics(self) -> None:
        # with open('metrics/spine_metrics.json', 'r') as f:
        # with open('metrics/9009/spine_metrics.json', 'r') as f:
        # with open('metrics/wt_old_st/spine_metrics.json', 'r') as f:
        with open('input/spine_metrics.json', 'r') as f:
            loaded_dict = json.load(f)
        self.metrics = loaded_dict[self.name]

    def calculate_metrics(self) -> None:
        for metric_name in spine_metrics_name:
            class_name = metric_name + "SpineMetric"
            metric_class = spine_metric_classes.get(class_name)
            
            if metric_class is not None:
                self.metrics[metric_name] = metric_class(self.mesh)._value
            else:
                raise ValueError(f"Метрика '{metric_name}' не найдена в spine_analysis.shape_metric")
            
            # self.metrics[metric_name] = globals()[metric_name + 'SpineMetric'](self.mesh)._value

    def add_spine_class(self) -> None:
        class_df = pd.read_csv('input/classification.csv')

        if self.name in class_df['Path'].values:
            self.class_type = class_df[class_df["Path"]==self.name]["Group"].values[0]
        else:
            self.class_type = 'Undefined' 
    
    def add_spine_cluster(self) -> None:
        cluster_df = pd.read_csv('input/clusterization.csv')

        if self.name in cluster_df['Path'].values:
            self.cluster_type = int(cluster_df[cluster_df["Path"]==self.name]["Group"].values[0])
        else:
            self.cluster_type = 0 
