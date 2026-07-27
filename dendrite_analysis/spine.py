from .dependencies import *
from .config import *

class Spine:
    name: str
    mesh: Polyhedron_3

    junction_center_coord: Tuple[float] # координаты центра области крепления шипика к дендриту - вспомогательная переменная, в вычислении метрик не участвует
    center_coord: Tuple[float] # координаты центра шипика к дендриту 

    metrics: Dict[str, float] = {}

    class_type: str
    cluster_type: int

    def __init__(self, name: str,  
                 junction_center_coord: Any,
                 center_coord: Any,
                 spine_mesh: Polyhedron_3) -> None:
        """Создаёт объект шипика и рассчитывает его морфологические метрики.

        Входные данные: имя шипика, точка крепления, геометрический центр и
        mesh шипика.
        Выходные данные: объект `Spine` с заполненным словарём `metrics`.
        """
        print('Spine init')

        self.metrics = {}

        self.name = name
        if spine_mesh is None:
            raise ValueError("Spine requires spine_mesh; loading old precomputed spine metrics is disabled.")
        self.mesh = spine_mesh

        self.junction_center_coord = junction_center_coord
        self.center_coord = center_coord
        self.calculate_metrics()

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
