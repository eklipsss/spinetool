from .dependencies import *

from pathlib import Path


OUTPUT_DIR = Path("output")


def set_output_dir(path: str) -> Path:
    global OUTPUT_DIR
    OUTPUT_DIR = Path(path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def output_path(filename: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / filename


plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

plt.rcParams.update({'font.size': 18})

spine_metrics_name = [
    # "OpenAngle",
    # "CVD",
    # "AverageDistance",
    # "LengthVolumeRatio",
    # "LengthAreaRatio",
    # "JunctionArea",
    "Length",
    # "Area",
    "Volume",
    # "ConvexHullVolume",
    # "ConvexHullRatio",
]

spine_metric_classes = {
    "OpenAngleSpineMetric": OpenAngleSpineMetric,
    "CVDSpineMetric": CVDSpineMetric,
    "AverageDistanceSpineMetric": AverageDistanceSpineMetric,
    "LengthVolumeRatioSpineMetric": LengthVolumeRatioSpineMetric,
    "LengthAreaRatioSpineMetric": LengthAreaRatioSpineMetric,
    "JunctionAreaSpineMetric": JunctionAreaSpineMetric,
    "LengthSpineMetric": LengthSpineMetric,
    "AreaSpineMetric": AreaSpineMetric,
    "VolumeSpineMetric": VolumeSpineMetric,
    "ConvexHullVolumeSpineMetric": ConvexHullVolumeSpineMetric,
    "ConvexHullRatioSpineMetric": ConvexHullRatioSpineMetric,
    "JunctionCenterSpineMetric": JunctionCenterSpineMetric,
    "CenterSpineMetric": CenterSpineMetric,
}

save_spine_metric_dict = {}

save_coords = {}

save_dendr_metric_dict = {}
save_grouping_dendr_metric_dict = {}
save_cluster_dendr_metric_dict = {}
save_graph_dendr_metric_dict = {}
save_all_dendr_metric_dict = []
save_cluster_type_metric_dict = {}


def reset_saved_data() -> None:
    save_spine_metric_dict.clear()
    save_coords.clear()
    save_dendr_metric_dict.clear()
    save_grouping_dendr_metric_dict.clear()
    save_cluster_dendr_metric_dict.clear()
    save_graph_dendr_metric_dict.clear()
    save_all_dendr_metric_dict.clear()
    save_cluster_type_metric_dict.clear()

# class_dict = {'Undefined' : [], 'Stubby' : [], 'Mushroom' : [],  'Thin' : [], 'Filopodia' : []}
# cluster_dict = { 0 : [], 1 : [], 2 : [], 3 : [], 4 : [], 5 : [], 6: [] } 
# classes_list = ['Undefined', 'Stubby', 'Mashrooms', 'Thin',  'Filopodia']
# clusters_list = ['0', '1', '2', '3', '4', '5', '6']
class_dict = {'Stubby' : [], 'Mushroom' : [],  'Thin' : [], 'Filopodia' : []}
cluster_dict = { 1 : [], 2 : [], 3 : [], 4 : [], 5 : [], 6: [] } 
classes_list = ['Stubby', 'Mashrooms', 'Thin',  'Filopodia']
clusters_list = ['1', '2', '3', '4', '5', '6']

class_group_colors = {
    # 'Undefined' : '#a0a0a0', 
    'Stubby': '#fb8072', 
    'Mushroom': '#fdb462', 
    'Thin': '#ffffb3', 
    'Filopodia' : '#8dd3c7'}
cluster_group_colors = {
    # 0 : '#a0a0a0', 
    1: '#fb8072', 
    2: '#fdb462', 
    3: '#ffffb3', 
    4 : '#8dd3c7', 
    5 : '#80b1d3', 
    6: '#c6bada'}

# class_group_colors = {
#     # 'Undefined' : '#a0a0a0', 
#     'Stubby': "#d15c4f", 
#     'Mushroom': "#dda05b", 
#     'Thin': "#d6d64d", 
#     'Filopodia' : "#57ceba"}
# cluster_group_colors = {
#     # 0 : '#a0a0a0', 
#     1: "#d15c4f", 
#     2: '#dda05b', 
#     3: '#d6d64d', 
#     4 : '#57ceba', 
#     5 : "#65a4d1", 
#     6: "#ac92d4"}

class_group_edge_colors = {
    # 'Undefined': '#808080',
    'Stubby': '#e62e2e', # Более насыщенный красный
    'Mushroom': '#e67e22',  # Более насыщенный оранжевый
    'Thin': '#e6e600',  # Более насыщенный жёлтый
    'Filopodia': '#2e8b57',  # Более насыщенный зелёный
}

cluster_group_edge_colors = {
    # 0: '#808080',  # Более насыщенный фиолетовый
    1: '#e62e2e',  # Более насыщенный жёлтый
    2: '#e67e22',  # Более насыщенный оранжевый
    3: '#e6e600',  # Более насыщенный красный
    4: '#2e8b57',  # Более насыщенный зелёный
    5: '#1f78b4',  # Более насыщенный синий
    6: '#6a5acd'
}

# class_group_edge_colors = {
#     # 'Undefined': '#808080',
#     'Stubby': "#b30000", # Более насыщенный красный
#     'Mushroom': '#b95700',  # Более насыщенный оранжевый
#     'Thin': "#d4a301",  # Более насыщенный жёлтый
#     'Filopodia': '#008539',  # Более насыщенный зелёный
# }
# cluster_group_edge_colors = {
#     # 0: '#808080',  # Более насыщенный фиолетовый
#     1: "#b30000",  # Более насыщенный жёлтый
#     2: "#b95700",  # Более насыщенный оранжевый
#     3: "#d4a301",  # Более насыщенный красный
#     4: "#008539",  # Более насыщенный зелёный
#     5: "#005f9e",  # Более насыщенный синий
#     6: "#13008f"
# }

group_colors = {'Wt': '#c3e485', 'Ab': '#ffcde6'}
edge_colors = {'Wt': '#8cbd3a', 'Ab': '#fd89c3'}

# group_colors = {'Wt': "#a6dd40", 'Ab': "#ff89c4"}
