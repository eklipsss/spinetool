from .dependencies import *
from .config import *
from .comparison import DendrComparisonMixin
from .dendrite_type import DendrType

class DendrAnalysis(DendrComparisonMixin):
    ab: DendrType
    wt: DendrType
    ab9009: DendrType

    def __init__(self, all_types_dataset: Any = None, flag_9009: Any = None) -> None:
        print('DendrType init')
        if all_types_dataset is not None:
            self.ab = DendrType('Ab', all_types_dataset[0])
            self.wt = DendrType('Wt', all_types_dataset[1])
        else:
            self.ab = DendrType('Ab')
            self.wt = DendrType('Wt')
            if flag_9009 is not None:
                self.ab9009 = DendrType('Ab+9009')

        # with open('spine_metrics.json', 'w') as f:
        #     json.dump(save_spine_metric_dict, f)
        # with open('dendr_metrics.json', 'w') as f:
        #     json.dump(save_dendr_metric_dict, f)
        # with open('grouping_dendr_metrics.json', 'w') as f:
        #     json.dump(save_grouping_dendr_metric_dict, f) 

    def add_spine_class(self) -> None:
        for type in [self.ab, self.wt]:
            type.add_spine_class()

    def add_spine_cluster(self) -> None:
        for type in [self.ab, self.wt]:
            type.add_spine_cluster()

    def load_grouping_metrics(self) -> None:
        self.ab.load_grouping_metrics()
        self.wt.load_grouping_metrics()

    def load_cluster_metrics(self) -> None:
        self.ab.load_cluster_metrics()
        self.wt.load_cluster_metrics()

    def calculate_grouping_metrics(self) -> None:
        self.ab.calculate_grouping_metrics()
        self.wt.calculate_grouping_metrics()

        # plt.figure(figsize=(10, 6))
        # plt.plot(self.ab.r_values_c[:self.ab.pcf_len_c], self.ab.average_pcf_values_c, label='pair-distance profile', color=group_colors['Ab'], linewidth=3)
        # plt.plot(self.wt.r_values_c[:self.wt.pcf_len_c], self.wt.average_pcf_values_c, label='pair-distance profile', color=group_colors['Wt'], linewidth=3)
        # plt.xlabel('Попарные расстояния', fontsize = 18)
        # plt.ylabel('Pair-distance profile', fontsize = 18)
        # plt.legend(fontsize = 16)
        # plt.xticks(fontsize = 16)
        # plt.yticks(fontsize = 16)
        # title = 'Усредненный график pair-distance profile - Ab-Wt (цилиндрические координаты)'
        # plt.title(title, fontsize = 18)

        # output_filename = "graphics/" + title
        # plt.savefig(output_filename, dpi=300) 
        # print(f"График сохранен в файл: {output_filename}")
        
        plt.show()
    
    def calculate_cluster_metrics(self) -> None:
        self.ab.calculate_cluster_metrics()
        self.wt.calculate_cluster_metrics()

    def graph_analysis(self) -> None:
        for i, dendr_type in enumerate([self.ab, self.wt]):
            dendr_type.graph_analysis()

    def save_grouping_metrics(self) -> None:
        self.wt.save_grouping_metrics()
        self.ab.save_grouping_metrics()

    def save_cluster_metrics(self):
        self.wt.save_cluster_metrics()
        self.ab.save_cluster_metrics()

        # save_cluster_type_metric_dict['Ab'] = {}
        # save_cluster_type_metric_dict['Wt'] = {}

        # save_cluster_type_metric_dict['Ab']['dbscan_eps'] = np.mean(self.ab.dbscan_eps_list)
        # save_cluster_type_metric_dict['Wt']['dbscan_eps'] = np.mean(self.wt.dbscan_eps_list)
        # save_cluster_type_metric_dict['Ab']['dbscan_min_samples'] = np.mean(self.ab.dbscan_min_samples_list)
        # save_cluster_type_metric_dict['Wt']['dbscan_min_samples'] = np.mean(self.wt.dbscan_min_samples_list)
        # save_cluster_type_metric_dict['Ab']['dbscan_statistic'] = np.mean(self.ab.dbscan_statistic_list)
        # save_cluster_type_metric_dict['Wt']['dbscan_statistic'] = np.mean(self.wt.dbscan_statistic_list)
        # save_cluster_type_metric_dict['Ab']['dbscan_p_value'] = np.mean(self.ab.dbscan_p_value_list)
        # save_cluster_type_metric_dict['Wt']['dbscan_p_value'] = np.mean(self.wt.dbscan_p_value_list)
        # save_cluster_type_metric_dict['Ab']['dbscan_noise'] = np.mean(self.ab.dbscan_noise_list)
        # save_cluster_type_metric_dict['Wt']['dbscan_noise'] = np.mean(self.wt.dbscan_noise_list)

        # save_cluster_type_metric_dict['Ab']['dbscan_eps_c'] = np.mean(self.ab.dbscan_eps_c_list)
        # save_cluster_type_metric_dict['Wt']['dbscan_eps_c'] = np.mean(self.wt.dbscan_eps_c_list)
        # save_cluster_type_metric_dict['Ab']['dbscan_min_samples_c'] = np.mean(self.ab.dbscan_min_samples_c_list)
        # save_cluster_type_metric_dict['Wt']['dbscan_min_samples_c'] = np.mean(self.wt.dbscan_min_samples_c_list)
        # save_cluster_type_metric_dict['Ab']['dbscan_statistic_c'] = np.mean(self.ab.dbscan_statistic_c_list)
        # save_cluster_type_metric_dict['Wt']['dbscan_statistic_c'] = np.mean(self.wt.dbscan_statistic_c_list)
        # save_cluster_type_metric_dict['Ab']['dbscan_p_value_c'] = np.mean(self.ab.dbscan_p_value_c_list)
        # save_cluster_type_metric_dict['Wt']['dbscan_p_value_c'] = np.mean(self.wt.dbscan_p_value_c_list)
        # save_cluster_type_metric_dict['Ab']['dbscan_noise_c'] = np.mean(self.ab.dbscan_noise_c_list)
        # save_cluster_type_metric_dict['Wt']['dbscan_noise_c'] = np.mean(self.wt.dbscan_noise_c_list)

        # with open('output/cluster_type_metrics.json', 'w') as f:
        #     json.dump(save_cluster_type_metric_dict, f)

    def save_graph_metrics(self):
        self.wt.save_graph_metrics()
        self.ab.save_graph_metrics()

    def save_dendr_metrics(self) -> None:
        self.wt.save_dendr_metrics()
        self.ab.save_dendr_metrics()

        df = pd.DataFrame(save_all_dendr_metric_dict)
        df.to_csv(output_path('all_dendr_metrics.csv'), index=False)

    def save_dendr_metrics_without_class_cluster(self) -> None:
        self.wt.save_dendr_metrics_without_class_cluster()
        self.ab.save_dendr_metrics_without_class_cluster()

        df = pd.DataFrame(save_all_dendr_metric_dict)
        # df.to_csv('all_dendr_metrics.csv', index=False)
        # df.to_csv('metrics/9009/all_dendr_metrics.csv', index=False)
        df.to_csv(output_path('all_dendr_metrics.csv'), index=False)
