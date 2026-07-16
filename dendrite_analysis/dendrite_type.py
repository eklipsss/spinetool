from .dependencies import *
from .config import *
from .dendrite import Dendrite

class DendrType:
    type: str 
    dendrites: List[Dendrite] = []

    dists_list = []
    dists_c_list = []

    nndist_list: List[float] = []
    nndist_norm_list: List[float] = []
    nndist_c_list: List[float] = []
    nndist_norm_c_list: List[float] = []

    average_pcf_values_c = []
    r_values_c = []
    pcf_len_c: int

    entropy_list: List[float] = []
    entropy_c_list: List[float] = []

    moran_I_list: List[float] = []
    moran_z_list: List[float] = []
    moran_p_list: List[float] = []
    moran_I_c_list: List[float] = []
    moran_z_c_list: List[float] = []
    moran_p_c_list: List[float] = []

    moran_dict_c: Dict[str, List[float]] = {}

    getis_ord_G_list: List[float] = []
    getis_ord_z_list: List[float] = []
    getis_ord_p_list: List[float] = []
    getis_ord_G_c_list: List[float] = []
    getis_ord_z_c_list: List[float] = []
    getis_ord_p_c_list: List[float] = []

    dbscan_eps_list: List[float] = []
    dbscan_min_samples_list: List[float] = []
    dbscan_statistic_list: List[float] = []
    dbscan_p_value_list: List[float] = []

    dbscan_eps_c_list: List[float] = []
    dbscan_min_samples_c_list: List[float] = []
    dbscan_statistic_c_list: List[float] = []
    dbscan_p_value_c_list: List[float] = []

    dbscan_noise_list: List[float] = []
    dbscan_noise_c_list: List[float] = []

    db_sum_count_class: Dict[str, List[int]]
    db_sum_count_class_c: Dict[str, List[int]]
    db_sum_count_cluster: Dict[int, List[int]]
    db_sum_count_cluster_c: Dict[int, List[int]]

    db_total_matrix_class: np.ndarray = np.zeros((len(classes_list), len(classes_list)))
    db_total_matrix_class_c: np.ndarray = np.zeros((len(classes_list), len(classes_list)))
    db_total_matrix_cluster: np.ndarray = np.zeros((len(clusters_list), len(clusters_list)))
    db_total_matrix_cluster_c: np.ndarray = np.zeros((len(clusters_list), len(clusters_list)))

    n_sum_count_class: Dict[str, List[int]]
    n_sum_count_class_c: Dict[str, List[int]]
    n_sum_count_cluster: Dict[str, List[int]]
    n_sum_count_cluster_c: Dict[str, List[int]]

    n_total_matrix_class: np.ndarray = np.zeros((len(classes_list), len(classes_list)))
    n_total_matrix_class_c: np.ndarray = np.zeros((len(classes_list), len(classes_list)))
    n_total_matrix_cluster: np.ndarray = np.zeros((len(clusters_list), len(clusters_list)))
    n_total_matrix_cluster_c: np.ndarray = np.zeros((len(clusters_list), len(clusters_list)))

    g_mean_cluster_size_list: List[float] = []
    g_characteristic_extent_list: List[float] = []
    g_average_clustering_list: List[float] = []
    g_modularity_list: List[float] = []


    def __init__(self, type: str, dendrite_type_dataset: Any = None) -> None:
        print('OneDendrType init')
        self.type = type

        self.dendrites: List[Dendrite] = []

        self.dists_list = []
        self.dists_c_list = []

        self.nndist_list = []
        self.nndist_norm_list = []
        self.nndist_c_list = []
        self.nndist_norm_c_list = []

        self.average_pcf_values_c = []
        self.r_values_c = []

        self.entropy_list = []
        self.entropy_c_list = []

        self.moran_I_list = []
        self.moran_z_list = []
        self.moran_p_list = []
        self.moran_I_c_list = []
        self.moran_z_c_list = []
        self.moran_p_c_list = []
        
        self.moran_dict_c = {}
        for metric in spine_metrics_name:
            self.moran_dict_c[metric] = []

        self.getis_ord_G_list = []
        self.getis_ord_z_list = []
        self.getis_ord_p_list = []
        self.getis_ord_G_c_list = []
        self.getis_ord_z_c_list = []
        self.getis_ord_p_c_list = []

        self.dbscan_eps_list = []
        self.dbscan_min_samples_list = []
        self.dbscan_statistic_list = []
        self.dbscan_p_value_list = []

        self.dbscan_eps_c_list = []
        self.dbscan_min_samples_c_list = []
        self.dbscan_statistic_c_list = []
        self.dbscan_p_value_c_list = []

        self.dbscan_noise_list = []
        self.dbscan_noise_c_list = []

        self.db_sum_count_class = {}
        self.db_sum_count_class_c = {}
        self.db_sum_count_cluster = {}
        self.db_sum_count_cluster_c = {}

        self.db_total_matrix_class = np.zeros((len(classes_list), len(classes_list)))
        self.db_total_matrix_class_c = np.zeros((len(classes_list), len(classes_list)))
        self.db_total_matrix_cluster = np.zeros((len(clusters_list), len(clusters_list)))
        self.db_total_matrix_cluster_c = np.zeros((len(clusters_list), len(clusters_list)))

        self.n_sum_count_class = {}
        self.n_sum_count_cluster = {}
        self.n_sum_count_class_c = {}
        self.n_sum_count_cluster_c = {}

        self.n_total_matrix_class = np.zeros((len(classes_list), len(classes_list)))
        self.n_total_matrix_class_c = np.zeros((len(classes_list), len(classes_list)))
        self.n_total_matrix_cluster = np.zeros((len(clusters_list), len(clusters_list)))
        self.n_total_matrix_cluster_c = np.zeros((len(clusters_list), len(clusters_list)))

        self.g_mean_cluster_size_list = []
        self.g_characteristic_extent_list = []
        self.g_average_clustering_list = []
        self.g_modularity_list = []

        if dendrite_type_dataset is not None:
            self.create_dendrs_w_mesh(dendrite_type_dataset)
        else:
            self.create_dendrs()
        
        self.save_init_metrics()
        # for d in self.dendrites:
        #     save_dendr_metric_dict[d.name] = {'Volume' : d.volume, 'Length' : d.length, 'Radius' : d.radius, 'Dr' : d.dr, 'Volume_around_dendr' : d.volume_around_dendr}
        #     self.save_init_metrics()
        #     if d.cylindr_flag:
        #         self.dists_c_list.append(d.dists_c)

    def create_dendrs_w_mesh(self, dendrite_type_dataset: Any) -> None:
        for dendrite in dendrite_type_dataset:
            self.dendrites.append(Dendrite('None', dendrite.pure_dendrite_meshes, dendrite.spine_meshes))

    def create_dendrs(self) -> None:
        # with open('metrics/dendr_metrics.json', 'r') as f:
        # with open('metrics/9009/dendr_metrics.json', 'r') as f:
        # with open('metrics/wt_old_st/dendr_metrics.json', 'r') as f:
        with open('input/dendr_metrics.json', 'r') as f:
            loaded_dict = json.load(f)

        for dendr_name in loaded_dict.keys():
            if self.type in dendr_name:
                self.dendrites.append(Dendrite(dendr_name))

    def add_spine_class(self) -> None:
        for d in self.dendrites:
            d.add_spine_class()

    def add_spine_cluster(self) -> None:
        for d in self.dendrites:
            d.add_spine_cluster()

    def load_grouping_metrics(self) -> None:
        for d in self.dendrites:
            d.load_grouping_metrics()

        self.grouping_metrics_to_list()

    def load_cluster_metrics(self) -> None:
        for d in self.dendrites:
            d.load_cluster_metrics()

        self.cluster_metrics_to_list()

    def calculate_grouping_metrics(self) -> None:
        for d in self.dendrites:
            d.calculate_grouping_metrics()

        self.grouping_metrics_to_list()

    def grouping_metrics_to_list(self) -> None:
        for d in self.dendrites:
            self.nndist_list.append(d.nndist)
            self.nndist_norm_list.append(d.nndist_norm)
            self.entropy_list.append(d.entropy)

            self.moran_I_list.append(d.moran_I)
            self.moran_z_list.append(d.moran_z)
            self.moran_p_list.append(d.moran_p)

            # self.getis_ord_G_list.append(d.getis_ord_G)
            # self.getis_ord_z_list.append(d.getis_ord_z)
            # self.getis_ord_p_list.append(d.getis_ord_p)

            if d.cylindr_flag:
                self.nndist_c_list.append(d.nndist_c)
                self.nndist_norm_c_list.append(d.nndist_norm_c)
                self.entropy_c_list.append(d.entropy_c)

                self.moran_I_c_list.append(d.moran_I_c)
                self.moran_z_c_list.append(d.moran_z_c)
                self.moran_p_c_list.append(d.moran_p_c)

                # self.getis_ord_G_c_list.append(d.getis_ord_G_c)
                # self.getis_ord_z_c_list.append(d.getis_ord_z_c)
                # self.getis_ord_p_c_list.append(d.getis_ord_p_c)
                
                # for metric in spine_metrics_name:
                #     self.moran_dict_c[metric].append(d.moran_dict_c[metric])

        # self.draw_mean_pcf()

    def draw_mean_pcf(self) -> None:
        r_values = []
        r_values_c = []
        pcf_values_list = []
        pcf_values_list_c = []
        pcf_len = 1000
        pcf_len_c = 1000

        for d in self.dendrites:
            if len(d.r_values) < pcf_len and len(d.r_values) > 9:
                r_values = d.r_values
                pcf_len = len(d.r_values)

            if d.cylindr_flag:
                if len(d.r_values_c) < self.pcf_len_c and len(d.r_values_c) > 9:
                    self.r_values_c = d.r_values_c
                    self.pcf_len_c = len(d.r_values_c)

        for d in self.dendrites:
            if len(d.pcf_values) >= 9:
                pcf_values_list.append(d.pcf_values[:pcf_len])
            if d.cylindr_flag:
                if len(d.pcf_values_c) >= 9:
                    pcf_values_list_c.append(d.pcf_values_c[:self.pcf_len_c]) 

        self.average_pcf_values = np.mean(pcf_values_list, axis=0)
        self.average_pcf_values_c = np.mean(pcf_values_list_c, axis=0)
        
        plt.figure(figsize=(10, 6))
        plt.plot(r_values[:pcf_len], self.average_pcf_values, label='pair-distance profile', color='#8cbd3a')
        # plt.plot(self.dendrites[0].r_values[:pcf_len], wt_avg_hist, label='wt', color='#86dd18')
        # plt.plot(bins[:-1], ab_avg_hist, label='ab', color='#ff6889')
        plt.xlabel('Попарные расстояния', fontsize = 18)
        plt.ylabel('Pair-distance profile', fontsize = 18)
        plt.legend(fontsize = 16)
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
        title = 'Усредненный график pair-distance profile ' + self.type + ' (исходные координаты)'
        plt.title(title, fontsize = 18)

        output_filename = "graphics/" + title
        plt.savefig(output_filename, dpi=300) 
        print(f"График сохранен в файл: {output_filename}")
        
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(r_values_c[:pcf_len_c], self.average_pcf_values_c, label='pair-distance profile', color='#8cbd3a')
        plt.xlabel('Попарные расстояния', fontsize = 18)
        plt.ylabel('Pair-distance profile', fontsize = 18)
        plt.legend(fontsize = 16)
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
        title = 'Усредненный график pair-distance profile ' + self.type + ' (цилиндрические координаты)'
        plt.title(title, fontsize = 18)

        output_filename = "graphics/" + title
        plt.savefig(output_filename, dpi=300) 
        print(f"График сохранен в файл: {output_filename}")
        
        plt.show()

    def calculate_cluster_metrics(self) -> None:
        for d in self.dendrites:
            d.calculate_cluster_metrics()

        self.cluster_metrics_to_list()

    def cluster_metrics_to_list(self) -> None:
        for d in self.dendrites:
            self.dbscan_eps_list.append(d.dbscan_eps)
            self.dbscan_min_samples_list.append(d.dbscan_min_samples)
            self.dbscan_statistic_list.append(d.dbscan_statistic)
            self.dbscan_p_value_list.append(d.dbscan_p_value)
            self.dbscan_noise_list.append(d.dbscan_noise)                   

            if d.cylindr_flag:
                self.dbscan_eps_c_list.append(d.dbscan_eps_c)
                self.dbscan_min_samples_c_list.append(d.dbscan_min_samples_c)
                self.dbscan_statistic_c_list.append(d.dbscan_statistic_c)
                self.dbscan_p_value_c_list.append(d.dbscan_p_value_c)
                self.dbscan_noise_c_list.append(d.dbscan_noise_c)

    def cluster_analysis(self) -> None:
        self.db_sum_count_class = class_dict
        self.db_sum_count_class_c = class_dict
        self.db_sum_count_cluster = cluster_dict
        self.db_sum_count_cluster_c = cluster_dict

        for d in self.dendrites:
            d.cluster_analysis()

            if d.dbscan_labels.size != 0 :

                for type in self.db_sum_count_class.keys():
                    self.db_sum_count_class[type] = np.concatenate([self.db_sum_count_class[type], d.db_count_class[type]]) 
                    if d.cylindr_flag:
                        self.db_sum_count_class_c[type] = np.concatenate([self.db_sum_count_class_c[type], d.db_count_class_c[type]])

                for type in self.db_sum_count_cluster.keys():
                    self.db_sum_count_cluster[type] = np.concatenate([self.db_sum_count_cluster[type], d.db_count_cluster[type]]) 
                    if d.cylindr_flag:
                        self.db_sum_count_cluster_c[type] = np.concatenate([self.db_sum_count_cluster_c[type], d.db_count_cluster_c[type]])

                self.db_total_matrix_class += d.db_matrix_class
                self.db_total_matrix_class_c += d.db_matrix_class_c
                self.db_total_matrix_cluster += d.db_matrix_cluster
                self.db_total_matrix_cluster_c += d.db_matrix_cluster_c

        for dict in [self.db_sum_count_class, self.db_sum_count_class_c, self.db_sum_count_cluster, self.db_sum_count_class_c]:
            for type in dict.keys():
                dict[type] = dict[type][dict[type] != 0]
        
        classes = classes_list
        clusters = clusters_list

        self.db_total_matrix_class = self.db_total_matrix_class / self.db_total_matrix_class.max()
        self.db_total_matrix_class_c = self.db_total_matrix_class_c / self.db_total_matrix_class_c.max()
        self.db_total_matrix_cluster = self.db_total_matrix_cluster / self.db_total_matrix_cluster.max()
        self.db_total_matrix_cluster_c = self.db_total_matrix_cluster_c / self.db_total_matrix_cluster_c.max()

        df_db_table_class = pd.DataFrame(self.db_total_matrix_class, index=classes, columns=classes)
        df_db_table_class_c = pd.DataFrame(self.db_total_matrix_class_c, index=classes, columns=classes)
        df_db_table_cluster = pd.DataFrame(self.db_total_matrix_cluster, index=clusters, columns=clusters)
        df_db_table_cluster_c = pd.DataFrame(self.db_total_matrix_cluster_c, index=clusters, columns=clusters)

        for i, correlation_table in enumerate([df_db_table_class, df_db_table_class_c, df_db_table_cluster, df_db_table_cluster_c]):
            if i == 0 or i == 2:
                title = 'Кластеризация dbscan '                
            else:
                title = 'Кластеризация dbscan '
            
            title = title +'\n' + self.type

            if i == 0 or i == 2:
                title += ' (исходные координаты)'
            else:
                title += ' (цилиндрические координаты)'
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                correlation_table,
                # annot=True,  # Значения в ячейках
                # annot_kws={"size": 20},
                fmt=".2f",  
                cmap="PiYG",  
                vmin=0,  
                vmax=1,  
                linewidths=0.5,  
                linecolor="black"  
            )

            plt.title(title)
            if i == 0 or i == 1: 
                plt.xlabel("Классы", fontsize=14)
                plt.ylabel("Классы", fontsize=14)
                output_filename = "graphics/" + title.replace('\n', "") + " - классы.png"  
            else:
                plt.xlabel("Кластеры", fontsize=14)
                plt.ylabel("Кластеры", fontsize=14)
                output_filename = "graphics/" + title.replace('\n', "") + " - кластеры.png"  

            plt.xticks(rotation=45)  
            plt.yticks(rotation=0)   
            plt.tight_layout()  

            plt.savefig(output_filename, dpi=300) 
            print(f"График сохранен в файл: {output_filename}")

            plt.show()

    def graph_analysis(self) -> None:
        for d in self.dendrites:
            d.graph_analysis()

            self.g_mean_cluster_size_list.append(d.g_mean_cluster_size)
            self.g_characteristic_extent_list.append(d.g_characteristic_extent)
            self.g_average_clustering_list.append(d.g_average_clustering)
            self.g_modularity_list.append(d.g_modularity)
        
        print(self.g_mean_cluster_size_list)

    def neighborhood_analysis(self, eps: float = 0) -> None:
        self.n_sum_count_class = class_dict
        self.n_sum_count_class_c = class_dict
        self.n_sum_count_cluster = cluster_dict
        self.n_sum_count_cluster_c = cluster_dict

        mean_n_counts = []

        for d in self.dendrites:
            d.neighborhood_analysis(eps)

            if d.cylindr_flag:
                mean_n_counts.append(d.n_count_c)

            for type in self.n_sum_count_class.keys():
                self.n_sum_count_class[type] = np.concatenate([self.n_sum_count_class[type], d.n_count_class[type]]) 
                if d.cylindr_flag:
                    self.n_sum_count_class_c[type] = np.concatenate([self.n_sum_count_class_c[type], d.n_count_class_c[type]])

            for type in self.n_sum_count_cluster.keys():
                self.n_sum_count_cluster[type] = np.concatenate([self.n_sum_count_cluster[type], d.n_count_cluster[type]]) 
                if d.cylindr_flag:
                    self.n_sum_count_cluster_c[type] = np.concatenate([self.n_sum_count_cluster_c[type], d.n_count_cluster_c[type]])
            
            self.n_total_matrix_class += d.n_matrix_class
            self.n_total_matrix_class_c += d.n_matrix_class_c
            self.n_total_matrix_cluster += d.n_matrix_cluster
            self.n_total_matrix_cluster_c += d.n_matrix_cluster_c

        for dict in [self.n_sum_count_class, self.n_sum_count_class_c, self.n_sum_count_cluster, self.n_sum_count_class_c]:
            for type in dict.keys():
                dict[type] = dict[type][dict[type] != 0]

        print('mean_count: ', np.mean(mean_n_counts))

        # Среднее количество шипиков в окрестности
        # fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # sns.boxplot(y=mean_n_counts, ax=axes[0])
        # axes[0].set_title('Boxplot')
        # axes[0].set_ylabel('Values')
        # sns.violinplot(y=mean_n_counts, ax=axes[1])
        # axes[1].set_title('Violin Plot')
        # axes[1].set_ylabel('Values')
        # plt.tight_layout()
        # plt.show()

        self.n_total_matrix_class = self.n_total_matrix_class / self.n_total_matrix_class.max()
        self.n_total_matrix_class_c = self.n_total_matrix_class_c / self.n_total_matrix_class_c.max()
        self.n_total_matrix_cluster = self.n_total_matrix_cluster / self.n_total_matrix_cluster.max()
        self.n_total_matrix_cluster_c = self.n_total_matrix_cluster_c / self.n_total_matrix_cluster_c.max()

        df_n_table_class = pd.DataFrame(self.n_total_matrix_class, index=classes_list, columns=classes_list)
        df_n_table_class_c = pd.DataFrame(self.n_total_matrix_class_c, index=classes_list, columns=classes_list)
        df_n_table_cluster = pd.DataFrame(self.n_total_matrix_cluster, index=clusters_list, columns=clusters_list)
        df_n_table_cluster_c = pd.DataFrame(self.n_total_matrix_cluster_c, index=clusters_list, columns=clusters_list)


        for i, correlation_table in enumerate([df_n_table_class, df_n_table_class_c, df_n_table_cluster, df_n_table_cluster_c]):
            if i == 0 or i == 2:
                title = 'Формирование окрестностей '                
            else:
                title = 'Формирование окрестностей '
            
            title = title +'\n' + self.type

            if i == 0 or i == 2:
                title += ' (исходные координаты)'
            else:
                title += ' (цилиндрические координаты)'
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                correlation_table,
                # annot=True,  # Значения в ячейках
                # annot_kws={"size": 20},
                fmt=".2f",  
                cmap="PiYG",  
                vmin=0,  
                vmax=1,  
                linewidths=0.5,  
                linecolor="black"  
            )

            plt.title(title)
            if i == 0 or i == 1: 
                plt.xlabel("Классы", fontsize=14)
                plt.ylabel("Классы", fontsize=14)
                output_filename = "graphics/" + title.replace('\n', "") + " - классы.png"  
            else:
                plt.xlabel("Кластеры", fontsize=14)
                plt.ylabel("Кластеры", fontsize=14)
                output_filename = "graphics/" + title.replace('\n', "") + " - кластеры.png" 

            plt.xticks(rotation=45)  
            plt.yticks(rotation=0)   
            plt.tight_layout()  

            plt.savefig(output_filename, dpi=300) 
            print(f"График сохранен в файл: {output_filename}")

            plt.show()

    def save_init_metrics(self) -> None:
        for d in self.dendrites:
            d.save_init_metrics()

        # with open('dendr_metrics.json', 'w') as f:
        # with open('metrics/9009/dendr_metrics.json', 'w') as f:
        # with open('metrics/wt_old_st/dendr_metrics.json', 'w') as f:
        # with open('output/dendr_metrics.json', 'w') as f:
        #     json.dump(save_dendr_metric_dict, f)

    def save_grouping_metrics(self) -> None:
        for d in self.dendrites:
            d.save_grouping_metrics()

            # save_grouping_dendr_metric_dict[d.name] = {}
            # save_grouping_dendr_metric_dict[d.name]['NNdist'] = d.nndist
            # save_grouping_dendr_metric_dict[d.name]['NNdist_norm'] = d.nndist_norm

            # save_grouping_dendr_metric_dict[d.name]['PCF_values'] = d.pcf_values.tolist()
            # save_grouping_dendr_metric_dict[d.name]['r_values'] = d.r_values.tolist()
            # # save_grouping_dendr_metric_dict[d.name]['PCF_values'] = d.pcf_values
            # # save_grouping_dendr_metric_dict[d.name]['r_values'] = d.r_values
            # save_grouping_dendr_metric_dict[d.name]['Entropy'] = d.entropy

            # save_grouping_dendr_metric_dict[d.name]['Moran_I'] = d.moran_I
            # save_grouping_dendr_metric_dict[d.name]['Moran_zI'] = d.moran_z
            # save_grouping_dendr_metric_dict[d.name]['Moran_p'] = d.moran_p

            # # save_grouping_dendr_metric_dict[d.name]['Getis_Ord_G'] = d.getis_ord_G
            # # save_grouping_dendr_metric_dict[d.name]['Getis_Ord_zG'] = d.getis_ord_z
            # # save_grouping_dendr_metric_dict[d.name]['Getis_Ord_p'] = d.getis_ord_p

            # if d.cylindr_flag:
            #     save_grouping_dendr_metric_dict[d.name]['NNdist_c'] = d.nndist_c
            #     save_grouping_dendr_metric_dict[d.name]['NNdist_norm_c'] = d.nndist_norm_c

            #     save_grouping_dendr_metric_dict[d.name]['PCF_values_c'] = d.pcf_values_c.tolist()
            #     save_grouping_dendr_metric_dict[d.name]['r_values_c'] = d.r_values_c.tolist()
            #     # save_grouping_dendr_metric_dict[d.name]['PCF_values_c'] = d.pcf_values_c
            #     # save_grouping_dendr_metric_dict[d.name]['r_values_c'] = d.r_values_c
            #     save_grouping_dendr_metric_dict[d.name]['Entropy_c'] = d.entropy_c

            #     save_grouping_dendr_metric_dict[d.name]['Moran_I_c'] = d.moran_I_c
            #     save_grouping_dendr_metric_dict[d.name]['Moran_zI_c'] = d.moran_z_c
            #     save_grouping_dendr_metric_dict[d.name]['Moran_p_c'] = d.moran_p_c

            #     # save_grouping_dendr_metric_dict[d.name]['Getis_Ord_G_c'] = d.getis_ord_G_c
            #     # save_grouping_dendr_metric_dict[d.name]['Getis_Ord_zG_c'] = d.getis_ord_z_c
            #     # save_grouping_dendr_metric_dict[d.name]['Getis_Ord_p_c'] = d.getis_ord_p_c

            # # with open('metrics/grouping_dendr_metrics.json', 'w') as f:
            # # with open('metrics/9009/grouping_dendr_metrics.json', 'w') as f:
            # # with open('metrics/wt_old_st/grouping_dendr_metrics.json', 'w') as f:
            # with open('output/grouping_dendr_metrics.json', 'w') as f:
            #     json.dump(save_grouping_dendr_metric_dict, f)

    def save_cluster_metrics(self):
        for d in self.dendrites:
            d.save_cluster_metrics()

    def save_graph_metrics(self):
        for d in self.dendrites:
            d.save_graph_metrics()

    def save_dendr_metrics(self) -> None:
        for d in self.dendrites:
            d.save_dendr_metrics()

    def save_dendr_metrics_without_class_cluster(self) -> None:
        for d in self.dendrites:
            d.save_dendr_metrics_without_class_cluster()
