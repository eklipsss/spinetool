from .dependencies import *
from .config import *

def compare_ab_wt(ab: List[float], wt: List[float], title: str) -> None:
    data =  {'Wt': wt, 'Ab': ab}
    print('--------------------------------------------------------------------------')
    print(title)
    
    print('   wt_mean = ', np.mean(wt))
    print('   ab_mean = ', np.mean(ab))

    print('   wt_median = ', np.median(wt))
    print('   ab_median = ', np.median(ab))

    def print_iqr_stats(data, prefix):
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        median = np.median(data)
        print(f'\n{prefix} дополнительная статистика:')
        print(f'   {prefix}_Q1 = {q1:.2f}')
        print(f'   {prefix}_Q3 = {q3:.2f}')
        print(f'   {prefix}_IQR = {iqr:.2f}')
        print(f'   {prefix}_median - IQR = {median - iqr:.2f}')
        print(f'   {prefix}_median + IQR = {median + iqr:.2f}')
        print(f'   {prefix}_min = {np.min(data):.2f}')
        print(f'   {prefix}_max = {np.max(data):.2f}')

    print_iqr_stats(wt, 'wt')
    print_iqr_stats(ab, 'ab')

    # plt.hist(wt, bins=20, alpha=0.5, label='Wt', color = group_colors['Wt'])
    # plt.hist(ab, bins=20, alpha=0.5, label='Ab', color = group_colors['Ab'])
    plt.hist(wt, bins=20, alpha=0.5, label='Wt+9009', color = group_colors['Wt'])
    plt.hist(ab, bins=20, alpha=0.5, label='Ab+9009', color = group_colors['Ab'])
    plt.legend()
    plt.title(title)

    # output_filename = "graphics/Гистограмма - " + title.replace('\n', "") + ".png"  
    output_filename = "graphics/9009 Гистограмма - " + title.replace('\n', "") + ".png"  
    plt.savefig(output_filename, dpi=300)  
    print(f"График сохранен в файл: {output_filename}")

    plt.show()

    # Двухвыборочный тест Колмогорова-Смирнова
    statistic, p_value = ks_2samp(wt, ab)
    print(f"Статистика теста: {statistic}")
    print(f"P-значение: {p_value}")

    data = {'Wt': pd.DataFrame({'Значение': wt,'Group': ['Wt'] * len(wt)}), 'Ab': pd.DataFrame({'Значение': ab,'Group': ['Ab'] * len(ab)})} 

    plt.figure(figsize=(8, 6))

    for key in data.keys():
        sns.boxplot(x='Group', y='Значение', data=data[key], 
                    boxprops=dict(facecolor=group_colors[key], 
                                  edgecolor=edge_colors[key], 
                                  linewidth=2.5, 
                                #   linewidth=3, 
                                  alpha=0.7),
                    whiskerprops=dict(color=edge_colors[key], linewidth=2.5),
                    capprops=dict(color=edge_colors[key], linewidth=2.5),
                    medianprops=dict(color=edge_colors[key], linewidth=2.5),
                    # whiskerprops=dict(color=edge_colors[key], linewidth=3),
                    # capprops=dict(color=edge_colors[key], linewidth=3),
                    # medianprops=dict(color=edge_colors[key], linewidth=3),
                    width=0.5)
        
        vp = sns.violinplot(x='Group', y='Значение', data=data[key], 
                        color=group_colors[key],  
                        linewidth=1.5,       
                        # linewidth=2,       
                        inner=None, 
                        alpha=0.6)    
    
    for i, pc in enumerate(vp.collections):
        pc.set_edgecolor(edge_colors[list(data.keys())[i]])  # Устанавливаем цвет границ
        # pc.set_edgecolor(group_colors[list(data.keys())[i]])  # Устанавливаем цвет границ

    # # Смещение точек влево
    # df['x_offset'] = df['Group'].map({'Ab': -0.2, 'Wt': 0.8})  # Смещение по X

    # for group, color in group_colors.items():
    #     group_data = df[df['Group'] == group]
    #     # Точечный график (stripplot) с учетом смещения
    #     sns.stripplot(x='x_offset', y='Значение', data=group_data, color=color, alpha=0.7, jitter=True)

    plt.title(title)
    plt.ylabel("Значение")
    # plt.xticks(ticks=[0, 1], labels=['Wt', 'Ab'])  
    plt.xticks(ticks=[0, 1], labels=['Wt', 'Ab'])  

    # output_filename = "graphics/" + title.replace('\n', "") + ".png"  
    output_filename = "graphics/new " + title.replace('\n', "") + ".png"  
    plt.savefig(output_filename, dpi=300)  
    print(f"График сохранен в файл: {output_filename}")

    plt.show()


def average_histograms(distances, bins=30):
    all_histograms = [np.histogram(d, bins=bins, density=True)[0] for d in distances]
    return np.mean(all_histograms, axis=0), np.histogram(distances[0], bins=bins)[1]


def compare_distr(wt_distances, ab_distances, alpha = 0.05):
    plt.figure(figsize=(10, 6))
    plt.hist(wt_distances, bins=30, alpha=0.5, label='wt', density=True, color=group_colors['Wt'])
    plt.hist(ab_distances, bins=30, alpha=0.5, label='Ab', density=True, color=group_colors['Ab'])
    plt.xlabel('Попарные расстояния', fontsize = 18)
    plt.ylabel('Плотность', fontsize = 18)
    plt.legend(fontsize = 16)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.title('Распределения попарных расстояний для групп', fontsize = 18)
    plt.savefig('graphics/dists/distribution', dpi=300)
    plt.show()

    # Вычисляем описательные статистики
    wt_mean, wt_median, wt_var = np.mean(wt_distances), np.median(wt_distances), np.var(wt_distances)
    ab_mean, ab_median, ab_var = np.mean(ab_distances), np.median(ab_distances), np.var(ab_distances)

    print("   Описательные статистики:")
    print(f"   --> wt: среднее = {wt_mean:.2f}, медиана = {wt_median:.2f}, дисперсия = {wt_var:.2f}")
    print(f"   --> ab: среднее = {ab_mean:.2f}, медиана = {ab_median:.2f}, дисперсия = {ab_var:.2f}")
    print()

    # Тест Колмогорова-Смирнова
    ks_stat, p_value = ks_2samp(wt_distances, ab_distances)
    print(f"   Тест Колмогорова-Смирнова: D statistic={ks_stat}, p-value={p_value}")
    if p_value < alpha:
        print("   --> Различия в распределениях статистически значимы.")
    else:
        print("   --> Различия в распределениях статистически не значимы.")
    print()

    # Проверка на нормальность распределения (тест Шапиро-Уилка)
    shapiro_wt = shapiro(wt_distances)
    shapiro_ab = shapiro(ab_distances)
    print("   Тест Шапиро-Уилка на нормальность:")
    print(f"   --> wt: статистика = {shapiro_wt.statistic:.2f}, p-value = {shapiro_wt.pvalue:.2e}")
    print(f"   --> ab: статистика = {shapiro_ab.statistic:.2f}, p-value = {shapiro_ab.pvalue:.2e}")
    print()

    # Тест Манна-Уитни для сравнения медиан
    mannwhitney_test = mannwhitneyu(wt_distances, ab_distances)
    print("   Тест Манна-Уитни для сравнения медиан:")
    print(f"  статистика = {mannwhitney_test.statistic:.2f}, p-value = {mannwhitney_test.pvalue:.2e}")
    if mannwhitney_test.pvalue < alpha:
        print("   --> Различия в медианах статистически значимы.")
    else:
        print("   --> Различия в медианах не статистически значимы.")
    print()

    # Тест Бруннера-Мунзеля
    brunnermunzel_test = brunnermunzel(wt_distances, ab_distances)
    print("   Тест Бруннера-Мунзеля:")
    print(f"   --> статистика = {brunnermunzel_test.statistic:.2f}, p-value = {brunnermunzel_test.pvalue:.2e}")

    # Bootstrap-тест - Наблюдаемая разница средних
    observed_diff = np.mean(wt_distances) - np.mean(ab_distances)
    n_iterations = 10000  
    bootstrap_diffs = []

    combined_data = np.concatenate([wt_distances, ab_distances])
    for _ in range(n_iterations):
        # Генерация псевдовыборки
        bootstrap_sample = np.random.choice(combined_data, size=len(combined_data), replace=True)
        # Разделение на две группы
        bootstrap_wt = bootstrap_sample[:len(wt_distances)]
        bootstrap_ab = bootstrap_sample[len(wt_distances):]
        bootstrap_diffs.append(np.mean(bootstrap_wt) - np.mean(bootstrap_ab))

    p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))

    plt.hist(bootstrap_diffs, bins=30, alpha=0.7, label='Bootstrap распределение')
    plt.axvline(observed_diff, color='red', linestyle='dashed', linewidth=2, label='Наблюдаемая разница')
    plt.xlabel('Разница средних')
    plt.ylabel('Частота')
    plt.legend()
    plt.title(f'Bootstrap-тест: p-value = {p_value:.4f}')
    plt.show()

    print(f"   Bootstrap:")
    print(f"   --> Наблюдаемая разница средних: {observed_diff:.2f}")
    print(f"   --> Bootstrap p-value: {p_value:.4f}")

    # Перестановочный тест
    n_permutations = 10000 
    permutation_diffs = []

    for _ in range(n_permutations):
        # Перемешивание меток групп
        shuffled_labels = np.random.permutation(np.concatenate([np.zeros(len(wt_distances)), np.ones(len(ab_distances))]))
        shuffled_wt = combined_data[shuffled_labels == 0]
        shuffled_ab = combined_data[shuffled_labels == 1]
        permutation_diffs.append(np.mean(shuffled_wt) - np.mean(shuffled_ab))

    p_value_perm = np.mean(np.abs(permutation_diffs) >= np.abs(observed_diff))

    plt.hist(permutation_diffs, bins=30, alpha=0.7, label='Перестановочное распределение')
    plt.axvline(observed_diff, color='red', linestyle='dashed', linewidth=2, label='Наблюдаемая разница')
    plt.xlabel('Разница средних')
    plt.ylabel('Частота')
    plt.legend()
    plt.title(f'Перестановочный тест: p-value = {p_value_perm:.4f}')
    plt.show()

    print(f"Перестановочный тест:")
    print(f"Наблюдаемая разница средних: {observed_diff:.2f}")
    print(f"Перестановочный p-value: {p_value_perm:.4f}")



class DendrComparisonMixin:
    def compare_dists(self) -> None:
        wt_all_distances = np.concatenate(self.wt.dists_c_list)
        ab_all_distances = np.concatenate(self.ab.dists_c_list)

        ab_avg_hist, bins = average_histograms(self.ab.dists_c_list)
        
        # plt.figure(figsize=(10, 6))
        # plt.plot(bins[:-1], ab_avg_hist, label='Ab', color=group_colors['Ab'])
        # plt.xlabel('Попарные расстояния', fontsize=14)
        # plt.ylabel('Плотность', fontsize=14)
        # plt.legend(fontsize=12)
        # plt.title('Распределение попарных расстояний (Ab)', fontsize=14)
        
        # plt.savefig('/graphics/dists/ab_distribution', dpi=300)
        # plt.show()
        
        # wt_avg_hist, bins = average_histograms(self.wt.dists_c_list)
        
        # plt.figure(figsize=(10, 6))
        # plt.plot(bins[:-1], wt_avg_hist, label='Wt', color=group_colors['Wt'])
        # plt.xlabel('Попарные расстояния', fontsize=14)
        # plt.ylabel('Плотность', fontsize=14)
        # plt.legend(fontsize=12)
        # plt.title('Распределение попарных расстояний (Wt)', fontsize=14)
        
        # # Сохраняем второй график
        # plt.savefig('/graphics/dists/wt_distribution', dpi=300)
        # plt.show()

        compare_distr(wt_all_distances, ab_all_distances)
        
        compare_ab_wt(self.ab.dist_с_list, self.wt.dist_с_list, 'Попарные расстояния \n(цилиндрические координаты)')

    def compare_grouping_metrics(self) -> None:
        # compare_ab_wt(self.ab.nndist_list, self.wt.nndist_list, 'Метрика NNdist \n(исходные координаты)')      
        # compare_ab_wt(self.ab.nndist_norm_list, self.wt.nndist_norm_list, 'Нормализованная метрика NNdist \n(исходные координаты)')
        # compare_ab_wt(self.ab.entropy_list, self.wt.entropy_list, 'Энтропия Шеннона для pair-distance profile \n(исходные координаты)')

        # compare_ab_wt(self.ab.moran_I_list, self.wt.moran_I_list, 'Индекс Морана I \n(исходные координаты)')
        # compare_ab_wt(self.ab.moran_z_list, self.wt.moran_z_list, 'Индекс Морана zI \n(исходные координаты)')
        # compare_ab_wt(self.ab.moran_p_list, self.wt.moran_p_list, 'Индекс Морана p_value \n(исходные координаты)')

        # compare_ab_wt(self.ab.getis_ord_G_list, self.wt.getis_ord_G_list, 'Индекс Гетиса-Орда G \n(исходные координаты)')
        # compare_ab_wt(self.ab.getis_ord_z_list, self.wt.getis_ord_z_list, 'Индекс Гетиса-Орда zG \n(исходные координаты)')
        # compare_ab_wt(self.ab.getis_ord_p_list, self.wt.getis_ord_p_list, 'Индекс Гетиса-Орда p_value \n(исходные координаты)')

        compare_ab_wt(self.ab.nndist_c_list, self.wt.nndist_c_list, 'Метрика NNdist \n(цилиндрические координаты)')  
        compare_ab_wt(self.ab.nndist_norm_c_list, self.wt.nndist_norm_c_list, 'Нормализованная метрика NNdist \n(цилиндрические координаты)')
        compare_ab_wt(self.ab.entropy_c_list, self.wt.entropy_c_list, 'Энтропия Шеннона для pair-distance profile \n(цилиндрические координаты)')

        ab = filtered_list = [x for x in self.ab.moran_I_c_list if x >= -1]
        wt = filtered_list = [x for x in self.wt.moran_I_c_list if x >= -1]
        compare_ab_wt(ab, wt, 'Индекс Морана I \n(цилиндрические координаты)')
        compare_ab_wt(self.ab.moran_z_c_list, self.wt.moran_z_c_list, 'Индекс Морана zI \n(цилиндрические координаты)')
        compare_ab_wt(self.ab.moran_p_c_list, self.wt.moran_p_c_list, 'Индекс Морана p_value \n(цилиндрические координаты)')

    def compare_cluster_metrics(self) -> None:
        # compare_ab_wt(self.ab.dbscan_eps_list, self.wt.dbscan_eps_list, 'Оптимальный параметр eps для кластеризации DBSCAN \n(исходные координаты)')
        # compare_ab_wt(self.ab.dbscan_min_samples_list, self.wt.dbscan_min_samples_list, 'Оптимальный параметр min_samples для кластеризации DBSCAN \n(исходные координаты)')
        # compare_ab_wt(self.ab.dbscan_statistic_list, self.wt.dbscan_statistic_list, 'Статистика при оптимальных параметрах кластеризации DBSCAN \n(исходные координаты)')
        # compare_ab_wt(self.ab.dbscan_p_value_list, self.wt.dbscan_p_value_list, 'P-value при оптимальных параметрах кластеризации DBSCAN \n(исходные координаты)')
        # compare_ab_wt(self.ab.dbscan_noise_list, self.wt.dbscan_noise_list, 'Процент шума при оптимальных параметрах кластеризации DBSCAN \n(исходные координаты)')
        ab = filtered_list = [x for x in self.ab.dbscan_eps_c_list if x > 0]
        wt = filtered_list = [x for x in self.wt.dbscan_eps_c_list if x > 0]
        compare_ab_wt(ab, wt, 'Оптимальный параметр eps для кластеризации DBSCAN \n(цилиндрические координаты)')
        ab = filtered_list = [x for x in self.ab.dbscan_min_samples_c_list if x > 0]
        wt = filtered_list = [x for x in self.wt.dbscan_min_samples_c_list if x > 0]
        print(ab, wt)
        compare_ab_wt(ab, wt, 'Оптимальный параметр min_samples для кластеризации DBSCAN \n(цилиндрические координаты)')
        compare_ab_wt(self.ab.dbscan_statistic_c_list, self.wt.dbscan_statistic_c_list, 'Статистика при оптимальных параметрах кластеризации DBSCAN \n(цилиндрические координаты)')
        compare_ab_wt(self.ab.dbscan_p_value_c_list, self.wt.dbscan_p_value_c_list, 'P-value при оптимальных параметрах кластеризации DBSCAN \n(цилиндрические координаты)')
        compare_ab_wt(self.ab.dbscan_noise_c_list, self.wt.dbscan_noise_c_list, 'Процент шума при оптимальных параметрах кластеризации DBSCAN \n(цилиндрические координаты)')

    def compare_graph_metrics(self) -> None:
        # compare_ab_wt(self.ab.dbscan_eps_list, self.wt.dbscan_eps_list, 'Оптимальный параметр eps для кластеризации DBSCAN \n(исходные координаты)')
        # compare_ab_wt(self.ab.dbscan_min_samples_list, self.wt.dbscan_min_samples_list, 'Оптимальный параметр min_samples для кластеризации DBSCAN \n(исходные координаты)')
        # compare_ab_wt(self.ab.dbscan_statistic_list, self.wt.dbscan_statistic_list, 'Статистика при оптимальных параметрах кластеризации DBSCAN \n(исходные координаты)')
        # compare_ab_wt(self.ab.dbscan_p_value_list, self.wt.dbscan_p_value_list, 'P-value при оптимальных параметрах кластеризации DBSCAN \n(исходные координаты)')
        # compare_ab_wt(self.ab.dbscan_noise_list, self.wt.dbscan_noise_list, 'Процент шума при оптимальных параметрах кластеризации DBSCAN \n(исходные координаты)')

        ab = [x for x in self.ab.g_mean_cluster_size_list if x > 0]
        wt = [x for x in self.wt.g_mean_cluster_size_list if x > 0]
        compare_ab_wt(ab, wt, 'Средний размер подграфов \n(цилиндрические координаты)')
        ab = [x for x in self.ab.g_characteristic_extent_list if x > 0]
        wt = [x for x in self.wt.g_characteristic_extent_list if x > 0]
        compare_ab_wt(ab, wt, 'Средняя протяженность подграфов \n(цилиндрические координаты)')
        ab = [x for x in self.ab.g_average_clustering_list if x > 0]
        wt = [x for x in self.wt.g_average_clustering_list if x > 0]
        compare_ab_wt(ab, wt, 'Средний коэффициент группировки \n(цилиндрические координаты)')
        ab = [x for x in self.ab.g_modularity_list if x > 0]
        wt = [x for x in self.wt.g_modularity_list if x > 0]
        compare_ab_wt(ab, wt, 'Модульность \n(цилиндрические координаты)')

    def cluster_analysis(self) -> None:
        for i, dendr_type in enumerate([self.ab, self.wt]):
            dendr_type.cluster_analysis()

            for j, sum_count in enumerate([dendr_type.db_sum_count_class, 
                              dendr_type.db_sum_count_class_c, 
                              dendr_type.db_sum_count_cluster,
                              dendr_type.db_sum_count_cluster_c]):
                if j == 0 or j==1:
                    colors = class_group_colors
                    e_colors = class_group_edge_colors
                else:
                    colors = cluster_group_colors
                    e_colors = cluster_group_edge_colors

                for class_type in list(colors.keys()):
                    df = pd.DataFrame({
                        'Значение': sum_count[class_type],
                        'Тип шипиков': [class_type]*len(sum_count[class_type]) 
                    })

                    sns.boxplot(x='Тип шипиков', y='Значение', data=df, 
                                boxprops=dict(facecolor=colors[class_type], edgecolor=e_colors[class_type], linewidth=2.5, alpha=0.7),
                                whiskerprops=dict(color=e_colors[class_type], linewidth=2.5),
                                capprops=dict(color=e_colors[class_type], linewidth=2.5),
                                medianprops=dict(color=e_colors[class_type], linewidth=2.5),
                                width=0.5)
                    
                    vp = sns.violinplot(x='Тип шипиков', y='Значение', data=df, 
                                    color=colors[class_type], 
                                    linewidth=1.5,       
                                    inner=None, 
                                    alpha=0.5)    
                    
                for k, pc in enumerate(vp.collections):
                    pc.set_edgecolor(e_colors[list(colors.keys())[k]])

                if i == 0:
                    title = 'Количество шипиков в кластерах dbscan \nAb'
                else:
                    title = 'Количество шипиков в кластерах dbscan \nWt'

                if j == 0 or j==2:
                    title += ' (исходные координаты)'
                else:
                    title += ' (цилиндрические координаты)'

                plt.title(title)
                plt.ylabel("Значение")

                if j == 0 or j==1:
                    plt.xticks(ticks=[0, 1, 2, 3], labels=classes_list) 
                    output_filename = "graphics/" + title.replace('\n', "") + " - классы.png"
                else:
                    plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=clusters_list) 
                    output_filename = "graphics/" + title.replace('\n', "") + " - кластеры.png"
 
                plt.savefig(output_filename, dpi=300)  
                print(f"График сохранен в файл: {output_filename}")

                plt.show()


        delta_matrix_class = self.ab.db_total_matrix_class - self.wt.db_total_matrix_class
        delta_matrix_class_c = self.ab.db_total_matrix_class_c - self.wt.db_total_matrix_class_c
        delta_matrix_cluster = self.ab.db_total_matrix_cluster - self.wt.db_total_matrix_cluster
        delta_matrix_cluster_c = self.ab.db_total_matrix_cluster_c - self.wt.db_total_matrix_cluster_c

        mins = [delta_matrix_class.min(), delta_matrix_class_c.min(), delta_matrix_cluster.min(), delta_matrix_cluster_c.min()]
        maxs = [delta_matrix_class.max(), delta_matrix_class_c.max(), delta_matrix_cluster.max(), delta_matrix_cluster_c.max()]

        df_db_table_class = pd.DataFrame(delta_matrix_class, index=classes_list, columns=classes_list)
        df_db_table_class_c = pd.DataFrame(delta_matrix_class_c, index=classes_list, columns=classes_list)
        df_db_table_cluster = pd.DataFrame(delta_matrix_cluster, index=clusters_list, columns=clusters_list)
        df_db_table_cluster_c = pd.DataFrame(delta_matrix_cluster_c, index=clusters_list, columns=clusters_list)

        for i, correlation_table in enumerate([df_db_table_class, df_db_table_class_c, df_db_table_cluster, df_db_table_cluster_c]):
            if i == 0 or i == 2:
                title = 'Таблица разности - Кластеризация dbscan '                
            else:
                title = 'Таблица разности - Кластеризация dbscan '
            
            title = title +'\n' + 'Ab-Wt'

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
                # cmap="coolwarm",  
                cmap="PiYG",  
                vmin=mins[i],  
                vmax=maxs[i],  
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

    def neighborhood_analysis(self, eps: float = 0) -> None:
        for i, dendr_type in enumerate([self.ab, self.wt]):
            dendr_type.neighborhood_analysis(eps)

            for j, sum_count in enumerate([dendr_type.n_sum_count_class, 
                              dendr_type.n_sum_count_class_c, 
                              dendr_type.n_sum_count_cluster,
                              dendr_type.n_sum_count_cluster_c]):
                if j == 0 or j==1:
                    colors = class_group_colors
                    e_colors = class_group_edge_colors
                else:
                    colors = cluster_group_colors
                    e_colors = cluster_group_edge_colors

                for class_type in list(colors.keys()):
                    df = pd.DataFrame({
                        'Значение': sum_count[class_type],
                        'Тип шипиков': [class_type]*len(sum_count[class_type]) 
                    })

                    sns.boxplot(x='Тип шипиков', y='Значение', data=df, 
                                boxprops=dict(facecolor=colors[class_type], edgecolor=e_colors[class_type], linewidth=2.5, alpha=0.7),
                                whiskerprops=dict(color=e_colors[class_type], linewidth=2.5),
                                capprops=dict(color=e_colors[class_type], linewidth=2.5),
                                medianprops=dict(color=e_colors[class_type], linewidth=2.5),
                                width=0.5)
                    
                    vp = sns.violinplot(x='Тип шипиков', y='Значение', data=df, 
                                    color=colors[class_type], 
                                    linewidth=1.5,       
                                    inner=None, 
                                    alpha=0.5)    
                    
                for k, pc in enumerate(vp.collections):
                    pc.set_edgecolor(e_colors[list(colors.keys())[k]])

                if i == 0:
                    title = 'Количество шипиков в окрестностях \nAb'
                else:
                    title = 'Количество шипиков в окрестностях \nWt'

                if j == 0 or j==2:
                    title += ' (исходные координаты)'
                else:
                    title += ' (цилиндрические координаты)'

                plt.title(title)
                plt.ylabel("Значение")

                if j == 0 or j==1:
                    plt.xticks(ticks=[0, 1, 2, 3], labels=classes_list) 
                    output_filename = "graphics/" + title.replace('\n', "") + " - классы.png"
                else:
                    plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=clusters_list)  
                    output_filename = "graphics/" + title.replace('\n', "") + " - кластеры.png"
                
                plt.savefig(output_filename, dpi=300)  
                print(f"График сохранен в файл: {output_filename}")

                plt.show()

        delta_matrix_class =  self.ab.n_total_matrix_class - self.wt.n_total_matrix_class
        delta_matrix_class_c = self.ab.n_total_matrix_class_c - self.wt.n_total_matrix_class_c
        delta_matrix_cluster = self.ab.n_total_matrix_cluster - self.wt.n_total_matrix_cluster
        delta_matrix_cluster_c = self.ab.n_total_matrix_cluster_c - self.wt.n_total_matrix_cluster_c

        mins = [delta_matrix_class.min(), delta_matrix_class_c.min(), delta_matrix_cluster.min(), delta_matrix_cluster_c.min()]
        maxs = [delta_matrix_class.max(), delta_matrix_class_c.max(), delta_matrix_cluster.max(), delta_matrix_cluster_c.max()]

        df_n_table_class = pd.DataFrame(delta_matrix_class, index=classes_list, columns=classes_list)
        df_n_table_class_c = pd.DataFrame(delta_matrix_class_c, index=classes_list, columns=classes_list)
        df_n_table_cluster = pd.DataFrame(delta_matrix_cluster, index=clusters_list, columns=clusters_list)
        df_n_table_cluster_c = pd.DataFrame(delta_matrix_cluster_c, index=clusters_list, columns=clusters_list)

        for i, correlation_table in enumerate([df_n_table_class, df_n_table_class_c, df_n_table_cluster, df_n_table_cluster_c]):
            if i == 0 or i == 2:
                title = 'Таблица разности - Формирование окрестностей '                
            else:
                title = 'Таблица разности - Формирование окрестностей '
            
            title = title +'\n' + 'Ab-Wt'

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
                vmin=mins[i],  
                vmax=maxs[i],  
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

