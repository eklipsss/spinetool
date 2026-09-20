from .dependencies import *
from .config import *


def compare_ab_wt(ab: List[float], wt: List[float], title: str) -> None:
    """Сравнивает числовую метрику между двумя группами дендритных ветвей.
    Печатает описательные статистики, строит histogram/box/violin
    plot и выполняет двухвыборочный тест Колмогорова--Смирнова.

    Входные данные: значения группы `ab`, значения группы `wt`, заголовок.
    Выходные данные: графики и статистический вывод в stdout.
    """
    data = {"Wt": wt, "Ab": ab}
    print("--------------------------------------------------------------------------")
    print(title)

    print("   wt_mean = ", np.mean(wt))
    print("   ab_mean = ", np.mean(ab))
    print("   wt_median = ", np.median(wt))
    print("   ab_median = ", np.median(ab))

    def print_iqr_stats(values, prefix):
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        median = np.median(values)
        print(f"\n{prefix} дополнительная статистика:")
        print(f"   {prefix}_Q1 = {q1:.2f}")
        print(f"   {prefix}_Q3 = {q3:.2f}")
        print(f"   {prefix}_IQR = {iqr:.2f}")
        print(f"   {prefix}_median - IQR = {median - iqr:.2f}")
        print(f"   {prefix}_median + IQR = {median + iqr:.2f}")
        print(f"   {prefix}_min = {np.min(values):.2f}")
        print(f"   {prefix}_max = {np.max(values):.2f}")

    print_iqr_stats(wt, "wt")
    print_iqr_stats(ab, "ab")

    plt.hist(wt, bins=20, alpha=0.5, label="Wt", color=group_colors["Wt"])
    plt.hist(ab, bins=20, alpha=0.5, label="Ab", color=group_colors["Ab"])
    plt.legend()
    plt.title(title)
    plt.show()

    statistic, p_value = ks_2samp(wt, ab)
    print(f"Статистика теста Колмогорова--Смирнова: {statistic}")
    print(f"P-значение: {p_value}")

    data = {
        "Wt": pd.DataFrame({"Значение": wt, "Group": ["Wt"] * len(wt)}),
        "Ab": pd.DataFrame({"Значение": ab, "Group": ["Ab"] * len(ab)}),
    }

    plt.figure(figsize=(8, 6))
    for key in data.keys():
        sns.boxplot(
            x="Group",
            y="Значение",
            data=data[key],
            boxprops=dict(
                facecolor=group_colors[key],
                edgecolor=edge_colors[key],
                linewidth=2.5,
                alpha=0.7,
            ),
            whiskerprops=dict(color=edge_colors[key], linewidth=2.5),
            capprops=dict(color=edge_colors[key], linewidth=2.5),
            medianprops=dict(color=edge_colors[key], linewidth=2.5),
            width=0.5,
        )

        sns.violinplot(
            x="Group",
            y="Значение",
            data=data[key],
            color=group_colors[key],
            linewidth=1.5,
            inner=None,
            alpha=0.6,
        )

    plt.title(title)
    plt.ylabel("Значение")
    plt.xticks(ticks=[0, 1], labels=["Wt", "Ab"])
    plt.show()


def average_histograms(distances, bins=30):
    """Усредняет histogram-представления нескольких распределений расстояний.

    Входные данные: список массивов расстояний и число бинов.
    Выходные данные: средняя histogram-кривая и границы бинов.
    """
    all_histograms = [np.histogram(d, bins=bins, density=True)[0] for d in distances]
    return np.mean(all_histograms, axis=0), np.histogram(distances[0], bins=bins)[1]


def compare_distr(wt_distances, ab_distances, alpha: float = 0.05):
    """Сравнивает распределения попарных расстояний между двумя группами.
    Строит histogram, считает описательные статистики, выполняет
    тесты Колмогорова--Смирнова, Манна--Уитни и Бруннера--Мунцеля.

    Входные данные: массивы расстояний `wt_distances` и `ab_distances`,
    уровень значимости `alpha`.
    Выходные данные: графики и статистический вывод в stdout.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(wt_distances, bins=30, alpha=0.5, label="Wt", density=True, color=group_colors["Wt"])
    plt.hist(ab_distances, bins=30, alpha=0.5, label="Ab", density=True, color=group_colors["Ab"])
    plt.xlabel("Попарные расстояния", fontsize=18)
    plt.ylabel("Плотность", fontsize=18)
    plt.legend(fontsize=16)
    plt.title("Распределения попарных расстояний для групп", fontsize=18)
    plt.show()

    wt_mean, wt_median, wt_var = np.mean(wt_distances), np.median(wt_distances), np.var(wt_distances)
    ab_mean, ab_median, ab_var = np.mean(ab_distances), np.median(ab_distances), np.var(ab_distances)

    print("   Описательные статистики:")
    print(f"   --> Wt: среднее = {wt_mean:.2f}, медиана = {wt_median:.2f}, дисперсия = {wt_var:.2f}")
    print(f"   --> Ab: среднее = {ab_mean:.2f}, медиана = {ab_median:.2f}, дисперсия = {ab_var:.2f}")

    ks_stat, p_value = ks_2samp(wt_distances, ab_distances)
    print(f"   Тест Колмогорова--Смирнова: D statistic={ks_stat}, p-value={p_value}")
    print("   --> Различия статистически значимы." if p_value < alpha else "   --> Различия статистически не значимы.")

    mannwhitney_test = mannwhitneyu(wt_distances, ab_distances)
    print("   Тест Манна--Уитни:")
    print(f"   --> statistic = {mannwhitney_test.statistic:.2f}, p-value = {mannwhitney_test.pvalue:.2e}")

    brunnermunzel_test = brunnermunzel(wt_distances, ab_distances)
    print("   Тест Бруннера--Мунцеля:")
    print(f"   --> statistic = {brunnermunzel_test.statistic:.2f}, p-value = {brunnermunzel_test.pvalue:.2e}")


class DendrComparisonMixin:
    """Mixin для сравнения актуальных признаков двух групп дендритных ветвей.

    Входные данные: объект с атрибутами `ab` и `wt`, каждый из которых содержит
    список `dendrites`.
    Выходные данные: графики и статистический вывод в stdout.
    """

    def compare_structural_metric(self, metric_name: str) -> None:
        """Сравнивает один признак итогового вектора между группами Ab и Wt.

        Входные данные: имя признака `metric_name`.
        Выходные данные: графики и статистический вывод в stdout.
        """
        ab_values = [
            dendrite.structural_organization_vector[metric_name]
            for dendrite in self.ab.dendrites
            if metric_name in getattr(dendrite, "structural_organization_vector", {})
        ]
        wt_values = [
            dendrite.structural_organization_vector[metric_name]
            for dendrite in self.wt.dendrites
            if metric_name in getattr(dendrite, "structural_organization_vector", {})
        ]
        compare_ab_wt(ab_values, wt_values, metric_name)
