from .dependencies import *
from .comparison import DendrComparisonMixin
from .dendrite_type import DendrType


class DendrAnalysis(DendrComparisonMixin):
    """Контейнер для парного анализа групп дендритных ветвей.

    Входные данные: опционально две группы уже загруженных дендритных ветвей
    `all_types_dataset=(ab_dataset, wt_dataset)`.
    Выходные данные: объект с группами `ab` и `wt`.
    """

    def __init__(self, all_types_dataset: Any = None, flag_9009: Any = None) -> None:
        if all_types_dataset is None:
            raise ValueError(
                "DendrAnalysis no longer loads old precomputed JSON metrics. "
                "Pass already loaded dendrite datasets as all_types_dataset=(ab, wt)."
            )

        self.ab = DendrType("Ab", all_types_dataset[0])
        self.wt = DendrType("Wt", all_types_dataset[1])
        if flag_9009 is not None and len(all_types_dataset) > 2:
            self.ab9009 = DendrType("Ab+9009", all_types_dataset[2])

    def add_spine_class(self) -> None:
        """Загружает классы шипиков для обеих групп.

        Входные данные: группы `ab` и `wt`.
        Выходные данные: отсутствуют; обновляются объекты шипиков.
        """
        for dendrite_type in [self.ab, self.wt]:
            dendrite_type.add_spine_class()

    def add_spine_cluster(self) -> None:
        """Загружает внешние cluster labels шипиков для обеих групп.

        Входные данные: группы `ab` и `wt`.
        Выходные данные: отсутствуют; обновляются объекты шипиков.
        """
        for dendrite_type in [self.ab, self.wt]:
            dendrite_type.add_spine_cluster()

    def calculate_cluster_metrics(self) -> None:
        """Выполняет DBSCAN-кластеризацию для обеих групп.

        Входные данные: загруженные ветви в группах `ab` и `wt`.
        Выходные данные: отсутствуют; результаты сохраняются в объектах ветвей.
        """
        self.ab.calculate_cluster_metrics()
        self.wt.calculate_cluster_metrics()

    def calculate_comprehensive_spatial_analysis(self, **kwargs) -> None:
        """Выполняет полный актуальный анализ ветвей для обеих групп.

        Входные данные: параметры `DendrType.calculate_comprehensive_spatial_analysis`.
        Выходные данные: отсутствуют; результаты сохраняются в объектах ветвей.
        """
        self.ab.calculate_comprehensive_spatial_analysis(**kwargs)
        self.wt.calculate_comprehensive_spatial_analysis(**kwargs)

    def graph_analysis(self) -> None:
        """Выполняет графовый анализ DBSCAN-кластеров для обеих групп.

        Входные данные: группы с рассчитанными DBSCAN-метками.
        Выходные данные: отсутствуют; результаты сохраняются в объектах ветвей.
        """
        self.ab.graph_analysis()
        self.wt.graph_analysis()

    def save_spatial_morphology_analysis(self) -> None:
        """Сохраняет подробные результаты анализа ветвей для обеих групп.

        Входные данные: рассчитанные результаты анализа.
        Выходные данные: JSON-файлы подробных результатов.
        """
        self.ab.save_spatial_morphology_analysis()
        self.wt.save_spatial_morphology_analysis()

    def save_structural_organization_vector(self) -> None:
        """Сохраняет итоговые ML-векторы ветвей для обеих групп.

        Входные данные: рассчитанные признаки ветвей.
        Выходные данные: общий CSV с актуальными признаками дендритных ветвей.
        """
        self.ab.save_structural_organization_vector()
        self.wt.save_structural_organization_vector()
