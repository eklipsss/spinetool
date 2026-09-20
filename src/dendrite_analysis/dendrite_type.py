from .dependencies import *
from .dendrite import Dendrite


class DendrType:
    """Контейнер для группы дендритных ветвей одного типа.

    Входные данные: имя группы и опциональный набор уже загруженных объектов
    или записей датасета с mesh дендрита и mesh шипиков.
    Выходные данные: объект-контейнер с методами пакетного запуска.
    """

    def __init__(self, type: str, dendrite_type_dataset: Any = None) -> None:
        self.type = type
        self.dendrites: List[Dendrite] = []

        if dendrite_type_dataset is not None:
            self.create_dendrs_w_mesh(dendrite_type_dataset)

    def create_dendrs_w_mesh(self, dendrite_type_dataset: Any) -> None:
        """Создаёт дендритные ветви из объектов датасета с mesh.

        Входные данные: iterable объектов, содержащих `pure_dendrite_meshes`
        и `spine_meshes`, или уже готовых объектов `Dendrite`.
        Выходные данные: отсутствуют; изменяется `self.dendrites`.
        """
        for index, dendrite in enumerate(dendrite_type_dataset):
            if isinstance(dendrite, Dendrite):
                self.dendrites.append(dendrite)
                continue

            dendrite_name = getattr(dendrite, "name", f"{self.type}_{index}")
            self.dendrites.append(
                Dendrite(
                    dendrite_name,
                    dendrite.pure_dendrite_meshes,
                    dendrite.spine_meshes,
                )
            )

    def add_spine_class(self) -> None:
        """Загружает классы шипиков для всех ветвей группы.

        Входные данные: состояние `self.dendrites`.
        Выходные данные: отсутствуют; обновляются объекты шипиков.
        """
        for dendrite in self.dendrites:
            dendrite.add_spine_class()

    def add_spine_cluster(self) -> None:
        """Загружает внешние cluster labels шипиков для всех ветвей группы.

        Входные данные: состояние `self.dendrites`.
        Выходные данные: отсутствуют; обновляются объекты шипиков.
        """
        for dendrite in self.dendrites:
            dendrite.add_spine_cluster()

    def calculate_cluster_metrics(self) -> None:
        """Выполняет DBSCAN-кластеризацию для всех ветвей группы.

        Входные данные: загруженные ветви с координатами точек крепления.
        Выходные данные: отсутствуют; результаты сохраняются в объектах ветвей.
        """
        for dendrite in self.dendrites:
            dendrite.calculate_cluster_metrics()

    def calculate_comprehensive_spatial_analysis(self, **kwargs) -> None:
        """Выполняет полный актуальный spatial morphology analysis.

        Входные данные: параметры `Dendrite.calculate_comprehensive_spatial_analysis`.
        Выходные данные: отсутствуют; результаты сохраняются в объектах ветвей.
        """
        for dendrite in self.dendrites:
            dendrite.calculate_comprehensive_spatial_analysis(**kwargs)

    def graph_analysis(self) -> None:
        """Выполняет графовый анализ DBSCAN-кластеров для всех ветвей.

        Входные данные: ветви с рассчитанными DBSCAN-метками.
        Выходные данные: отсутствуют; результаты сохраняются в объектах ветвей.
        """
        for dendrite in self.dendrites:
            dendrite.graph_analysis()

    def save_init_metrics(self) -> None:
        """Сохраняет базовые метрики всех ветвей.

        Входные данные: состояние `self.dendrites`.
        Выходные данные: файлы базовых метрик.
        """
        for dendrite in self.dendrites:
            dendrite.save_init_metrics()

    def save_spatial_morphology_analysis(self) -> None:
        """Сохраняет подробные результаты spatial morphology analysis.

        Входные данные: рассчитанные результаты анализа ветвей.
        Выходные данные: JSON-файлы с подробными результатами.
        """
        for dendrite in self.dendrites:
            dendrite.save_spatial_morphology_analysis()

    def save_structural_organization_vector(self) -> None:
        """Сохраняет итоговый ML-вектор каждой ветви.

        Входные данные: рассчитанные признаки ветвей.
        Выходные данные: общий CSV с актуальными признаками дендритных ветвей.
        """
        for dendrite in self.dendrites:
            dendrite.save_structural_organization_vector()
