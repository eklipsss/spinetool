# План реализации моделей генерации шипиков (модуль S)

Реализация ТЗ `docs/neuron-model/s-module-technical-spec.md`: три финальные модели

```text
A. MoGen pretrained (mouse_mixed) → fine-tuned на шипиках
B. MoGen random init → обучение на шипиках
C. PointNeXt → VAE → SIREN-SDF
```

с общей инфраструктурой, общим split, общим reconstruction backend (mesh) и единым evaluation.

## 0. Факты, найденные при подготовке плана (влияют на решения)

| Факт | Следствие для плана |
| --- | --- |
| Официальный MoGen — `google-research/connectomics`, `connectomics/mogen/` — написан на **JAX/Flax**; чекпоинты — **Orbax (OCDBT)** в `gs://mogen-release/models/<name>/best_checkpoints/<step>/` + `config.json`. Лицензия Apache-2.0. | Официальный smoke test (этап 4.1) требует отдельного JAX-окружения. JAX с GPU официально поддерживается только на Linux — на нативной Windows только CPU (GPU — через WSL2). |
| `mouse_mixed/config.json`: `pfty_point_dim=128, pfty_latent_dim=256, pfty_n_latents=256, pfty_n_blocks=4, pfty_n_subblocks=2, pfty_n_heads=8, pfty_k_nn=16` — совпадает с параметрами из ТЗ. `n_points=8192`, `schedule=sample_schedule=cosine_2.0`, `sample_steps=100`, `optimizer=prodigy, lr=0.5, clip_grad=0.1, polyak_decay=0.999`, `jitter_std=16.0`, `do_rotate=true`, `coord_scale=1.0`, чекпоинт шага 750000 (~224 МБ). | Архитектура маленькая (≈200 строк: RMSNorm, attention с QK-нормализацией, MLP с GELU-tanh, sinusoidal time embedding, kNN-токенизатор). **Решение: порт PointInfinity на PyTorch + конвертация весов Flax → PyTorch с проверкой численной эквивалентности** против официальной JAX-реализации. Тогда все три модели обучаются в одном стеке на RTX 4090 под Windows. |
| Модель имеет вход `cond` (`Dense(latent_dim)` → дополнительный latent token); в `mouse_mixed` это `cond_mode=mean_cov_mst_leaves` + индекс датасета, а официальный inference подаёт **нулевой** `cond`. Время `t` тоже подаётся как дополнительный latent token. | Безусловная генерация = нулевой `cond` той же размерности, что в чекпоинте (веса `Dense` сохраняются); это же место — будущий вход `e(c)` для условной генерации (требование ТЗ). |
| Детали, критичные для численной эквивалентности порта: kNN при `N ≤ 8192` **включает саму точку** (первый сосед со смещением 0), при `N > 8192` — исключает; `nn.gelu` во Flax — tanh-аппроксимация; `RMSNorm` eps=1e-6; attention — с нормализацией q/k (`RMSNorm + bias`), выходная проекция и последний слой MLP инициализированы нулями. | Фиксируются в тесте эквивалентности (этап 4.3). |
| Масштаб координат MoGen: `coord_scale=1.0`, т.е. данные в их пайплайне уже в «модельных» единицах; ТЗ предлагает `x_MoGen = x_µm / 10` как стартовую гипотезу. `jitter_std=16` в тех же единицах. | Масштаб проверяется на этапе 4.1 по статистикам demo-данных (`demo_archive.zip`) и на zero-shot (4.4), затем фиксируется в конфиге. |
| `open3d` (pip, macOS arm64) не импортируется без Homebrew-`libusb`, а его `create_from_point_cloud_poisson` не имеет `point_weight`/`samples_per_node`, которые требует ТЗ. | Screened Poisson — через **`pymeshlab`** (`generate_surface_reconstruction_screened_poisson`: `depth, scale, samplespernode, pointweight, iters, ...` — официальная реализация Kazhdan & Hoppe). Оценка нормалей (PCA + согласование ориентации по графу соседства + глобальный flip наружу) — своя реализация на numpy/scipy, как описано в ТЗ. |
| Реальный объём Minnie65: ~1030 шипиков/нейрон (`spines/`; папка `spines_neurd/` — другая сегментация, пайплайн её не использует) → ~0.76 млн шипиков на 735 нейронов, ~1.4 ТБ выхода предобработки. | Прежняя оценка в `CLAUDE.md` (3.7 млн / 7 ТБ) была завышена из-за `spines_neurd/`; вывод «выход — на NAS, не на `C:`» остаётся в силе. |

## 1. Где что выполняется

- **mac** — только разработка кода, синтаксические/импортные проверки и крошечные тесты: синтетические фигуры (сфера и т.п.) и не более ~5 реальных шипиков, в одном процессе. Никаких фоновых/многопроцессных прогонов и обучения.
- **Windows (RTX 4090)** — полная предобработка, построение split, калибровка reconstruction backend, JAX smoke test MoGen, конвертация весов, все обучения и генерация.
- Для разработки на mac нужен маленький набор уже предобработанных шипиков (варианты — §6).

## 2. Структура кода

ТЗ предлагает `src/models/…`, `src/reconstruction/…`, `src/evaluation/…`. По правилам репозитория (`docs/project-architecture.md`) нельзя заводить общие top-level пакеты в `src/`, код проекта живёт в `src/neuron_model/`. Внутренняя структура из ТЗ сохранена, но под одним подпакетом:

```text
src/neuron_model/spine_generation/
  experiment/        # YAML-конфиги, seed, run dir, логирование, checkpoint/resume, сохранение samples
  data/              # manifest → индекс шипиков, split, torch Dataset'ы (point cloud, SDF)
  reconstruction/    # marching_cubes, estimate_normals, screened_poisson, mesh_validation
  evaluation/        # geometry/morphometrics/distribution/diversity/memorization/bootstrap/report (этап 7)
  models/
    spine_vae/       # pointnext_encoder, latent, siren_sdf_decoder, losses, model, train, generate
    spine_mogen/     # pointinfinity (порт), flow (schedule, ODE solver), weights_convert, dataset, train, infer
configs/neuron-model/
  data/splits_v1.yaml
  reconstruction/{poisson,marching_cubes}.yaml
  vae/{baseline,final}.yaml
  mogen/{pretrained_finetune,scratch_pilot,scratch_final}.yaml
  evaluation/default.yaml
scripts/neuron_model/        # тонкие CLI-обёртки (build_splits, calibrate_reconstruction, train_*, generate_*)
data/processed/neuron-model/splits/splits_v1.parquet
runs/training/neuron-model/<experiment_id>/{config.yaml, checkpoints/, logs/, samples/, meshes/, metrics/}
runs/analysis/neuron-model/<run_id>/     # калибровки reconstruction, evaluation-отчёты
tests/neuron_model/                      # pytest: синтетика + ≤10 реальных шипиков
```

`configs/` и `tests/` — новые top-level папки; отмечено в `docs/project-architecture.md`. `runs/` и `models/checkpoints/` добавлены в `.gitignore`.

## 3. Этапы

### Этап 1. Базовая структура проекта для модельных экспериментов

- Подпакет `src/neuron_model/spine_generation/` со структурой выше, `configs/neuron-model/`, `scripts/neuron_model/`.
- Зависимости: `torch` (mac — conda-forge `pytorch` с MPS; Windows — pip-колесо с CUDA 12.x), `pymeshlab` (pip). Позже: `prodigyopt` (Prodigy для PyTorch), `scikit-image` уже есть (Marching Cubes).
- `.gitignore`: `/runs/`, `/models/`.

Приёмка: пакет импортируется, конфиги читаются, документация обновлена.

### Этап 2. Загрузка результатов предобработки и экспериментальная инфраструктура

Данные:
- `SpineIndex` — чтение одного или нескольких `manifest.parquet`, фильтр `train_eligible`, проверка существования файлов, единый ключ шипика `dataset/neuron/limb/branch/spine`.
- **Split v1** (`build_splits`): групповой split по `neuron_id` (для Minnie/H01 это самый высокий доступный уровень группировки; `animal_id`/`sample_id` в метаданных отсутствуют), доли 80/10/10 по группам, мягкая стратификация по `dataset`/`species`/`health`/`cell_type_binary`/`compartment` без нарушения группировки (`no leakage > exact stratification`), фиксированный seed, отчёт о фактическом числе шипиков в каждой части. Сохраняется `splits_v1.parquet` (`spine_key, group_id, split` + версия/хэш) — один для всех моделей; производные представления шипика наследуют его split.
- Датасеты: `PointCloudDataset` (выбор размера 2048/4096/8192, случайный из 4 вариантов ресемплинга, jitter в физических единицах, опциональные нормали, без поворотов — локальная СК имеет биологический смысл), `SDFDataset` (point cloud для энкодера + minibatch query points с раздельной выборкой surface/near/uniform).
- Масштаб координат — один глобальный коэффициент на модель (nm → модельные единицы), без per-object нормализации; хранится в конфиге и в каждом артефакте.
- `FrozenBBox` — единый bbox по train + margin, считается один раз и замораживается (требование ТЗ для Marching Cubes).

Инфраструктура:
- YAML-конфиги с `defaults`/наследованием и override из CLI; итоговый конфиг сохраняется в run dir.
- `seed_everything` (python/numpy/torch, детерминированные генераторы для DataLoader).
- Run dir `runs/training/neuron-model/<experiment_id>/` + `run_info.json` (`experiment_id, git_commit, dataset_version (config_hash предобработки), split_version, config, random_seed`).
- Логирование скалярных метрик в JSONL/CSV (без обязательной зависимости от TensorBoard/W&B; можно подключить позже).
- Checkpoint manager: `latest`, `best_<metric>` (несколько отслеживаемых метрик, в т.ч. `best_val_loss` и `best_morphology_metric` для MoGen), модель + оптимизатор + scheduler + EMA + GradScaler + RNG-состояния + шаг/эпоха; атомарная запись; **resume** с восстановлением всего состояния.
- Сохранение промежуточных samples (raw generator output, raw mesh, postprocessed mesh, generation config, seed, checkpoint id — раздельно, как требует ТЗ).
- Общий training loop helper: mixed precision (CUDA — bf16/fp16, mac — fp32), gradient accumulation, global gradient clipping, EMA, периодическая валидация/сэмплинг.

Приёмка: юнит-тесты на синтетике (split без утечки групп, resume восстанавливает бит-в-бит состояние на игрушечной модели, конфиги/seed воспроизводимы); проверка загрузчиков на ~5 реальных шипиках.

### Этап 3. Реконструкция и валидация mesh + план калибровки

Реализация:
- `marching_cubes`: оценка SDF по регулярной сетке в замороженном bbox **chunk-ами** (`128³` — валидация, `256³` — финал), `skimage.measure.marching_cubes(level=0)`, флаг `surface_touches_grid_boundary`.
- `estimate_normals`: PCA по `k_normal` соседям (20–50) → согласование знаков по MST графа соседства (tangent-plane propagation, Hoppe et al.) → глобальный flip наружу (по знаку объёма/направлению от центра).
- `screened_poisson`: `pymeshlab`, параметры `depth, point_weight, samples_per_node, scale` из конфига.
- `postprocess_mesh`: удаление малых компонент, вырожденных граней, исправление ориентации — **только это**; крупные ошибки формы не чинятся.
- `mesh_validation`: `watertight, manifold, n_connected_components, self_intersections, degenerate_faces, genus, positive_volume, surface_touches_bbox` (переиспользуется `spine_geometry.mesh_geometry_report`/`self_intersection_check`), агрегаты `valid_mesh_rate, watertight_rate, single_component_rate, genus0_rate, self_intersection_rate`; raw и postprocessed оцениваются раздельно.
- Сохранение артефактов с именами из ТЗ (`generated_mesh_raw.off`, `generated_mesh_postprocessed.off`, `poisson_config.json`, `sdf_evaluation_config.json`, …).

План калибровки (выполняется на Windows **до** обучения генераторов, только на validation-части split):
1. **Poisson**: `local_sealed_mesh → pointcloud_8192 (готовый из предобработки) → нормали (оценённые, НЕ истинные — как будет у MoGen) → Screened Poisson → mesh`; сетка параметров `depth ∈ {6,7,8,9}`, `point_weight ∈ {0,2,4,10}`, `samples_per_node ∈ {1,1.5,3}`, `scale ∈ {1.1,1.25}`, `k_normal ∈ {20,30,50}`. Контрольный прогон с истинными нормалями отделяет ошибку нормалей от ошибки Poisson.
2. **Marching Cubes**: `local_sealed_mesh → точный SDF на сетке (winding number из spine_sampling) → Marching Cubes → mesh` при `128³/256³(/512³)` в замороженном bbox — нижняя граница ошибки для VAE.
3. Метрики сравнения с исходным mesh: Chamfer-L1/L2 и Hausdorff (95-й перцентиль) по 2048/8192 точкам, F-score@τ, отклонения Volume/Area, valid_mesh_rate, genus0_rate, и **отдельно тонкая шейка** (Chamfer в нижней по оси X части шипика у основания; ТЗ прямо указывает риск сглаживания шейки) + морфометрики `Length/Volume/Area/OpenAngle`.
4. Выбор: минимальная ошибка при `valid_mesh_rate≈1`; параметры замораживаются в `configs/neuron-model/reconstruction/*.yaml` (с хэшем) до финального тестирования. Если Poisson систематически теряет шейку — эскалация к Shape As Points (по ТЗ, только в этом случае).

### Этап 4. MoGen

1. **Официальный smoke test** (отдельное conda-окружение `mogen-jax` с JAX CPU, Windows или WSL2): клон `google-research/connectomics` с фиксированным commit, загрузка `mouse_mixed`, официальный demo/inference на небольшом числе сэмплов, сохранение baseline output + версии кода/чекпоинта. Здесь же — статистики координат demo-данных для выбора масштаба.
2. **Экспорт весов** из Orbax в `npz` (скрипт в том же окружении; `params` и `ema_params` отдельно).
3. **Порт на PyTorch** `PointInfinity` + `cosine_2.0` schedule + ODE-solver (в точности как в `flow_matching/utils.generate_samples`, ТЗ указывает midpoint) + тест эквивалентности: одинаковые входы/шум → совпадение выхода сети (≈1e-5) и сгенерированных облаков против JAX-референса.
4. **Zero-shot** в нашем пайплайне: адаптер масштаба координат, стабильность ODE, диапазоны, Poisson-меши, первая оценка domain shift (не модель шипиков — только проверка адаптера).
5. **Fine-tuning** на spine point clouds 8192 (full fine-tuning, LR ниже чем для scratch; Prodigy/AdamW; EMA 0.999; grad clip 0.1; jitter; без поворотов; checkpoints `latest/best_val_loss/best_morphology_metric`). Staged fine-tuning — резерв.

### Этап 5. `PointNeXt → VAE → SIREN-SDF` (фундамент)

- `pointnext_encoder`: FPS + ball query/kNN строятся на лету (радиусы/k/глубина — из конфига), stage widths `[64,128,256,512]`, InvResMLP-блоки, max(+avg) pooling → `h`; `use_normals`.
- `latent`: `μ`, `log σ²`, reparameterization, KL, статистики (`mean_abs_mu`, `mean_std_z`, `active_latent_dimensions`), KL warm-up и free bits.
- `siren_sdf_decoder`: `[q, z(, e(c))]` → SIREN с инициализацией Sitzmann et al. (`first_omega_0=30`), линейный выход.
- `losses`: SDF (L1/SmoothL1), surface, eikonal (через autograd по `q`), normal, KL — с весами и поэтапным включением.
- `model`, `train` (на общей инфраструктуре этапа 2), `generate` (`z=μ` для реконструкции, `z~N(0,I)` для prior → Marching Cubes этапа 3).
- Порядок: baseline `SDF+KL` → +surface → +eikonal → +normal.

### Этап 6. MoGen с нуля
Короткий pilot → продление при осмысленных loss/samples/metrics; строго те же split/число точек/масштаб/solver/reconstruction/evaluation, что у fine-tuned; дополнительно — скорость сходимости и чувствительность к seed.

### Этап 7. Evaluation и финальное сравнение
Морфометрики сгенерированных mesh (переиспользуется этап 8 предобработки; **`JunctionArea`/`Area` для сгенерированного mesh требуют алгоритма определения основания шипика** — открытый вопрос ТЗ, реализовано как шаг E — `evaluation/attachment.py`, см. §7; ось шипика в локальной СК — `+y`, не `x`), W1/KS/Energy, MMD в стандартизованном морфометрическом пространстве, `ΔR` корреляций, MMD-CD/COV-CD/1-NNA-CD, diversity, memorization (train vs test nearest), классификатор real-vs-generated, bootstrap-интервалы, единый отчёт; `1000 samples × 5 seeds` на модель.

## 4. Оценка объёма

| Этап | Объём кода | Где выполняется | Блокеры |
| --- | --- | --- | --- |
| 1 | малый | mac | — |
| 2 | средний | mac (тесты на синтетике + ≤5 шипиков) | маленький набор предобработанных шипиков для проверки загрузчиков |
| 3 | средний | код — mac (синтетика); калибровка — Windows | полная предобработка validation-части |
| 4.1–4.2 | малый | Windows/WSL2 (JAX CPU) | JAX-окружение |
| 4.3 | средний | код — mac; эквивалентность — Windows | экспорт весов (4.2) |
| 4.4–4.5 | средний | Windows (GPU) | 3 (Poisson заморожен), 2 (split) |
| 5 | большой | код — mac (синтетика); обучение — Windows | 2, 3 |
| 6 | малый (переиспользует 4) | Windows | 4.5 |
| 7 | большой | Windows | 3, 4, 5 |

## 5. Открытые вопросы

1. ~~Единицы координат MoGen (стартовая гипотеза ТЗ `/10 µm`)~~ — **проверено эмпирически, гипотеза ТЗ была ~4-5x заниженной**, см. ниже.
2. ~~Алгоритм основания шипика для `JunctionArea`/`Area` сгенерированных mesh~~ — **реализован (шаг E, 2026-10-05)**, см. §7; уточнить на сотнях реальных шипиков (`finder_bias.json`) и на первых реальных сгенерированных mesh.
3. Оптимизатор для fine-tuning в PyTorch: `prodigyopt` (аналог Prodigy из optax) — нужна проверка поведения на малом LR.
4. Нужен ли `animal_id`-уровень группировки split (если появится в метаданных LabID/H01 — повышаем уровень, версия split `v2`).

## 6. Данные для разработки на mac

Для проверки загрузчиков/датасетов нужны реальные предобработанные шипики (с `manifest.parquet`, point clouds, SDF). Варианты:
1. **Скопировать с Windows** папку `preprocessed/minnie65/` для одного branch (≈10–30 шипиков, ~20–60 МБ) вместе с `manifest.parquet` (рекомендуется — ничего не считается на mac).
2. Прогнать на mac полную предобработку для **≤5 шипиков** (как в `spine-preprocessing-visual-check.ipynb`, `limit=5`, `workers=1`).

Модели и реконструкция на этапах 1–3 тестируются на синтетике и без реальных данных.

## 7. Статус реализации (2026-09-29)

Реализовано и покрыто тестами (`tests/neuron_model/`, 79 тестов (этап 7 — 29 из них) на синтетике и поддельном мини-датасете, ~20 с, один процесс):

| Этап | Модули | Что проверено тестами |
| --- | --- | --- |
| 1 | `src/neuron_model/spine_generation/` (каркас), `configs/neuron-model/`, `scripts/neuron_model/`, `.gitignore` (`/runs/`, `/models/checkpoints/`), зависимости (`pytorch` из conda-forge на mac, `pymeshlab`, `pip_win_torch_requirements.txt`) | импорт, конфиги |
| 2 | `experiment/` — `config` (наследование `defaults`, CLI-override, **`1e-4` читается как число**, в PyYAML по умолчанию это строка), `seed`, `run` (`run_info.json`: git commit, dataset/split version, seed, версии библиотек), `metrics_logger` (JSONL, обрезка «будущего» при resume), `checkpoint` (`latest` + `best_<metric>` min/max, атомарная запись), `training` (device, autocast только на CUDA, EMA, накопление градиента, клиппинг), `artifacts` (имена файлов из ТЗ, PLY/OFF), `loop` (общий цикл обучения); `data/` — `index` (manifest → индекс, перенос путей Windows→mac), `splits` (групповой split v1), `datasets` (point cloud / SDF, глобальный `coord_scale`, jitter, без поворотов), `bbox` (замороженный bbox) | resume бит-в-бит воспроизводит непрерывное обучение (модель + EMA); накопление градиента = полный батч; best-чекпоинты; split без утечки групп, стратификация 40/5/5 из 100; пути `O:\...` читаются на mac; воспроизводимость DataLoader |
| 3 | `reconstruction/` — `marching_cubes` (chunked, флаг касания границы), `estimate_normals` (PCA + MST-ориентация + наружу), `screened_poisson` (`pymeshlab`), `postprocess` (только мелкие правки), `mesh_validation` (переиспользует QC предобработки), `calibration`; `evaluation/geometry_metrics` (Chamfer, Hausdorff-95, F-score, зона шейки); `scripts/neuron_model/{build_splits,calibrate_reconstruction}.py` | MC на сфере (объём ±3 %, ориентация наружу — **поймана и исправлена ошибка с лишним flip граней**); нормали на сфере/капсуле (0 % перевёрнутых, < 5°); валидация/агрегаты; калибровка MC на синтетическом шипике (ошибка падает с ростом разрешения) |
| 4 | `models/spine_mogen/` — `pointinfinity` (порт на PyTorch), `flow` (`cosine_2.0`, loss, midpoint), `weights` (Flax ↔ PyTorch), `train`, `infer`; `scripts/neuron_model/{download_mogen_checkpoint,export_mogen_weights,check_mogen_parity,train_mogen,generate_mogen}.py`; `configs/neuron-model/mogen/*` | перестановочная эквивалентность; нулевой `cond` = без `cond`; kNN включает саму точку при N ≤ 8192; расписание = формула MoGen; midpoint точен для постоянного поля и 2-го порядка; конвертация весов биективна; сквозное обучение крошечной модели (накопление, EMA, val, сэмплы, чекпоинты, resume); сэмплы зависят только от `(seed, i)` |
| 5 | `models/spine_vae/` — `pointnext_encoder`, `latent`, `siren_sdf_decoder`, `losses`, `model`, `train`, `generate`; `scripts/neuron_model/train_vae.py`; `configs/neuron-model/vae/{baseline,final}.yaml` | инвариантность энкодера к перестановке; SIREN-инициализация держит std активаций ≈0.7 на всех слоях; KL/free bits/warm-up; eikonal/normal = 0 на точном SDF сферы; переобучение на одной сфере → замкнутый mesh правильного объёма; сквозной цикл обучения с метриками латента и `mesh_validity_on_validation_samples` |
| 7 | `src/neuron_model/spine_morphometrics.py` (общий расчёт 12 метрик — один код для этапа 8 предобработки и для сгенерированных mesh); `evaluation/` — `distribution_metrics` (W1 / W1, нормированный на std train / KS / Energy, гистограммы `OldChordDistribution`, `Standardizer` по train, RBF-MMD² с σ по median heuristic на train, permutation p-value, ΔR Pearson/Spearman), `point_cloud_metrics` (Chamfer-матрица на torch, MMD-CD/COV-CD/1-NNA-CD, дескрипторы облаков + Chamfer с пред-отбором кандидатов), `diversity`, `memorization` (DCR), `classifier` (real-vs-generated), `bootstrap` (CI + агрегация по seed), `attachment` (шаг E: основание шипика сгенерированного mesh), `morphometrics` (реальные из `morphometrics.json`, сгенерированные через `AttachmentFinder`, `finder_bias`), `inputs` (раскладка `seed_<s>/sample_<i>`, кэш train-облаков в один memmap `.npy`), `report` (`EvaluationConfig`, `RealReference`, `evaluate_model`, `summary.csv/md`, `report.json`); `scripts/neuron_model/evaluate_generators.py`; `configs/neuron-model/evaluation/default.yaml` | рефактор `_morphometrics_worker` на общий модуль даёт те же числа на 10 реальных шипиках (макс. отн. расхождение 8e-16); каждая метрика — на синтетике «хорошая модель vs коллапс/копирование» (MMD, 1-NNA, COV, diversity, DCR ловит копии, классификатор); Chamfer vs brute force; пред-отбор кандидатов находит точный ближайший; сквозной отчёт ранжирует хорошую модель выше коллапсирующей; загрузчик сгенерированных наборов не теряет сэмплы без mesh; кэш облаков отказывается работать с чужим списком шипиков; finder основания — точная площадь/центр на цилиндре, fallback на заострённом основании |

**Не проверено на ноутбуке (нужна Windows-машина):**

- **Screened Poisson** (2 теста пропускаются). На macOS pip-`pymeshlab` несёт свою `libomp`, conda-OpenBLAS (numpy) — свою; в одном процессе это `OMP: Error #15` / segfault при любом порядке импорта. conda-forge-`pymeshlab` на mac есть, но его установка не решилась без изменения окружения, а на win-64 его нет. На Windows conda-numpy использует Intel OpenMP, а pymeshlab — другой рантайм, поэтому конфликта ожидать не стоит, **но это первое, что нужно проверить**: `python -m pytest tests/neuron_model -rs` — пропусков быть не должно.
- Скорость и память на реальных 8192-точечных облаках (dense kNN — 256 МБ на сэмпл; FPS в PointNeXt — последовательный цикл).

**Проверено на mac, численная эквивалентность порта MoGen подтверждена:**

- Официальный чекпоинт `mouse_mixed` (шаг 750000) скачан (`download_mogen_checkpoint.py`, 226 МБ), отдельное JAX-окружение `mogen-jax` (python 3.11, CPU) поднято, `google-research/connectomics` склонирован (`data/external/connectomics`, commit `b722f858`). Два недостающих транзитивных импорта официального кода закрыты без правки самого клона кода Google (кроме одного шим-пакета — см. ниже): `pip install jaxkd ott-jax`.
  - `ott-jax` на PyPI импортируется как `import ott`, а код `reorder.py` делает `import ott_jax as ott` — добавлен локальный шим-пакет `data/external/connectomics/ott_jax/__init__.py` (реэкспорт `ott`); используется только для опциональной OT-переупорядочки (`reorder_type='ot'`), которую `mouse_mixed` не использует (`combine_z=1`), так что дрейф API `sinkhorn.solve` в новых версиях `ott-jax` значения не имеет.
- `export_mogen_weights.py` отработал с первого раза (614 массивов, `cond_dim=22`).
- `check_mogen_parity.py` нашёл реальный баг нашего порта: формула kNN-расстояний `sq_a + sq_b - 2·a·b` (алгебраически верна, но вычитает два больших близких числа) теряла точность в fp32 ровно там, где два соседа почти совпадают — из-за этого иногда выбирался не тот ближайший сосед, что в оригинале (`connectomics.jax.spatial.squared_distances` считает расстояние напрямую через `(a-b)**2`, без этой неустойчивости). После замены формулы в `pointinfinity.knn_indices` на прямую разность (`src/neuron_model/spine_generation/models/spine_mogen/pointinfinity.py`) velocity/trajectory совпали с эталоном (`7e-5`/`5e-4` против порогов `1e-4`/`1e-3`; раньше было `5e-2`/`1.5e-3`). Регрессия закреплена тестом `test_knn_matches_direct_distance_for_near_degenerate_points`.

**Масштаб координат MoGen сверен с официальными данными (2026-10-05).** Смоук-тест реальных данных (§6 выше, 10 шипиков ветки `limb_000/branch_000` нейрона `864691134886335738`) дал реальные размеры шипика в нм; эти же статистики посчитаны на выгруженных официальных эмбеддингах обучающей выборки MoGen (`mogen-release.web.app` → `demo_archive.zip` → `debug/embs/emb_train_8192_train_mst.npz`, формула `simple_embs` из `mogen_demo.ipynb`, 16384 примеров):

| метрика (формула `simple_embs`) | наши шипики (нм), 10 шт. | официальный train MoGen (модельные единицы), 16384 шт. | implied `coord_scale` |
| --- | --- | --- | --- |
| `mean_max_dist` (среднее макс. попарное расстояние) | 2917.1 | 1.332 | `4.57e-4` |
| `cov_eigval` (std по 3 главным компонентам) | [144.3, 220.7, 811.2] | [0.051, 0.143, 0.485] | `3.5e-4` / `6.5e-4` / `6.0e-4` |
| `mean_min_dist` (среднее расстояние до ближайшего соседа) | 14.8 | 0.002 | `1.35e-4` — не используется, см. ниже |

`mean_max_dist`/`cov_eigval` — метрики общего размера/разброса облака точек, их оценки `coord_scale` согласованы между собой (`3.5e-4`–`6.5e-4`). `mean_min_dist` — метрика локальной плотности точек (зависит не от масштаба, а от площади поверхности при фиксированном числе точек: шипики — компактные объекты с относительно развитой/бугристой поверхностью (шейка+головка), в отличие от вытянутых фрагментов нейритов MoGen), поэтому её оценка масштаба закономерно расходится и не учитывается.

Стартовая гипотеза ТЗ (`x_nm · 1e-4`, "`x_MoGen = x_um/10`") оказалась **примерно в 4–5 раз заниженной**. Рабочее значение обновлено на `coord_scale: 5.0e-4` (`configs/neuron-model/mogen/_common.yaml`) — середина согласованного диапазона `3.5e-4`–`6.5e-4`. Косвенное подтверждение того же порядка: `jitter_std=16` в `mouse_mixed`/демо-конфиге явно прокомментирован как `# nm` и почти совпадает с реальным средним расстоянием до ближайшего соседа в наших шипиках (14.8 нм) — то есть джиттер в оригинале тоже задаётся в сырых нанометрах, до их (невидимого нам) внутреннего шага нормализации в единицы модели, что согласуется с общей картиной (`coord_scale=1.0` в их конфиге применяется уже к ПРЕДварительно отмасштабированным данным, а не к сырым нм).

На 10 шипиках из одной ветки — не статистически надёжная оценка; при полной предобработке на Windows стоит пересчитать эти же 4 числа на репрезентативной выборке (сотни-тысячи шипиков из разных нейронов) и уточнить `coord_scale`, прежде чем обучать/дообучать MoGen по-настоящему. Файлы (`data/external/mogen_demo/`, скачанный демо-архив ~224 МБ + ноутбук) сохранены локально для повторного использования методики, gitignore-ятся.

**Этап 7 (evaluation) — решения, отличающиеся от буквального текста ТЗ (2026-10-05):**

- **Memorization через DCR, а не «ближе к train, чем к test».** Буквальная проверка требует подвыборки train до размера test (иначе 8× больший train «ближе» случайно), и эта подвыборка разбавляет сигнал: скопированный train-шипик попадает в подвыборку с вероятностью |test|/|train|, поэтому чистый копировщик даёт ~`0.5 + 0.5·|test|/|train|` (≈0.56 при 8:1), а не 1 — на синтетике копии действительно получали 0.55 и не отличались от честной модели. Основная проверка — DCR (distance to closest record): расстояние сгенерированного шипика до ближайшего во **всём** train сравнивается с тем же расстоянием для реальных test-шипиков (`dcr_median_ratio` ≈1 норма, ≪1 — копирование; `near_copy_rate` против 5%-квантиля test; односторонний Mann–Whitney). Буквальная проверка ТЗ оставлена в отчёте как вторичная, вместе с ожидаемым значением для копировщика. В морфометрическом пространстве — KD-tree по всему train; в Chamfer-пространстве — k=50 кандидатов по дешёвым 7-D дескрипторам облака, затем точный Chamfer (train-облака один раз собираются в локальный memmap `real.train_cloud_cache`, т.к. NAS ~1 МБ/с).
- **Chamfer без нормировки каждой формы** (в отличие от большинства статей): локальная СК и физический размер — часть того, что генератор должен воспроизвести. Только один глобальный множитель `chamfer_scale=1e-3` (нм → мкм) для читаемых чисел. У Chamfer есть «пол» шума семплирования (два разных облака одной поверхности дают CD > 0), поэтому абсолютные MMD-CD сравниваются между моделями при одинаковом числе точек, а не с нулём.
- **σ для MMD и стандартизация фиксируются один раз по реальному train** (`RealReference`) — все модели меряются одной линейкой; W1 нормируется на std train по признаку.
- 1-NNA-CD требует равных размеров наборов → реальный и сгенерированный наборы подвыбираются до `surface_reference_size=1000`; в отчёте также точности по классам (коллапс виден как `accuracy_generated ≫ accuracy_real`).
- Сгенерированные сэмплы без mesh не выбрасываются молча: они остаются в отчёте как невалидные (`mesh_present=False`) и считаются против модели.

**Шаг E — основание шипика у сгенерированного mesh (`evaluation/attachment.py`, `DownFacingCapFinder`, 2026-10-05).** 8 из 12 метрик (`OpenAngle`, `CVD`, `AverageDistance`, `Length`, `LengthVolumeRatio`, `LengthAreaRatio`, `JunctionArea`, `Area`) требуют области крепления; у реальных шипиков она известна с этапа 1, у сгенерированного (замкнутого) mesh её нужно распознать. Ось шипика в локальной СК — **`+y`** (нормаль заплатки → `−y`, `x` — касательная к ветке; в первой версии плана ошибочно было написано `x≈0`).

- Исследование на 10 реальных шипиках: настоящая заплатка **не плоская** — седлом пересекает `y=0` (например, от −69 до +41 нм), поэтому сечение ровно по `y=0` режет саму заплатку (2 контура), а сечение «первый одиночный контур над 0» занижает `JunctionArea` на ~37% (систематически).
- Принятый метод: грань, через которую ось `y` входит в mesh снизу (луч вдоль `+y` через `x=z=0`), — затравка; заплатка = связная область граней вокруг неё с нормалью в пределах `max_tilt_deg=60°` от `−y`. `cap_area` = её площадь, `JunctionArea` = её проекция на `xz` (площадь, охваченная каймой, вдоль оси шипика), центр = центроид. Fallback (затравка не смотрит вниз — седло/заострённое основание): сечения `y=const` в полосе до 5% высоты — центр по нижнему одиночному контуру, площадь — максимальный одиночный контур.
- Валидация против истинной области этапа 1 — и на сырых `local_sealed` mesh, и на гладких MC-ремешах тех же шипиков (winding number → Marching Cubes, сетка 80³; ближе к выходу генераторов). Медиана |отн. ошибки|:

| mesh | OpenAngle | CVD | AverageDistance | Length | JunctionArea | Area |
| --- | --- | --- | --- | --- | --- | --- |
| сырые (8 — основной метод, 2 — fallback) | 1.9% | 0.4% | 0.6% | 0.4% | 3.7% | 0.2% |
| гладкие MC (10 — основной метод) | 9.3% | 9.6% | 4.7% | 0.7% | 12.6% (смещение +6%, макс. 65%) | 0.2% |
| гладкие MC с **истинными** центром/площадями | 11.0% | 8.6% | 4.4% | 0.2% | 0 | 0 |

  На гладких mesh ошибки OpenAngle/CVD/AverageDistance — от самой ремешировки (у «истины» на том же MC-mesh они такие же), finder сверх этого почти ничего не добавляет. Основной источник погрешности finder — `JunctionArea` (~12%, отдельные выбросы до 65%, когда основание переходит в «юбку» дендрита). Время — 3–35 мс на mesh.
- `evaluate_generators.py` сам считает `finder_bias.json` — finder на `finder_bias_spines=500` реальных test-шипиках против истины этапа 1 (медиана и 5–95% отн. ошибки по метрике): разрыв real-vs-generated по `JunctionArea`, сопоставимый с этим смещением, нельзя приписывать генератору. Реальные шипики в сравнении по-прежнему со своей истинной областью этапа 1. В отчёте у каждого набора — `attachment_found_rate` и разбивка по методу.

**Не реализовано (следующие этапы):** чекпоинт `best_morphology_metric` для MoGen (подключить `evaluate_generated_set` к валидации); у VAE нет отдельного CLI генерации (`generate_vae.py`, аналог `generate_mogen.py`) для протокола `1000 × 5 seed` — раскладка файлов (`SampleWriter`, `generated_mesh_{raw,postprocessed}.off`) уже совместима с `evaluate_generators.py`, нужен только скрипт-обёртка. Также пока нет упаковки данных для обучения на локальный диск: при измеренной скорости чтения NAS ~1 МБ/с чтение npz напрямую с `O:\` будет узким местом. Решать после того, как будет известна реальная скорость сети (Ethernet vs Wi-Fi) и итоговый объём train-части.

## 8. Порядок запуска на Windows

```powershell
conda activate neuron-model
cd <repo>
python -m pip install -r requirements\pip_win_requirements.txt        # pymeshlab
python -m pip install -r requirements\pip_win_torch_requirements.txt  # torch + CUDA
python -m pytest tests\neuron_model -rs                               # 79 passed, 0 skipped ожидается

# 1) после полной предобработки: split + bbox (один раз, замораживаются)
python scripts\neuron_model\build_splits.py --config configs\neuron-model\data\splits_v1.yaml "manifests=['O:/Datasets/Minnie65/preprocessed/minnie65/manifest.parquet']"

# 2) калибровка reconstruction (до обучения генераторов), затем перенести выбранное в reconstruction/*.yaml и frozen: true
python scripts\neuron_model\calibrate_reconstruction.py --config configs\neuron-model\reconstruction\calibration.yaml "data.manifests=['O:/Datasets/Minnie65/preprocessed/minnie65/manifest.parquet']" workers=16

# 3) MoGen: чекпоинт -> (JAX-окружение) экспорт весов + эталон -> проверка порта
python scripts\neuron_model\download_mogen_checkpoint.py --name mouse_mixed --out data\external\mogen
#   в отдельном окружении с JAX (CPU), см. этап 4.1:
#   python scripts\neuron_model\export_mogen_weights.py --checkpoint data\external\mogen\mouse_mixed\best_checkpoints\750000\train_state --train-config data\external\mogen\mouse_mixed\config.json --connectomics-repo <clone> --out data\external\mogen\mouse_mixed_750000
python scripts\neuron_model\check_mogen_parity.py --export data\external\mogen\mouse_mixed_750000
python scripts\neuron_model\generate_mogen.py --weights data\external\mogen\mouse_mixed_750000_weights.npz --config configs\neuron-model\mogen\pretrained_finetune.yaml --n-samples 16   # zero-shot
python scripts\neuron_model\train_mogen.py --config configs\neuron-model\mogen\pretrained_finetune.yaml "data.manifests=[...]"

# 4) VAE
python scripts\neuron_model\train_vae.py --config configs\neuron-model\vae\baseline.yaml "data.manifests=[...]"

# 5) оценка: 1000 сэмплов x 5 seed на модель (generate_mogen.py --n-samples 1000 --seed 1..5), затем одна команда на все модели
python scripts\neuron_model\evaluate_generators.py --config configs\neuron-model\evaluation\default.yaml --model mogen_ft=runs\training\neuron-model\<id>\meshes --model mogen_scratch=runs\training\neuron-model\<id>\meshes "data.manifests=['O:/Datasets/Minnie65/preprocessed/minnie65/manifest.parquet']"
```

JAX-окружение для этапа 4.1/4.2 (отдельное, чтобы не трогать `neuron-model`; проверено на mac, воспроизводимо и на Windows/WSL2):

```bash
conda create -n mogen-jax python=3.11 -y
conda activate mogen-jax
python -m pip install "jax[cpu]" flax optax orbax-checkpoint ml-collections etils absl-py tensorstore numpy jaxkd ott-jax
git clone https://github.com/google-research/connectomics data/external/connectomics   # зафиксировать commit (git log -1)
```

`jaxkd`/`ott-jax` — не входят в список ТЗ, но нужны транзитивно (`connectomics.jax.spatial`, `connectomics.mogen.reorder`). Для `ott-jax` дополнительно нужен шим-пакет `data/external/connectomics/ott_jax/__init__.py` (`from ott import *`) — PyPI-пакет `ott-jax` импортируется как `ott`, а код Google ждёт `ott_jax`; это не используется для `mouse_mixed` (`combine_z=1`), так что его не нужно чинить по-настоящему, только чтобы сам `import` не падал.

Официальный demo-ноутбук и `demo_archive.zip` — на странице проекта `mogen-release.web.app`.
