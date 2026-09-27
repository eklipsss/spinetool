# Техническое задание: модуль генерации геометрии дендритных шипиков

## 1. Цель этапа

Цель этапа — реализовать и сравнить две независимые генеративные модели геометрии дендритных шипиков:

1. вариационный автоэнкодер с кодировщиком PointNeXt и неявным SDF-представлением поверхности;
2. модель Flow Matching на основе архитектуры MoGen / PointInfinity с локальным k-NN-контекстом.

На текущем этапе **условная генерация по `species`, `health`, `cell_type`, `cell_type_binary`, `compartment` не используется**. Основная задача — проверить, способны ли выбранные архитектуры в целом воспроизводить распределение морфологически правдоподобных шипиков.

Архитектуры должны быть реализованы так, чтобы на следующем этапе можно было добавить условный вектор без полной переработки моделей.

Итогом каждой модели должен быть **полигональный mesh отдельного шипика в локальной системе координат**, а не только скрытый вектор, SDF или облако точек.

Реализуются две основные ветки:

```math
\boxed{
PointNeXt
\rightarrow
VAE
\rightarrow
SDF\;decoder\;(SIREN)
\rightarrow
Marching\;Cubes
\rightarrow
mesh
}
```

и

```math
\boxed{
MoGen/PointInfinity+kNN
\rightarrow
Flow\;Matching
\rightarrow
point\;cloud
\rightarrow
Screened\;Poisson
\rightarrow
mesh
}
```

---

# 2. Исходное состояние данных

К началу данного этапа для каждого шипика уже должны быть выполнены следующие операции:

1. исходный mesh шипика запаян;
2. ложные шипики детектированы;
3. исходный незапаянный mesh переведён в локальную систему координат;
4. ошибочно разделённые ветви дендрита объединены;
5. сохранена метаинформация:
   - `species`;
   - `health`;
   - `cell_type`;
   - `cell_type_binary`;
   - `compartment`;
6. рассчитаны морфологические характеристики:
   - `OldChordDistribution`;
   - `OpenAngle`;
   - `CVD`;
   - `AverageDistance`;
   - `LengthVolumeRatio`;
   - `LengthAreaRatio`;
   - `JunctionArea`;
   - `Length`;
   - `Area`;
   - `Volume`;
   - `ConvexHullVolume`;
   - `ConvexHullRatio`.

На данном этапе метаинформация сохраняется и используется для стратификации, анализа и будущего условного обучения, но **не подаётся в генеративные модели как условие**.

---

# 3. Дополнительная предобработка данных

## 3.1. Формирование единого канонического mesh

Для обучения необходимо выбрать одно основное представление каждого шипика:

```text
sealed_local_mesh
```

то есть **запаянный mesh, переведённый в локальную систему координат**.

Если текущий алгоритм отдельно запаивает mesh и отдельно переводит только исходный незапаянный mesh в локальную систему координат, необходимо добавить применение уже рассчитанного преобразования `global_to_local` к запаянному mesh.

Именно `sealed_local_mesh` используется как первичный геометрический источник для всех последующих представлений.

## 3.2. Сохранение преобразований координат

Для каждого шипика необходимо сохранять:

```text
global_to_local
local_to_global
```

в виде матриц `4x4`.

Также необходимо сохранять:

```text
physical_unit
model_scale
```

Рекомендуемая физическая единица — `µm`.

Преобразование в модельные координаты должно быть обратимым:

```math
x_{model}=\frac{x_{\mu m}}{s_{global}},
```

```math
x_{\mu m}=s_{global}x_{model}.
```

Критически важно: запрещено независимо нормировать каждый шипик к единичной сфере или единичному bounding box. Индивидуальная нормализация уничтожит реальные различия размера и сделает некорректными `Length`, `Area`, `Volume`, `JunctionArea`.

Допускается только единый масштаб для всего набора данных или отдельный глобальный масштаб, фиксированный для конкретной модели.

## 3.3. Проверка геометрической корректности mesh

Для каждого `sealed_local_mesh` выполнять автоматическую проверку:

```text
is_watertight
is_manifold
n_connected_components
has_self_intersections
n_degenerate_faces
has_consistent_winding
surface_area
volume
euler_characteristic
genus
```

Для замкнутой ориентируемой поверхности:

```math
\chi=V-E+F,
```

```math
\chi=2-2g,
```

где `g` — род поверхности.

Для обычного изолированного шипика после корректного запаивания ожидается:

```math
g=0.
```

Необходимо разделять `geometry_error` и `morphology_outlier`. Mesh исключается из обучения только при подтверждённой геометрической ошибке либо ложной сегментации, а не просто из-за редкой формы.

## 3.4. Сохранение искусственной области запаивания

При запаивании сохранить созданные искусственно грани:

```text
cap_face_indices
```

или:

```text
face_is_cap: bool[n_faces]
```

Это необходимо потому, что искусственная крышка не является реальной биологической поверхностью, может создавать технический признак и в дальнейшем должна использоваться как область стыка с дендритом.

На первом этапе обучения допустимо использовать полностью запаянный mesh, поскольку SDF требует замкнутой поверхности. При этом `cap_faces` обязательно сохранять и контролировать, не начинает ли генератор воспроизводить характерный артефакт алгоритма запаивания.

---

# 4. Преобразование mesh в облако точек

## 4.1. Нельзя использовать исходные вершины mesh

Запрещено использовать:

```python
points = mesh.vertices
```

как стандартное облако точек. Количество и плотность исходных вершин зависят от триангуляции и являются техническим признаком.

## 4.2. Семплирование по площади треугольников

Для грани `f_i` с площадью `A_i`:

```math
P(f_i)=\frac{A_i}{\sum_j A_j}.
```

После выбора треугольника случайная точка генерируется внутри него в барицентрических координатах.

## 4.3. Требуемые размеры point cloud

Подготовить:

```text
pointcloud_2048
pointcloud_4096
pointcloud_8192
```

Назначение:

- `2048` — быстрые эксперименты;
- `4096` — основной кандидат для PointNeXt/VAE;
- `8192` — основной формат для MoGen и экспериментов с предобученными весами.

Финальное значение PointNeXt/VAE выбирается между `2048` и `4096` по validation-качеству и стоимости обучения.

Для MoGen первым полноценным экспериментом использовать `8192`, поскольку это исходное разрешение MoGen, а k-NN-модификация делает модель частично зависимой от разрешения.

## 4.4. Нормали поверхности

Для каждой точки сохранить:

```text
xyz      [N, 3]
normal   [N, 3]
```

Для точки внутри треугольника предпочтительно интерполировать вершинные нормали и затем нормировать:

```math
n_i\leftarrow\frac{n_i}{\|n_i\|}.
```

PointNeXt получает:

```math
X\in\mathbb R^{N\times6}
```

в виде `[x, y, z, nx, ny, nz]`.

MoGen на первом этапе получает только `[x, y, z]`, чтобы максимально сохранить исходную архитектуру.

## 4.5. Несколько реализаций семплирования

Для train желательно иметь несколько независимых семплирований одного mesh. Стартово:

```text
K = 4
```

На каждой эпохе выбирается случайный вариант. Для validation/test используется фиксированный seed и одно фиксированное каноническое семплирование.

---

# 5. Данные для SDF

## 5.1. Определение SDF

```math
d(q)=
\begin{cases}
-\operatorname{dist}(q,\partial S), & q\in S,\\
0, & q\in\partial S,\\
+\operatorname{dist}(q,\partial S), & q\notin S.
\end{cases}
```

Поверхность:

```math
S=\{q:f(q)=0\}.
```

Для корректного знака mesh должен быть замкнутым и иметь согласованную ориентацию.

## 5.2. Набор query points

Для каждого mesh сформировать:

```math
Q=Q_{surface}\cup Q_{near}\cup Q_{uniform}.
```

### `Q_surface`

Точки на поверхности, для них `d(q)≈0`.

### `Q_near`

```math
q=p+\epsilon,
```

```math
p\in\partial S,
```

```math
\epsilon\sim\mathcal N(0,\sigma^2I).
```

Использовать несколько значений `sigma`.

### `Q_uniform`

Точки равномерно внутри общего 3D bounding box. Они нужны, чтобы модель учила знак и геометрию пространства не только вблизи поверхности.

## 5.3. Стартовое распределение query points

```text
surface       20%
near-surface  60%
uniform       20%
```

Доли должны храниться в конфигурации и при необходимости проверяться на validation.

## 5.4. Truncated SDF

Допускается использовать:

```math
\tilde d(q)=\operatorname{clip}(d(q),-\tau,\tau).
```

`tau` задаётся в общих модельных или физических единицах, но не отдельно для каждого объекта.

---

# 6. Формат подготовленного датасета

Рекомендуемая логическая структура:

```text
spine_id/
├── sealed_local_mesh.off
├── pointcloud_2048
├── pointcloud_4096
├── pointcloud_8192
├── sdf_samples
├── transform
├── metadata
├── morphometrics
└── quality
```

Минимальные поля:

```text
spine_id
dataset_id
sample_id
animal_id
neuron_id
branch_id
species
health
cell_type
cell_type_binary
compartment
geometry_valid
```

Если идентификатор отсутствует, сохранять `null`.

Для большого количества объектов рекомендуется отделить канонические mesh, табличный manifest и массивы для обучения, например:

```text
manifest.parquet
pointclouds.zarr
sdf.zarr
meshes/
```

---

# 7. Версионирование предобработки

После подготовки набора зафиксировать:

```text
preprocessing_version
sampling_seed
physical_unit
model_scale
point_count
sdf_config
mesh_qc_config
split_version
```

После начала финального сравнения нельзя незаметно менять preprocessing только для одной модели.

---

# 8. Модель 1: PointNeXt → VAE → SDF/SIREN → Marching Cubes

## 8.1. Уточнение терминологии

Выбор между `SDF` и `SIREN` не требуется.

**SDF** — способ представить поверхность.

**SIREN** — нейронная сеть с синусоидальными функциями активации, используемая для аппроксимации непрерывной функции.

Фиксируется:

```math
\boxed{
PointNeXt
\rightarrow
VAE
\rightarrow
SIREN\;decoder\;of\;SDF
\rightarrow
Marching\;Cubes
}
```

---

# 9. Общая схема VAE

Для реального шипика:

```math
X=\{(p_i,n_i)\}_{i=1}^{N}
```

кодировщик строит:

```math
h=E_\phi(X).
```

Затем:

```math
\mu=\mu_\phi(h),
```

```math
\log\sigma^2=\ell_\phi(h).
```

Скрытый вектор:

```math
z=\mu+\sigma\odot\epsilon,
```

```math
\epsilon\sim\mathcal N(0,I).
```

Декодер реализует:

```math
f_\theta(q,z)\rightarrow\hat d,
```

где `q=(x,y,z)` — произвольная координата пространства, `\hat d` — предсказанный SDF.

---

# 10. PointNeXt Encoder

## 10.1. Вход

Основная конфигурация:

```text
N = 4096
```

Тензоры:

```text
xyz      [B, N, 3]
features [B, N, 3]
```

`features` — нормали поверхности.

Предусмотреть `use_normals = true/false`.

## 10.2. Локальная иерархия

PointNeXt самостоятельно выполняет subsampling центральных точек, поиск локальных соседей, агрегацию локальных признаков и увеличение receptive field. Заранее сохранять локальные группы в preprocessing не требуется.

## 10.3. Стартовая конфигурация

```yaml
encoder:
  type: pointnext
  input_points: 4096
  input_features: 3
  base_width: 64
  stage_widths: [64, 128, 256, 512]
  global_feature_dim: 512
```

Число блоков, радиусы и число соседей вынести в конфигурацию.

## 10.4. Global pooling

После последнего уровня получить глобальное представление:

```math
h\in\mathbb R^{D_h}.
```

Использовать permutation-invariant pooling: `max pooling` или `max + average pooling`.

---

# 11. VAE latent space

Начальное значение:

```text
latent_dim = 128
```

При необходимости проверить `64/128/256` после получения стабильного baseline.

Реализовать два выхода:

```text
fc_mu
fc_logvar
```

Reparameterization:

```math
\sigma=\exp\left(\frac12\log\sigma^2\right),
```

```math
z=\mu+\sigma\odot\epsilon.
```

Для реконструкции можно использовать `z=mu`, для генерации:

```math
z\sim\mathcal N(0,I).
```

---

# 12. SIREN SDF Decoder

## 12.1. Вход

Для query point `q`:

```math
u=[q,z].
```

Размер входа: `3 + latent_dim`.

В будущем:

```math
u=[q,z,e(c)].
```

На текущем этапе условный блок отключён.

## 12.2. Архитектура

```yaml
decoder:
  type: siren_sdf
  hidden_dim: 256
  hidden_layers: 5
  output_dim: 1
  first_omega_0: 30
```

Скрытые слои:

```math
h_{l+1}=\sin\left(\omega_l(W_lh_l+b_l)\right).
```

Последний слой линейный. Использовать корректную SIREN-инициализацию.

---

# 13. Функция потерь VAE

Полная схема:

```math
\mathcal L_{VAE}
=
\lambda_{sdf}\mathcal L_{sdf}
+
\lambda_{surface}\mathcal L_{surface}
+
\lambda_{eik}\mathcal L_{eikonal}
+
\lambda_{normal}\mathcal L_{normal}
+
\beta\mathcal L_{KL}.
```

Первый baseline:

```math
\mathcal L_{baseline}=\mathcal L_{sdf}+\beta\mathcal L_{KL}.
```

После проверки добавить геометрические регуляризаторы.

## 13.1. SDF loss

```math
\mathcal L_{sdf}=\frac1M\sum_i|\hat d_i-d_i|.
```

Допускается Smooth L1.

## 13.2. Surface loss

```math
\mathcal L_{surface}=\frac1{|Q_{surface}|}\sum_i|f(q_i,z)|.
```

## 13.3. Eikonal regularization

```math
\mathcal L_{eikonal}
=
\mathbb E_q
\left(
\|\nabla_qf(q,z)\|_2-1
\right)^2.
```

## 13.4. Normal consistency

```math
\mathcal L_{normal}
=
1-
\frac{\nabla f(q)\cdot n}{\|\nabla f(q)\|\|n\|}.
```

Добавлять после стабильного baseline.

## 13.5. KL divergence

```math
\mathcal L_{KL}
=
-\frac12
\sum_j
\left(
1+\log\sigma_j^2-\mu_j^2-\sigma_j^2
\right).
```

Обязательно использовать `KL warm-up` или `free bits`, чтобы снизить риск posterior collapse.

---

# 14. Обучение VAE

Для каждого spine в batch:

1. загрузить `pointcloud_4096`;
2. передать его в PointNeXt;
3. получить `mu`, `logvar`;
4. семплировать `z`;
5. выбрать minibatch SDF query points;
6. декодировать SDF;
7. вычислить loss;
8. выполнить backpropagation.

Полный 3D grid на каждом train step не декодировать.

Логировать:

```text
train_total_loss
val_total_loss
sdf_loss
surface_loss
eikonal_loss
normal_loss
kl_loss
mean_abs_mu
mean_std_z
active_latent_dimensions
```

Периодически генерировать reconstruction validation spine и prior sample из `z~N(0,I)`.

---

# 15. VAE: преобразование результата в полигональный mesh

## 15.1. Генерация скрытого вектора

```math
z\sim\mathcal N(0,I).
```

## 15.2. Регулярная 3D-сетка

Определить общий bounding box по train с фиксированным margin. Он должен быть единым для всех объектов и зафиксирован до test.

Разрешения:

```text
128^3 — validation и быстрые эксперименты
256^3 — финальная генерация
```

`512^3` использовать только если `256^3` недостаточно для тонкой шейки.

## 15.3. Вычисление SDF

Для каждой grid point `q_j` вычислить:

```math
d_j=f_\theta(q_j,z).
```

Grid вычислять chunk-ами, чтобы не переполнять GPU.

## 15.4. Marching Cubes

После получения 3D массива `D[i,j,k]` запустить Marching Cubes с:

```text
iso_level = 0
```

Результат:

```text
vertices
faces
```

## 15.5. Обратное масштабирование

```text
grid coordinates
→ model coordinates
→ µm
```

В рамках модуля `S` основной результат сохраняется в локальных координатах.

## 15.6. Проверка mesh

Сохранять:

```text
generated_raw.off
generated_postprocessed.off
```

`raw` используется для честной диагностики. В postprocessing допускается удаление крайне маленьких компонент, вырожденных граней, исправление ориентации и лёгкое remeshing. Крупные ошибки модели нельзя исправлять молча.

## 15.7. Выход за bounding box

Если нулевой уровень SDF касается границы grid:

```text
surface_touches_grid_boundary = true
```

sample считается потенциально некорректным. Допускается повторная extraction с увеличенным bounding box, но событие логируется.

---

# 16. Модель 2: MoGen / PointInfinity + k-NN → Flow Matching

## 16.1. Общая цель

Необходимо адаптировать открытую реализацию MoGen для распределения изолированных дендритных шипиков.

MoGen исходно генерирует локальные фрагменты нейритов как point cloud из `8192` точек. В данной работе:

```text
один spine = один point cloud
```

---

# 17. Формат входных данных MoGen

Основной формат:

```text
[B, 8192, 3]
```

Содержимое: `x, y, z` в локальной системе координат шипика.

Нормали в Flow Matching на первом этапе не моделируются.

## 17.1. Масштаб для pretrained checkpoint

При использовании предобученной MoGen необходимо максимально сохранить исходный масштаб модели. В исходном MoGen одна модельная единица соответствует радиусу локального фрагмента `10 µm`.

Для эксперимента с pretrained weights первым вариантом использовать:

```math
x_{MoGen}=\frac{x_{\mu m}}{10}.
```

Это **не индивидуальная нормализация** и не уничтожает реальный размер шипика.

После генерации:

```math
x_{\mu m}=10x_{MoGen}.
```

Для обучения MoGen с нуля первоначально сохранить тот же масштаб, чтобы сравнение pretrained vs scratch не смешивалось с изменением численного представления.

Если окажется, что относительный размер шипиков ухудшает обучение, отдельным экспериментом допускается другой единый `global_scale`, но нельзя нормировать каждый объект независимо.

---

# 18. Flow Matching formulation

Реальный point cloud:

```math
x_1\sim p_{data}.
```

Начальный шум:

```math
x_0\sim\mathcal N(0,I).
```

Промежуточное состояние:

```math
x_t=(1-t)x_0+tx_1.
```

Целевая скорость базового flow:

```math
u_t=x_1-x_0.
```

Сеть аппроксимирует:

```math
v_\theta(x_t,t).
```

Loss:

```math
\mathcal L_{FM}
=
\mathbb E
\left[
\|v_\theta(x_t,t)-(x_1-x_0)\|_2^2
\right].
```

На данном этапе:

```text
conditioning = disabled
```

Архитектурный интерфейс для `c` сохраняется, но используется пустой/нулевой condition mask.

---

# 19. Time sampling

В основной реализации необходимо сохранить **modified cosine schedule из официальной реализации MoGen**.

Не заменять его линейным или uniform schedule до воспроизведения baseline.

Причина: в исходной работе линейный вариант показал худшее качество.

В конфигурации:

```yaml
flow:
  time_schedule: mogen_cosine
```

После успешного воспроизведения допускаются `linear` и `uniform` только как ablation.

---

# 20. Архитектура MoGen

Максимально близко сохранить исходный backbone.

Базовые параметры:

```yaml
model:
  point_token_dim: 128
  latent_token_dim: 256
  n_latent_tokens: 256
  n_stages: 4
  transformer_blocks_per_stage: 2
  n_attention_heads: 8
  knn_k: 16
```

## 20.1. Data stream

Каждая точка point cloud является data token.

Для точки `p_i` на каждом forward pass находятся `k=16` ближайших соседей:

```math
\mathcal N_k(i).
```

Для каждого соседа:

```math
\Delta p_{ij}=p_j-p_i.
```

Входные признаки:

```math
f_i=[p_i,\Delta p_{i1},...,\Delta p_{ik}].
```

При `k=16` координатный вход до токенизации имеет:

```text
3 + 16*3 = 51
```

признак.

k-NN вычисляется для текущего `x_t`, а не сохраняется заранее в preprocessing.

## 20.2. Tokenizer

Tokenizer преобразует локальные признаки каждой точки в point token:

```math
x_i\in\mathbb R^{128}.
```

Также добавляется embedding времени `e(t)`.

## 20.3. Latent stream

Используется фиксированный набор:

```text
256 latent tokens
```

размерности:

```text
256
```

Latent tokens являются обучаемыми параметрами.

## 20.4. Read cross-attention

На каждой стадии latent tokens считывают информацию из point tokens:

```math
Z\leftarrow CrossAttention(Q=Z,K=X,V=X).
```

## 20.5. Latent Transformer

После read-операции выполняются:

```text
2 Transformer blocks
```

с:

```text
8 attention heads
```

на каждой стадии.

## 20.6. Write cross-attention

```math
X\leftarrow CrossAttention(Q=X,K=Z,V=Z).
```

## 20.7. Количество стадий

Выполнить `4` последовательных стадии:

```text
read
→ latent transformer
→ write
```

## 20.8. Velocity head

Для каждой точки итоговый data token преобразуется в:

```math
\hat v_i\in\mathbb R^3.
```

Выход сети:

```text
[B, N, 3]
```

— предсказанное направление движения каждой точки.

---

# 21. Обучение MoGen

## 21.1. Baseline from scratch

В качестве референса исходной MoGen использовать:

```yaml
optimizer: Prodigy
learning_rate: 0.5
global_gradient_clip_norm: 0.1
ema_decay: 0.999
```

Эти значения являются **референсными**, а не обязательным вычислительным бюджетом проекта.

Обязательно поддержать:

```text
gradient_accumulation
mixed_precision
checkpointing
resume_training
EMA
```

Checkpoints:

```text
latest
best_val_loss
best_morphology_metric
```

## 21.2. Augmentation

В исходном MoGen применялись случайные повороты и небольшой Gaussian jitter.

В нашей задаче произвольные 3D-повороты по умолчанию **не использовать**, поскольку локальная система координат имеет биологический смысл.

Разрешённые augmentation:

```text
small Gaussian coordinate jitter
random surface resampling
```

Jitter задавать в физических единицах и не делать больше характерной погрешности/дискретизации данных.

---

# 22. Inference Flow Matching

Начальное состояние:

```math
x(0)\sim\mathcal N(0,I).
```

Решается ODE:

```math
\frac{dx}{dt}=v_\theta(x,t)
```

от `t=0` до `t=1`.

Основной solver — тот же midpoint solver, что и в MoGen.

Результат:

```text
generated_points [8192, 3]
```

После этого координаты возвращаются в `µm`.

---

# 23. Эксперимент с предобученной MoGen

Данный эксперимент обязателен и выполняется раньше полноценного обучения MoGen с нуля.

Из опубликованных checkpoints первым использовать:

```text
mouse_mixed
```

как наиболее близкий по типу исходной нейрональной морфологии.

## 23.1. Этап A — smoke test

До изменения архитектуры:

1. загрузить официальный checkpoint;
2. запустить официальный inference/demo;
3. проверить ожидаемые neurite point clouds;
4. зафиксировать версию кода и checkpoint;
5. сохранить baseline output.

Цель — отделить ошибки окружения от ошибок адаптации модели.

## 23.2. Этап B — zero-shot inference

Запустить pretrained checkpoint без дообучения.

Этот эксперимент **не считается полноценной моделью шипиков**. Ожидаемый output — фрагменты исходного распределения MoGen, а не изолированные spine.

Zero-shot используется для проверки:

```text
coordinate range
sampling
ODE stability
масштаба domain shift
```

Нельзя делать вывод о непригодности transfer learning только по zero-shot результату.

## 23.3. Этап C — fine-tuning pretrained MoGen на spine

Checkpoint `mouse_mixed` используется как initialization:

```math
\theta_0=\theta_{MoGen}.
```

Основной вариант — `full fine-tuning`, то есть обновляются все параметры.

Для fine-tuning использовать меньший effective learning rate, чем при обучении с нуля. Точное значение выбирается на validation и хранится в config.

Если full fine-tuning нестабилен, допускается staged вариант:

```text
1. tokenizer + velocity head;
2. разморозить latent transformer;
3. fine-tune entire model.
```

Но staged fine-tuning — резервный вариант, а не обязательная первая реализация.

---

# 24. MoGen initialized randomly → training on spines

После получения работоспособного pretrained+fine-tuned варианта обучить ту же архитектуру с нуля:

```math
\theta_0\sim RandomInitialization.
```

Обязательные условия:

```text
одинаковый train/val/test split
одинаковый point count
одинаковый coordinate scale
одинаковый preprocessing
одинаковый evaluation pipeline
одинаковая архитектура
одинаковый point-cloud-to-mesh backend
```

Сравнить:

```text
pretrained → fine-tuned
random initialization → training
```

по:

1. финальному качеству;
2. скорости сходимости;
3. train steps до заданного validation-качества.

---

# 25. Point cloud → mesh для MoGen

## 25.1. Основной метод

Фиксируется:

```math
\boxed{Screened\;Poisson\;Surface\;Reconstruction}
```

SAP в основной pipeline не используется.

## 25.2. Почему Poisson и SAP — не одно и то же

`Poisson Surface Reconstruction` — алгоритм восстановления поверхности из ориентированного point cloud.

`Screened Poisson` — вариант Poisson Reconstruction, дополнительно учитывающий исходные точки как интерполяционные ограничения.

`Shape As Points (SAP)` — отдельная обучаемая/дифференцируемая система, внутри которой используется дифференцируемый Poisson solver.

Следовательно:

```text
SAP != другое название Poisson
```

но:

```text
SAP использует Poisson как часть архитектуры.
```

## 25.3. Почему основной вариант — Screened Poisson

Для исследования важно сравнить **генеративные модели**, а не одновременно генератор и обучаемый mesh-реконструктор.

Screened Poisson:

- детерминирован;
- не требует отдельного обучения;
- создаёт замкнутую поверхность;
- одинаково применяется ко всем Flow Matching outputs;
- не добавляет обучаемый блок, способный скрыть или усилить ошибки генератора.

SAP тестировать только если Screened Poisson систематически плохо восстанавливает реальные spine point clouds.

---

# 26. Оценка нормалей для generated point cloud

Poisson Reconstruction требует ориентированные точки `(p_i,n_i)`, а MoGen генерирует только `p_i`.

## 26.1. Локальная PCA

Для каждой generated point:

1. найти `k_normal` соседей;
2. вычислить covariance matrix;
3. найти собственный вектор минимального собственного значения;
4. использовать его как локальную нормаль.

Параметр `k_normal` подбирается на validation.

Стартовый диапазон:

```text
20–50
```

## 26.2. Ориентация нормалей

PCA определяет нормаль с точностью до знака. Необходимо выполнить глобальное согласование ориентации по графу соседства, затем global flip, если большинство нормалей направлено внутрь.

Допускается tangent-plane propagation из Open3D или эквивалентный алгоритм.

---

# 27. Screened Poisson Reconstruction

Вход:

```text
points  [8192, 3]
normals [8192, 3]
```

Выход:

```text
vertices
faces
```

Параметры:

```text
depth
point_weight
samples_per_node
scale
```

хранятся в конфигурации и не подбираются отдельно для каждой генеративной модели.

---

# 28. Калибровка point cloud → mesh до сравнения моделей

До применения Poisson к generated point clouds отдельно измерить ошибку реконструктора:

```text
real_mesh
→ surface_sampling
→ real_point_cloud
→ normal estimation / true normals
→ Screened Poisson
→ reconstructed_mesh
```

Сравнить reconstructed mesh с real mesh.

По validation подобрать единый набор Poisson-параметров и заморозить их до test.

Это позволяет отличить:

```text
ошибка генератора
```

от:

```text
ошибка point-cloud-to-mesh reconstruction.
```

---

# 29. Когда нужно тестировать SAP

SAP добавляется только если на **реальных**, а не generated point clouds Screened Poisson систематически:

- сглаживает тонкую шейку;
- закрывает важные локальные вогнутости;
- создаёт неверную топологию;
- плохо работает при реальной плотности sampling.

Сначала проводится отдельный validation-эксперимент:

```text
Screened Poisson
vs
SAP
```

на реальных point clouds.

Только если SAP статистически значимо улучшает реконструкцию реальных mesh, можно повторить финальное сравнение генераторов с SAP.

---

# 30. Рекомендуемый порядок реализации

## Этап 0. Единый dataset + evaluation framework

До обучения моделей реализовать:

1. `sealed_local_mesh`;
2. mesh QC;
3. area-weighted point sampling;
4. point clouds `2048/4096/8192`;
5. normals;
6. SDF samples;
7. train/val/test split;
8. Screened Poisson reconstruction;
9. Marching Cubes extraction;
10. единый evaluation pipeline.

## Этап 1. Reproduce pretrained MoGen

Выполнить официальный demo/checkpoint без изменения модели.

## Этап 2. Fine-tune pretrained MoGen на spine

Это рекомендуемый **первый полноценный генеративный эксперимент**.

Преимущества порядка:

1. уже есть обученные веса;
2. уже есть рабочая архитектура;
3. быстрее получить первые spine samples;
4. можно проверить transfer learning;
5. сразу появляется полный pipeline:

```text
spine point clouds
→ Flow Matching
→ generated point cloud
→ Screened Poisson
→ generated mesh
```

## Этап 3. Реализовать PointNeXt → VAE → SIREN-SDF

После первого Flow Matching baseline реализовать независимую архитектуру.

## Этап 4. MoGen from scratch

Полноценное обучение MoGen с нуля запускать **после** pretrained fine-tuning и VAE, поскольку это самый дорогой эксперимент.

Сначала выполнить короткий pilot:

```text
random initialization
→ limited steps
→ validation
```

и только затем продлевать обучение.

## 30.1. Итоговый порядок

```text
Data + Evaluation
        ↓
MoGen official checkpoint smoke test
        ↓
MoGen pretrained → fine-tune on spines
        ↓
PointNeXt → VAE → SIREN-SDF
        ↓
MoGen random init → training on spines
        ↓
final comparison
```

Это предпочтительнее, чем тратить большой вычислительный бюджет на полный MoGen-from-scratch до появления независимого VAE baseline.

---

# 31. Разбиение Train / Validation / Test

## 31.1. Основное правило

Запрещено делать случайный split по отдельным spine.

Несколько spine одной ветви являются зависимыми наблюдениями. Если spine одной ветви попадут одновременно в train и test, модель частично увидит один и тот же биологический объект в обучении и тестировании.

## 31.2. Уровень группировки

Split выполнять по максимально высокому доступному идентификатору:

```text
animal_id
    ↓ если нет
sample_id
    ↓ если нет
neuron_id
    ↓ если нет
branch_id
```

Минимально допустимый уровень:

```text
branch_id
```

Все spine одного group находятся только в одном split.

## 31.3. Пропорции

Основная конфигурация:

```text
Train      80%
Validation 10%
Test       10%
```

Доли считать по группам, а не непосредственно по spine. После группового разбиения проверить фактическое число spine в каждом split.

## 31.4. Стратификация

Хотя модели пока unconditional, по возможности сохранить сопоставимое распределение:

```text
dataset_id
species
health
cell_type_binary
compartment
```

Стратификация не должна нарушать group split.

Для редких групп приоритет:

```text
no leakage > exact stratification
```

## 31.5. Фиксированный manifest

Сохранить:

```text
splits_v1.parquet
```

или:

```text
splits_v1.csv
```

со столбцами:

```text
spine_id
group_id
split
```

Этот split обязателен для всех моделей.

## 31.6. Производные одного spine

Все:

```text
pointcloud_2048
pointcloud_4096
pointcloud_8192
sdf_samples
mesh
```

наследуют split исходного `spine_id`.

---

# 32. Validation: общая логика

Необходимо разделять четыре типа качества:

```text
1. качество reconstruction backend;
2. качество реконструкции VAE;
3. техническая корректность generated mesh;
4. правдоподобие распределения generated spines.
```

Главная цель сравнения генеративных моделей — пункты `3` и `4`.

---

# 33. Этап A: проверка mesh reconstruction backend

До обучения моделей измерить:

```text
real mesh
→ point cloud
→ Screened Poisson
→ reconstructed mesh
```

Метрики:

```text
Chamfer Distance
Hausdorff-95
Normal Consistency
Area relative error
Volume relative error
```

Для VAE отдельно проверить:

```text
real mesh
→ exact SDF
→ Marching Cubes
→ reconstructed mesh
```

Это позволяет оценить нижнюю границу ошибки, создаваемой самим представлением и процедурой extraction.

---

# 34. Этап B: VAE reconstruction quality

Для test spine:

```text
real point cloud
→ encoder
→ z = mu
→ SDF decoder
→ Marching Cubes
→ reconstructed mesh
```

Сравнить с исходным mesh.

Метрики:

```text
Chamfer Distance ↓
Hausdorff-95 ↓
Normal Consistency ↑
F-score ↑
Area Error ↓
Volume Error ↓
```

Это диагностическая метрика VAE и она **не используется напрямую для сравнения VAE с Flow Matching**, потому что Flow Matching не является autoencoder.

---

# 35. Главная оценка: unconditional generation

Для каждой модели сгенерировать независимый набор объектов.

Минимальный protocol:

```text
1000 samples × 5 random seeds
```

или больше, если вычислительно допустимо.

Для каждой generated sample обязательно получить **финальный mesh**.

VAE:

```text
z
→ SDF
→ Marching Cubes
→ mesh
```

MoGen:

```text
noise
→ point cloud
→ normals
→ Screened Poisson
→ mesh
```

Основной leaderboard строится по final mesh.

---

# 36. Техническая валидность generated mesh

Для каждой generated sample считать:

```text
watertight
manifold
n_connected_components
self_intersections
degenerate_faces
genus
positive_volume
surface_touches_bbox
```

Итоговые показатели:

```text
valid_mesh_rate
watertight_rate
single_component_rate
genus0_rate
self_intersection_rate
```

Модель, которая получает хорошие метрики только после агрессивного исправления mesh, не должна считаться лучшей.

---

# 37. Морфологическая валидация

Для каждого generated mesh рассчитать тем же кодом, что и для реальных данных:

```text
OldChordDistribution
OpenAngle
CVD
AverageDistance
LengthVolumeRatio
LengthAreaRatio
JunctionArea
Length
Area
Volume
ConvexHullVolume
ConvexHullRatio
```

## 37.1. JunctionArea

`JunctionArea` требует отдельного внимания.

У реального spine искусственная область запаивания известна из `cap_face_indices`. У generated mesh такого mask автоматически нет.

До включения `JunctionArea` в основную таблицу необходимо реализовать детерминированный алгоритм поиска области основания generated spine на основе канонической локальной системы координат и плоскости присоединения.

До этого `JunctionArea`:

```text
не использовать как primary comparison metric
```

либо считать только там, где generated cap однозначно идентифицируется.

---

# 38. Сравнение распределений отдельных морфологических признаков

Для каждого скалярного признака `m_j` сравнить:

```math
p_{real}(m_j)
```

и:

```math
p_{gen}(m_j).
```

Использовать:

```text
Wasserstein-1 distance ↓
Kolmogorov–Smirnov statistic ↓
Energy distance ↓
```

Визуализации:

```text
histogram / KDE
ECDF
boxplot / violin plot
```

---

# 39. Совместное распределение морфологических признаков

Сформировать вектор:

```math
m=
[
OpenAngle,
CVD,
AverageDistance,
LengthVolumeRatio,
LengthAreaRatio,
Length,
Area,
Volume,
ConvexHullVolume,
ConvexHullRatio,
...
].
```

Стандартизировать статистикой **train**:

```math
\tilde m_j=
\frac{m_j-\mu_{train,j}}
{\sigma_{train,j}}.
```

Основная multivariate metric:

```text
MMD in morphometric feature space ↓
```

Дополнительно:

```text
Energy distance ↓
```

---

# 40. Корреляционная структура

Рассчитать:

```math
R_{real}=Corr(M_{real}),
```

```math
R_{gen}=Corr(M_{gen}).
```

Сравнить:

```math
\|R_{real}-R_{gen}\|_F.
```

Это выявляет ситуацию, когда marginal distributions выглядят правильно, но связи между `Length`, `Area`, `Volume`, `ConvexHullRatio` и другими признаками нарушены.

---

# 41. Point-cloud / surface generation metrics

Из каждого real/generated mesh семплировать одинаковое число точек, например `2048`, и считать:

```text
MMD-CD ↓
Coverage-CD ↑
1-NNA-CD → 0.5
```

При вычислительной возможности дополнительно:

```text
MMD-EMD
Coverage-EMD
```

EMD не должен блокировать основной эксперимент из-за высокой стоимости.

---

# 42. Метрики, адаптированные из MoGen

Дополнительно можно использовать:

```text
distance of centroid to origin
std along PCA axis 1
std along PCA axis 2
std along PCA axis 3
nearest-neighbor distance mean
nearest-neighbor distance std
farthest-neighbor distance mean
farthest-neighbor distance std
MST total length
MST longest edge
```

Для изолированных spine эти признаки считать **auxiliary**, а не primary metrics.

---

# 43. Diversity

Измерить:

```text
pairwise Chamfer distribution
pairwise distance in morphometric space
variance of each morphometric feature
```

Сравнить:

```text
real-real
generated-generated
```

Если generated diversity значительно ниже real diversity, возможен mode collapse.

---

# 44. Проверка на memorization

Для каждого generated object найти:

```text
nearest train sample
nearest test sample
```

по:

1. Chamfer Distance;
2. morphometric feature distance.

Сравнить распределения:

```text
d(gen, train)
d(gen, test)
```

Для визуального контроля сохранить пары:

```text
generated spine
nearest real train spine
```

для 50–100 случайных samples.

---

# 45. Real-vs-generated classifier

Дополнительная диагностика — классификатор:

```text
real vs generated
```

на морфологических признаках.

Интерпретация:

```text
accuracy ≈ 0.5
```

— real и generated трудно разделить по выбранному пространству признаков;

```text
accuracy >> 0.5
```

— существует систематический сдвиг.

Это auxiliary metric, а не единственный критерий качества.

---

# 46. Визуальная экспертная проверка

Для каждой модели автоматически сформировать:

```text
100 random generated meshes
100 random real meshes
```

в одинаковой ориентации, масштабе и способе рендеринга.

Желательно blind review, где `real/generated` скрыт.

Оцениваемые признаки:

```text
общая правдоподобность
наличие головки
наличие шейки
аномальная фрагментация
аномальные отверстия
нереалистичные острые выступы
артефакты области основания
```

---

# 47. Статистическая устойчивость сравнения

Каждую обучаемую модель желательно запускать минимум:

```text
3 random seeds
```

Для генерации:

```text
5 generation seeds
```

Результаты представлять как `mean ± std`.

Для распределительных метрик использовать bootstrap confidence intervals. Реальные данные по возможности bootstrap-ить на уровне `neuron/branch`, а не считать все spine полностью независимыми.

---

# 48. Primary criteria для сравнения моделей

Основные показатели:

```text
1. valid_mesh_rate ↑
2. MMD_morphometrics ↓
3. Wasserstein distances по ключевым morphometrics ↓
4. Coverage-CD ↑
5. 1-NNA-CD → 0.5
6. diversity similarity to real ↑
7. memorization rate ↓
```

Дополнительные:

```text
training stability
inference time
mesh reconstruction time
GPU memory
number of parameters
```

---

# 49. Правила честного сравнения VAE и MoGen

Обязательно одинаковые:

```text
train/val/test split
source meshes
physical units
local coordinate system
quality filtering
morphometric implementation
evaluation sampling
number of generated samples
evaluation seeds
```

Разрешаются разные внутренние представления:

```text
VAE → SDF
MoGen → point cloud
```

Но финальное сравнение выполняется после преобразования обоих outputs в:

```text
mesh
```

и расчёта одних и тех же предметных метрик.

---

# 50. Отдельная оценка ошибки постобработки

Необходимо хранить `generator_output` и `final_mesh` раздельно.

Для MoGen:

```text
generated_pointcloud.ply
generated_mesh_raw.off
generated_mesh_postprocessed.off
```

Для VAE:

```text
latent.npy
sdf_evaluation_config
generated_mesh_raw.off
generated_mesh_postprocessed.off
```

Это позволяет определить, на каком этапе появился дефект.

---

# 51. Структура проекта

Рекомендуемая структура:

```text
src/
├── preprocessing/
│   ├── mesh_qc.py
│   ├── coordinate_transform.py
│   ├── surface_sampling.py
│   ├── normals.py
│   ├── sdf_sampling.py
│   ├── build_manifest.py
│   └── build_splits.py
│
├── models/
│   ├── spine_vae/
│   │   ├── pointnext_encoder.py
│   │   ├── latent.py
│   │   ├── siren_sdf_decoder.py
│   │   ├── losses.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── generate.py
│   │
│   └── spine_mogen/
│       ├── dataset.py
│       ├── model_adapter.py
│       ├── train_finetune.py
│       ├── train_scratch.py
│       ├── infer.py
│       └── checkpoint.py
│
├── reconstruction/
│   ├── marching_cubes.py
│   ├── estimate_normals.py
│   ├── screened_poisson.py
│   └── mesh_validation.py
│
└── evaluation/
    ├── geometry_metrics.py
    ├── morphometrics.py
    ├── distribution_metrics.py
    ├── diversity.py
    ├── memorization.py
    ├── bootstrap.py
    └── report.py
```

---

# 52. Конфигурационные файлы

Все важные параметры должны находиться вне кода.

```text
configs/
├── data/
│   └── spines_v1.yaml
├── vae/
│   ├── baseline.yaml
│   └── final.yaml
├── mogen/
│   ├── pretrained_finetune.yaml
│   ├── scratch_pilot.yaml
│   └── scratch_final.yaml
├── reconstruction/
│   └── poisson.yaml
└── evaluation/
    └── default.yaml
```

---

# 53. Логирование экспериментов

Каждый запуск должен сохранять:

```text
experiment_id
git_commit
dataset_version
split_version
config
random_seed
checkpoint
training_curves
validation_metrics
generation_metrics
```

Структура:

```text
runs/
└── experiment_id/
    ├── config.yaml
    ├── checkpoints/
    ├── logs/
    ├── samples/
    ├── meshes/
    └── metrics/
```

---

# 54. Контрольные точки реализации

## Milestone 1 — Dataset ready

Готово:

```text
sealed_local_mesh
QC
point clouds
normals
SDF samples
fixed split
```

## Milestone 2 — Mesh reconstruction backend ready

Готово:

```text
real point cloud
→ Screened Poisson
→ acceptable real mesh reconstruction
```

и:

```text
real SDF
→ Marching Cubes
→ acceptable real mesh reconstruction
```

## Milestone 3 — Pretrained MoGen adapted

Готово:

```text
official inference works
fine-tuning works
spine point clouds generated
meshes reconstructed
metrics calculated
```

## Milestone 4 — VAE ready

Готово:

```text
reconstruction works
prior generation works
Marching Cubes meshes valid
metrics calculated
```

## Milestone 5 — MoGen from scratch ready

Готово:

```text
training converges
point clouds generated
meshes reconstructed
metrics calculated
```

## Milestone 6 — Final unconditional comparison

Сравниваются:

```text
A. MoGen pretrained → fine-tuned
B. MoGen random initialization → training
C. PointNeXt → VAE → SIREN-SDF
```

по единому evaluation protocol.

---

# 55. Критерий завершения текущего этапа

Этап считается завершённым, если:

1. обе архитектуры способны генерировать новые spine;
2. результат обеих архитектур автоматически преобразуется в полигональный mesh;
3. используется один frozen train/val/test split;
4. реализован общий validation pipeline;
5. рассчитаны предметные морфологические распределения;
6. измерена техническая валидность mesh;
7. проверены diversity и memorization;
8. можно количественно сравнить:
   - VAE;
   - pretrained+fine-tuned MoGen;
   - MoGen from scratch;
9. на основании сравнения можно выбрать основной генератор для следующего этапа — условной генерации по биологическим характеристикам.

---

# 56. Зафиксированные технические решения

## 56.1. VAE

Не выбирать между SDF и SIREN.

Использовать:

```math
\boxed{SDF\ representation + SIREN\ decoder}
```

Mesh:

```math
SDF
\rightarrow
Marching\;Cubes
\rightarrow
mesh.
```

## 56.2. Flow Matching

Использовать:

```math
\boxed{
MoGen/PointInfinity
+
kNN
+
Flow\;Matching
}
```

Основной point-cloud-to-mesh backend:

```math
\boxed{Screened\;Poisson\;Surface\;Reconstruction}
```

SAP:

```text
не основной метод;
использовать только как отдельный reconstruction experiment,
если Screened Poisson недостаточно хорошо восстанавливает реальные spine.
```

## 56.3. Условная генерация

На текущем этапе:

```text
disabled
```

но архитектура должна позволять позже добавить:

```text
species
health
cell_type
cell_type_binary
compartment
```

без полной переработки модели.

---

# 57. Литература и техническая база

1. Rieger F., Lueckmann J.-M., Jain V., Januszewski M. *MoGen: Detailed Neuronal Morphology Generation via Point Cloud Flow Matching*. ICLR, 2026.
2. Huang Z., Johnson J., Debnath S., Rehg J. M., Wu C.-Y. *PointInfinity: Resolution-Invariant Point Diffusion Models*. CVPR, 2024.
3. Qian G., Li Y., Peng H., et al. *PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies*. NeurIPS, 2022.
4. Park J. J., Florence P., Straub J., Newcombe R., Lovegrove S. *DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation*. CVPR, 2019.
5. Sitzmann V., Martel J. N. P., Bergman A. W., Lindell D. B., Wetzstein G. *Implicit Neural Representations with Periodic Activation Functions*. NeurIPS, 2020.
6. Lorensen W. E., Cline H. E. *Marching Cubes: A High Resolution 3D Surface Construction Algorithm*. SIGGRAPH, 1987.
7. Kazhdan M., Hoppe H. *Screened Poisson Surface Reconstruction*. ACM Transactions on Graphics, 2013.
8. Peng S., Jiang C., Liao Y., Niemeyer M., Pollefeys M., Geiger A. *Shape As Points: A Differentiable Poisson Solver*. NeurIPS, 2021.
