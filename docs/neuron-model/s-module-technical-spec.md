## Формализация моделей генерации дендритных шипиков

### Цель модуля генерации

Цель модуля – реализовать и сравнить несколько независимых подходов к безусловной генерации геометрии дендритных шипиков. На текущем этапе модель не получает на вход `species`, `health`, `cell_type`, `cell_type_binary` и `compartment` как условие. Эти поля сохраняются для стратификации, анализа и последующего этапа условной генерации, но первая версия генераторов должна отвечать на базовый вопрос: способны ли выбранные архитектуры воспроизводить распределение морфологически правдоподобных изолированных шипиков.

Финальным результатом каждой модели должена быть полигональная сетка отдельного шипика в локальной системе координат. Это требование необходимо для честного сравнения моделей через единые морфологические, геометрические и визуальные критерии.

Для реализации выбраны две основные архитектуры:

```math
\boxed{
PointNeXt
\rightarrow
VAE
\rightarrow
SIREN\text{-}SDF
\rightarrow
Marching\ Cubes
\rightarrow
mesh
}
```

и

```math
\boxed{
MoGen/PointInfinity
+
kNN
\rightarrow
Flow\ Matching
\rightarrow
point\ cloud
\rightarrow
Screened\ Poisson
\rightarrow
mesh
}.
```

Внутри второй архитектуры сравниваются две экспериментальные конфигурации: дообучение предобученной MoGen-модели на шипиках и обучение той же архитектуры с нуля. Таким образом, финальное сравнение включает три модели: `PointNeXt → VAE → SIREN-SDF`, `MoGen pretrained → fine-tuned` и `MoGen random initialization → training on spines`.

### Обоснование выбора моделей

Выбранные модели решают задачу генерации формы разными способами и поэтому дают содержательно полезное сравнение. VAE с неявным SDF-декодером строит компактное латентное пространство шипиков и восстанавливает поверхность как нулевой уровень непрерывной функции. Flow Matching на point cloud, напротив, моделирует распределение облаков точек напрямую как динамику переноса шума в геометрию объекта. Эти подходы отличаются типом представления, характером ошибок, вычислительной стоимостью и способом получения mesh, что делает их хорошими кандидатами для сравнительного исследования.

Модель `PointNeXt → VAE → SIREN-SDF` выбрана как независимая альтернатива архитектуре с FoldingNet-декодером. Ее преимущества для данной задачи состоят в следующем:
1) SDF-представление задает поверхность непрерывно, а не только конечным набором точек, поэтому mesh можно извлекать с разным разрешением без изменения обученной модели. 
2) SIREN-декодер хорошо подходит для аппроксимации функций с высокочастотными деталями, что важно для тонкой шейки, головки и локальной кривизны шипика. 
3) VAE формирует явное вероятностное латентное пространство, удобное для интерполяций, анализа распределения форм и будущего добавления условных признаков.

PointNeXt выбран как энкодер, потому что он является современной развитием идей PointNet++: модель строит локальные иерархические признаки point cloud, расширяет receptive field и при этом сохраняет инвариантность к перестановке точек. Для шипиков это особенно важно, поскольку геометрия задается поверхностной выборкой, где порядок точек не несет смысла, а локальные структуры (тонкая шейка, расширение головки, изгибы поверхности) имеют прямое морфологическое значение.

Модель `MoGen/PointInfinity + kNN → Flow Matching` выбрана как современный генеративный подход для point cloud. Ее главное преимущество состоит в том, что Flow Matching обучает непрерывное векторное поле, переводящее простой шум в распределение реальных объектов. По сравнению с VAE такая модель не обязана сжимать объект в один фиксированный латентный вектор энкодера и потенциально лучше сохраняет разнообразие распределения. Кроме того, MoGen уже разрабатывалась для нейрональной морфологии, а значит дает возможность проверить transfer learning с близкого домена: локальных фрагментов нейритов.

Использование pretrained MoGen и отдельное обучение MoGen с нуля имеют разные исследовательские цели. Дообучение предобученной модели проверяет, можно ли перенести знания о нейрональной геометрии на задачу изолированных шипиков и быстрее получить правдоподобные образцы. Обучение с нуля проверяет, насколько качество объясняется самой архитектурой, а не предобучением. Сравнение этих двух режимов покажет, полезен ли transfer learning и насколько велик domain shift между исходными neurite fragments и локальными mesh шипиков.

### Общие требования к моделям

Все модели обучаются и оцениваются на одном и том же фиксированном разбиении `train/validation/test`. Нельзя подбирать split отдельно под архитектуру. Все модели используют одинаковую локальную систему координат, одинаковые физические единицы, одинаковые критерии фильтрации по качеству и одинаковый набор evaluation-метрик. Разные внутренние представления допускаются: VAE использует SDF, Flow Matching использует point cloud. Однако итоговое сравнение выполняется только после преобразования результата каждой модели в mesh.

На текущем этапе генерация является безусловной:

```math
x \sim p_{\theta}(x),
```

где $x$ – геометрия одного шипика в локальных координатах. При этом интерфейс моделей должен проектироваться так, чтобы позднее можно было добавить условный вектор $c$:

```math
x \sim p_{\theta}(x \mid c),
```

где $c$ может включать `species`, `health`, `cell_type`, `cell_type_binary`, `compartment` и другие биологические признаки. В первой версии условный блок отключен или получает нулевой embedding, но места его подключения должны быть предусмотрены в архитектуре.

Для предотвращения некорректных сравнений все модели должны сохранять раздельно:

```text
raw generator output
raw extracted mesh
postprocessed mesh
generation config
random seed
checkpoint id
```

Это позволяет отличить ошибку генератора от ошибки извлечения поверхности или постобработки.

---

### Модель 1: PointNeXt → VAE → SIREN-SDF

#### Общая постановка

Пусть один реальный шипик представлен облаком точек с нормалями:

```math
X=\{(p_i,n_i)\}_{i=1}^{N},
```

где $p_i\in\mathbb R^3$ – координаты точки поверхности, $n_i\in\mathbb R^3$ – нормаль поверхности, $N$ – число точек. Основной размер point cloud для этой модели выбирается из диапазона 2048–8192 по результатам validation; стартовая конфигурация использует $N=8192$.

Энкодер строит глобальное представление формы:

```math
h=E_{\phi}(X).
```

Далее из $h$ выводятся параметры диагонального гауссовского апостериорного распределения:

```math
\mu=\mu_{\phi}(h),
```

```math
\log\sigma^2=\ell_{\phi}(h).
```

Латентный вектор выбирается через reparameterization trick:

```math
z=\mu+\sigma\odot\epsilon,\qquad \epsilon\sim\mathcal N(0,I),
```

где

```math
\sigma=\exp\left(\frac{1}{2}\log\sigma^2\right).
```

Декодер задает signed distance function (SDF):

```math
f_{\theta}(q,z)=\hat d,
```

где $q\in\mathbb R^3$ – произвольная query point, а $\hat d$ – предсказанное signed distance до поверхности шипика. Поверхность сгенерированного объекта определяется как нулевой уровень:

```math
\mathcal S_z=\{q: f_{\theta}(q,z)=0\}.
```

#### Архитектура энкодера PointNeXt

Вход модели имеет форму:

```text
xyz      [B, N, 3]
features [B, N, 3]
```

где `features` – нормали поверхности. Нужно предусмотреть конфигурационный параметр `use_normals`, чтобы можно было проверить ablation без нормалей. В базовой конфигурации входной признак каждой точки равен:

```math
x_i=[p_i,n_i]\in\mathbb R^6.
```

PointNeXt выполняет иерархическую агрегацию признаков: выбирает подмножество центральных точек, находит локальные соседства, агрегирует признаки внутри соседств и последовательно расширяет receptive field. Локальные группы заранее сохранять не требуется: они должны строиться во время forward pass, чтобы радиусы, число соседей и глубину сети можно было менять через конфигурацию.

Стартовая конфигурация энкодера:

```yaml
encoder:
  type: pointnext
  input_points: 8192
  input_features: 3
  base_width: 64
  stage_widths: [64, 128, 256, 512]
  global_feature_dim: 512
```

После последнего уровня применяется permutation-invariant pooling. Допустимые варианты: max pooling или конкатенация max pooling и average pooling. Результатом является глобальный вектор $h\in\mathbb R^{D_h}$.

#### Латентное пространство VAE

Начальная размерность латентного пространства:

```text
latent_dim = 128
```

На validation необходимо проверить как минимум значения 64, 128 и 256 после получения стабильного baseline. Малый latent может не вместить разнообразие форм, а слишком большой latent повышает риск слабой регуляризации и ухудшения prior sampling.

Для реконструкции объекта используется $z=\mu$, чтобы получить детерминированную оценку формы. Для генерации новых шипиков используется:

```math
z\sim\mathcal N(0,I).
```

Необходимо логировать статистики латентного пространства: средний модуль $\mu$, среднее $\sigma$, число активных латентных измерений, KL-loss и признаки posterior collapse. Если KL быстро стремится к нулю, требуется KL warm-up (временное ослабление влияния KL-дивергенции в начале обучения) или free bits (обнуление штрафа, если KL-дивергенция для конкретного признака падает ниже заданного порога).

#### SIREN-SDF декодер

SIREN-декодер получает query point и latent vector:

```math
u=[q,z].
```

В будущем вход может быть расширен до:

```math
u=[q,z,e(c)],
```

где $e(c)$ – embedding условного признака. На текущем этапе $e(c)$ отсутствует или равен нулю.

Стартовая конфигурация декодера:

```yaml
decoder:
  type: siren_sdf
  hidden_dim: 256
  hidden_layers: 5
  output_dim: 1
  first_omega_0: 30
```

Скрытые слои имеют вид:

```math
h_{l+1}=\sin\left(\omega_l(W_lh_l+b_l)\right).
```

Последний слой является линейным и возвращает одно число $\hat d$. Для SIREN обязательно использовать специальную инициализацию весов, чтобы синусоидальные активации сохраняли стабильный масштаб сигналов и градиентов. Без такой инициализации сеть может либо переусреднять поверхность, либо становиться численно нестабильной.

#### Функция потерь VAE

Базовая функция потерь состоит из SDF-loss и KL-регуляризации:

```math
\mathcal L_{\mathrm{baseline}}
=
\mathcal L_{\mathrm{sdf}}
+
\beta\mathcal L_{\mathrm{KL}}.
```

SDF-loss вычисляется по minibatch query points:

```math
\mathcal L_{\mathrm{sdf}}
=
\frac{1}{M}\sum_{i=1}^{M}
\left|\hat d_i-d_i\right|.
```

Допускается Smooth L1 как более устойчивый вариант при редких больших ошибках расстояния. 

KL-дивергенция для диагонального гауссовского апостериорного распределения (posterior):

```math
\mathcal L_{\mathrm{KL}}
=
-\frac{1}{2}
\sum_j
\left(
1+\log\sigma_j^2-\mu_j^2-\sigma_j^2
\right).
```

После получения стабильного baseline добавляются геометрические регуляризаторы. Surface-loss усиливает точность нулевого уровня:

```math
\mathcal L_{\mathrm{surface}}
=
\frac{1}{|Q_{\mathrm{surface}}|}
\sum_{q_i\in Q_{\mathrm{surface}}}
|f_{\theta}(q_i,z)|.
```

Eikonal regularization задает свойство настоящего signed distance field:

```math
\mathcal L_{\mathrm{eikonal}}
=
\mathbb E_q
\left(
\|\nabla_q f_{\theta}(q,z)\|_2-1
\right)^2.
```

Normal consistency на поверхностных точках согласует нормаль SDF с нормалью mesh:

```math
\mathcal L_{\mathrm{normal}}
=
1-
\frac{
\nabla_q f_{\theta}(q,z)\cdot n
}{
\|\nabla_q f_{\theta}(q,z)\|\,\|n\|
}.
```

Итоговая расширенная функция:

```math
\mathcal L_{\mathrm{VAE}}
=
\lambda_{\mathrm{sdf}}\mathcal L_{\mathrm{sdf}}
+
\lambda_{\mathrm{surface}}\mathcal L_{\mathrm{surface}}
+
\lambda_{\mathrm{eik}}\mathcal L_{\mathrm{eikonal}}
+
\lambda_{\mathrm{normal}}\mathcal L_{\mathrm{normal}}
+
\beta\mathcal L_{\mathrm{KL}}.
```

В первой реализации нужно начинать с baseline, затем включать регуляризаторы по одному, чтобы можно было понять вклад каждого блока.

#### Обучение VAE

Один шаг обучения устроен следующим образом. Для каждого шипика из batch загружается surface point cloud с нормалями и minibatch SDF query points. PointNeXt кодирует point cloud, VAE-секция строит $\mu$, $\log\sigma^2$ и $z$, затем SIREN-декодер предсказывает SDF в выбранных query points. Полный 3D grid во время обучения не декодируется, поскольку это слишком дорого и не требуется для оптимизации.

В training loop должны быть реализованы:

```text
mixed precision
gradient clipping
checkpointing
resume training
periodic validation
periodic reconstruction samples
periodic prior samples
```

Логировать необходимо:

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
mesh_validity_on_validation_samples
```

Валидация для VAE выполняется в двух режимах:
1) Реконструкция: реальный test/validation шипик кодируется в $z=\mu$, декодируется через SDF, затем переводится в mesh. Этот режим показывает, насколько автоэнкодер способен восстановить форму. 
2) Prior generation (генерация априорных данных): $z\sim\mathcal N(0,I)$, после чего генерируется новый mesh. Именно второй режим используется для сравнения с Flow Matching как генеративной моделью.

#### Преобразование VAE-результата в mesh

Для генерации нового объекта выбирается:

```math
z\sim\mathcal N(0,I).
```

Затем задается единый bounding box в локальных координатах. Он должен быть рассчитан по обучающей выборке с фиксированным margin и заморожен до финального тестирования. Нельзя подбирать bounding box отдельно для каждого сгенерированного образца, так как это скроет ошибки масштаба.

Внутри bounding box создается регулярная 3D-сетка. Для валидации и быстрых экспериментов используется разрешение $128^3$, для финальной генерации – $256^3$. Разрешение $512^3$ допускается только если $256^3$ систематически теряет тонкую шейку шипика.

Для каждой grid point $q_j$ вычисляется:

```math
d_j=f_{\theta}(q_j,z).
```

Оценку SDF по grid необходимо выполнять chunk-ами, чтобы не переполнять GPU-память. После получения массива $D[i,j,k]$ запускается Marching Cubes с уровнем:

```text
iso_level = 0
```

Результатом являются вершины и грани mesh. В выходной директории сохраняются как минимум:

```text
latent.npy
sdf_evaluation_config.json
generated_mesh_raw.off
generated_mesh_postprocessed.off
generation_metadata.json
```

Raw mesh используется для честной диагностики. В рамках постобработки допускается удаление очень малых компонент, удаление вырожденных граней, исправление ориентации и незначительное перестроение полигональной сетки. Крупные ошибки формы нельзя исправлять на постобработке: если нулевой уровень касается границы grid, необходимо записать флаг `surface_touches_grid_boundary=true`.

---

### Модель 2: MoGen/PointInfinity + kNN → Flow Matching

#### Общая постановка

Flow Matching модель генерирует point cloud шипика напрямую. Один шипик представлен как:

```math
x_1\in\mathbb R^{N\times3},
```

где стартовая конфигурация использует $N=8192$, поскольку это соответствует разрешению исходной MoGen-архитектуры и снижает риск деградации при переносе pretrained weights.

Начальное состояние выбирается из стандартного нормального шума:

```math
x_0\sim\mathcal N(0,I).
```

Промежуточное состояние задается линейной интерполяцией:

```math
x_t=(1-t)x_0+tx_1,\qquad t\in[0,1].
```

Для этой траектории целевая скорость равна:

```math
u_t=x_1-x_0.
```

Нейронная сеть аппроксимирует векторное поле:

```math
v_{\theta}(x_t,t)\approx u_t.
```

Функция потерь:

```math
\mathcal L_{\mathrm{FM}}
=
\mathbb E_{x_0,x_1,t}
\left[
\|v_{\theta}(x_t,t)-(x_1-x_0)\|_2^2
\right].
```

На этапе генерации решается ODE (обыкновенное дифференциальное уравнение):

```math
\frac{dx}{dt}=v_{\theta}(x,t),
```

от $t=0$ до $t=1$, начиная с $x(0)\sim\mathcal N(0,I)$. Итогом является generated point cloud.

#### Масштаб координат

При работе с предобученной MoGen нужно сохранить численный масштаб, близкий к исходной модели. В качестве стартового варианта используется глобальное преобразование:

```math
x_{\mathrm{MoGen}}=\frac{x_{\mu m}}{10}.
```

Обратное преобразование после генерации:

```math
x_{\mu m}=10x_{\mathrm{MoGen}}.
```

Это не является индивидуальной нормализацией каждого объекта: все шипики масштабируются одним глобальным коэффициентом. Для обучения MoGen с нуля на первом этапе используется то же масштабирование, чтобы сравнение pretrained и scratch не смешивалось с изменением представления координат.

#### Архитектура MoGen/PointInfinity + kNN

Архитектура должна максимально сохранять реализацию MoGen/PointInfinity. Базовые параметры:

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

Каждая точка $p_i$ текущего $x_t$ является токеном. Для нее на каждом forward pass находятся $k=16$ ближайших соседей:

```math
\mathcal N_k(i).
```

Для каждого соседа вычисляется относительное смещение:

```math
\Delta p_{ij}=p_j-p_i.
```

Координатный вход токенизатора:

```math
g_i=[p_i,\Delta p_{i1},\ldots,\Delta p_{ik}].
```

При $k=16$ размер этого входа равен $3+16\cdot3=51$. Важно, что k-NN вычисляется для текущего зашумленного состояния $x_t$, а не сохраняется заранее. Это позволяет локальному контексту следовать текущей геометрии в процессе flow.

Tokenizer преобразует $g_i$ в point token:

```math
X_i\in\mathbb R^{128}.
```

К point tokens добавляется embedding времени $e(t)$. Условный embedding $e(c)$ в первой версии отключен, но интерфейс должен позволять добавить его позднее.

Параллельно используется набор обучаемых latent токенов:

```math
Z\in\mathbb R^{256\times256}.
```

Одна стадия модели имеет структуру read – latent transformer – write. На read-step latent tokens считывают информацию из point tokens:

```math
Z\leftarrow CrossAttention(Q=Z,K=X,V=X).
```

Затем над latent токенами выполняются transformer blocks. На write-step point tokens получают глобальный контекст обратно из latent tokens:

```math
X\leftarrow CrossAttention(Q=X,K=Z,V=Z).
```

Таких стадий выполняется четыре. В конце velocity head преобразует каждый point token в трехмерную скорость:

```math
\hat v_i\in\mathbb R^3.
```

Выход модели имеет форму:

```text
[B, N, 3]
```

и интерпретируется как предсказанное направление движения каждой точки в текущий момент времени.

#### Time sampling и schedule

В основной реализации необходимо сохранить modified cosine schedule из MoGen. До воспроизведения baseline не следует заменять его на линейное или равномерное распределение времени, поскольку изменение schedule может само по себе ухудшить качество и сделать результаты несопоставимыми с исходной архитектурой.

В конфигурации schedule должен задаваться явно:

```yaml
flow:
  time_schedule: mogen_cosine
```

После получения стабильного baseline допускаются ablation-эксперименты с `linear` и `uniform`, но они не должны подменять основной результат.

#### Обучение Flow Matching

Для каждого шага обучения выбирается реальный point cloud $x_1$, генерируется шум $x_0$, выбирается время $t$, строится $x_t$, после чего модель предсказывает скорость $v_{\theta}(x_t,t)$. Оптимизируется MSE между предсказанной и целевой скоростью.

В качестве референсной настройки из MoGen фиксируются:

```yaml
optimizer: Prodigy
learning_rate: 0.5
global_gradient_clip_norm: 0.1
ema_decay: 0.999
```

Эти значения являются стартовой точкой, а не жестким требованием. В реализации обязательно поддержать:

```text
gradient_accumulation
mixed_precision
global gradient clipping
checkpointing
resume training
EMA weights
latest checkpoint
best_val_loss checkpoint
best_morphology_metric checkpoint
```

Произвольные 3D-повороты как аугментация по умолчанию не используются, поскольку локальная система координат имеет биологический смысл. Разрешены только малый Gaussian jitter в физических единицах и случайный выбор одного из заранее подготовленных surface resampling вариантов.

#### Inference Flow Matching

На inference выбирается:

```math
x(0)\sim\mathcal N(0,I).
```

Далее численно решается ODE:

```math
\frac{dx}{dt}=v_{\theta}(x,t),
```

от $t=0$ до $t=1$. Основной solver – midpoint solver, совпадающий с используемым в MoGen. Число solver steps является параметром inference config и подбирается на validation как компромисс между качеством и временем генерации.

Результат inference:

```text
generated_points [8192, 3]
```

После обратного масштабирования координаты возвращаются в локальные физические координаты. Для каждого sample сохраняются:

```text
generated_pointcloud.ply
flow_inference_config.json
generation_seed
checkpoint_id
```

#### Pretrained MoGen → fine-tuning

Эксперимент с предобученным MoGen выполняется раньше обучения с нуля. Первым checkpoint выбирается `mouse_mixed`, так как он ближе всего к рассматриваемому домену нейрональной морфологии.

Работа с предобученной моделью делится на три шага. Сначала проводится smoke test официального checkpoint без изменения архитектуры: нужно запустить официальный inference/demo, получить ожидаемые neurite point clouds и зафиксировать версию кода, checkpoint и baseline output. Затем выполняется zero-shot inference на нашем pipeline. Этот результат не является полноценной моделью шипиков: он нужен только для проверки диапазона координат, стабильности ODE solver'a, масштаба технического сдвига (domain shift) и корректности адаптера данных.

Третьим шагом выполняется fine-tuning на spine point clouds. Начальная точка:

```math
\theta_0=\theta_{\mathrm{MoGen}}.
```

Основной вариант – full fine-tuning, при котором обновляются все параметры. Learning rate должен быть меньше, чем при обучении с нуля, и подбирается на этапе валидации. Если full fine-tuning нестабилен, допускается staged fine-tuning:

```text
1. tokenizer + velocity head
2. latent transformer
3. entire model
```

Staged fine-tuning является резервным вариантом, а не первой обязательной реализацией.

#### MoGen random initialization → training on spines

После получения работоспособного pretrained+fine-tuned варианта та же архитектура обучается с нуля:

```math
\theta_0\sim RandomInitialization.
```

Обязательные условия сравнения pretrained и scratch:

```text
same train/validation/test split
same point count
same coordinate scale
same preprocessing version
same model architecture
same inference solver
same point-cloud-to-mesh backend
same evaluation pipeline
```

Сравнивать нужно не только финальное качество, но и скорость сходимости: число шагов обучения до заданного validation-качества, стабильность обучения, чувствительность к random seed и вычислительную стоимость.

#### Преобразование Flow Matching point cloud в mesh

Flow Matching генерирует только точки, поэтому для финального сравнения требуется восстановление поверхности, для этого используется:

```math
\boxed{Screened\ Poisson\ Surface\ Reconstruction}.
```

Screened Poisson выбран потому, что он детерминирован, не требует отдельного обучения, создает замкнутую поверхность и одинаково применяется ко всем Flow Matching outputs. Это важно: цель эксперимента – сравнить генераторы, а не одновременно обучать отдельный mesh-реконструктор, который может скрывать ошибки генеративной модели.

Poisson reconstruction требует ориентированного point cloud $(p_i,n_i)$, тогда как MoGen генерирует только координаты $p_i$. Поэтому перед реконструкцией необходимо оценить нормали. Для каждой точки выбираются $k_{\mathrm{normal}}$ соседей, строится ковариационная матрица локального соседства, после чего нормалью считается собственный вектор, соответствующий минимальному собственному значению. Стартовый диапазон:

```text
k_normal = 20–50
```

PCA-нормаль определена с точностью до знака, поэтому далее выполняется согласование ориентации нормалей по графу соседства и глобальный flip, если большинство нормалей направлено внутрь. Можно использовать реализацию tangent-plane propagation, например из Open3D, или эквивалентный алгоритм.

Вход Screened Poisson:

```text
points  [8192, 3]
normals [8192, 3]
```

Параметры:

```text
depth
point_weight
samples_per_node
scale
```

подбираются один раз на этапе валидации на реальных point clouds и затем замораживаются до тестирования. Для этого отдельно проводится калибровка:

```text
real mesh
→ surface sampling
→ real point cloud
→ normal estimation
→ Screened Poisson
→ reconstructed mesh
```

Реконструированный mesh сравнивается с исходным mesh. Это позволяет отделить ошибку генератора от ошибки реконструкции mesh из облака точек. Shape As Points – дополнительный метод, добавляется только если Screened Poisson систематически плохо восстанавливает реальные шипики, например сглаживает тонкую шейку или разрушает важные вогнутости.

Для каждого Flow Matching sample сохраняются:

```text
generated_pointcloud.ply
estimated_normals.ply
generated_mesh_raw.off
generated_mesh_postprocessed.off
poisson_config.json
```
---

### Валидация и подбор параметров

Валидация должна разделять четыре типа качества:

```text
1. качество реконструкции mesh из облака точек;
2. качество реконструкции для VAE;
3. техническая корректность сгенерированного mesh;
4. правдоподобие распределения сгенерированных шипиков.
```

Первый тип качества измеряется до обучения моделей. Для Flow Matching проверяется цепочка `real mesh → point cloud → Screened Poisson → reconstructed mesh`. Для VAE проверяется цепочка `real mesh/SDF → Marching Cubes → reconstructed mesh`. Эти эксперименты дают нижнюю границу ошибки, вызванной представлением и extraction-процедурой.

Для VAE на этапе валидации подбираются:

```text
point count: 2048 или 4096 
latent_dim: 64, 128, 256
beta / KL schedule
SDF loss type
weights of surface/eikonal/normal losses
SIREN hidden_dim and number of layers
Marching Cubes grid resolution
```

Для Flow Matching на этапе валидации подбираются:

```text
fine-tuning learning rate
number of ODE solver steps
EMA usage
jitter scale
k_normal for normals
Screened Poisson parameters
training length and early stopping rule
```

Этап валидации (validation) не должен использоваться для окончательного сравнения моделей. Его задача – выбрать конфигурации, которые затем один раз оцениваются на тестировании.

---

### Тестирование и единый протокол сравнения

Главное сравнение проводится в режиме безусловной генерации. Для каждой финальной модели генерируется независимый набор объектов. Минимальный протокол:

```text
1000 samples × 5 generation seeds
```

Если вычислительно возможно, число samples увеличивается. Каждая generated sample обязательно переводится в финальный mesh. Для VAE цепочка:

```text
z ~ N(0, I)
→ SDF decoder
→ Marching Cubes
→ mesh
```

Для MoGen:

```text
noise
→ Flow Matching ODE
→ point cloud
→ normal estimation
→ Screened Poisson
→ mesh
```

Техническая валидность сгенерированного mesh измеряется по следующим признакам:

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

Модель, которая получает хорошие результаты только после агрессивного исправления mesh, не должна считаться лучшей. Поэтому raw и postprocessed mesh оцениваются раздельно.

Морфологическая валидация выполняется теми же метриками, что и для реальных данных:

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

Для сгенерированного mesh `JunctionArea` требует отдельного алгоритма определения основания шипика. Признаки сравниваются как по отдельным распределениям, так и совместно.

Для каждого скалярного признака $m_j$ сравниваются распределения:

```math
p_{\mathrm{real}}(m_j)
```

и

```math
p_{\mathrm{gen}}(m_j).
```

Основные одномерные метрики:

```text
Wasserstein-1 distance
Kolmogorov-Smirnov statistic
Energy distance
```

Для совместного распределения морфологических признаков формируется вектор:

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
ConvexHullRatio
].
```

Стандартизация выполняется статистиками train set:

```math
\tilde m_j=
\frac{m_j-\mu_{\mathrm{train},j}}
{\sigma_{\mathrm{train},j}}.
```

Основная многомерная метрика:

```text
MMD in morphometric feature space
```

Дополнительно сравнивается корреляционная структура:

```math
R_{\mathrm{real}}=Corr(M_{\mathrm{real}}),
```

```math
R_{\mathrm{gen}}=Corr(M_{\mathrm{gen}}),
```

```math
\Delta_R=\|R_{\mathrm{real}}-R_{\mathrm{gen}}\|_F.
```

Это важно, потому что модель может правильно воспроизвести частные распределения длины, площади и объема, но нарушить связи между ними.

Для surface-level оценки из каждого реального/сгенерированного mesh семплируется одинаковое число точек, например 2048. Считаются:

```text
MMD-CD
Coverage-CD
1-NNA-CD
```


Отдельно оценивается diversity:

```text
pairwise Chamfer distribution
pairwise distance in morphometric space
variance of morphometric features
```

Сравниваются распределения `real-real` и `generated-generated`. Слишком узкое generated-generated распределение указывает на mode collapse.

Для проверки memorization для каждого сгенерированного объекта находится ближайший train sample и ближайший test sample по Chamfer distance и по расстоянию в морфометрическом пространстве. Если сгенерированные объекты систематически ближе к train, чем к test, это может указывать на запоминание.

Дополнительная диагностика – классификатор `real vs generated` на морфологических признаках. Accuracy около 0.5 означает, что generated и real трудно разделить в выбранном пространстве признаков, высокая accuracy указывает на систематический сдвиг.

### Основные критерии для выбора модели

Основные критерии сравнения:

```text
valid_mesh_rate ↑
MMD_morphometrics ↓
Wasserstein distances по ключевым morphometrics ↓
Coverage-CD ↑
1-NNA-CD → 0.5
diversity similarity to real ↑
memorization rate ↓
```

Дополнительные критерии:

```text
training stability
inference time
mesh reconstruction time
GPU memory
number of parameters
implementation complexity
```

Финальный выбор генератора не должен основываться на одной метрике. Предпочтительной считается модель, которая одновременно генерирует технически корректные mesh, хорошо воспроизводит морфометрическое распределение, сохраняет разнообразие и не демонстрирует memorization.

## 10. Этапность реализации

Первый этап – единая модельная инфраструктура. Необходимо реализовать loaders для manifest, point clouds и SDF samples, общий train/validation/test split, конфигурации экспериментов, директории запусков, логирование, checkpointing, evaluation API и сохранение generated artifacts. На этом этапе также реализуются Marching Cubes extraction, normal estimation, Screened Poisson reconstruction и mesh validation.

Второй этап – калибровка reconstruction backend. До обучения генераторов нужно проверить, насколько хорошо Screened Poisson восстанавливает реальные mesh из real point clouds, и насколько Marching Cubes восстанавливает поверхность из real SDF. Параметры reconstruction backend после validation замораживаются.

Третий этап – воспроизведение предобученной MoGen. Сначала запускается официальный checkpoint без изменения модели, затем zero-shot inference в нашем формате, после чего выполняется fine-tuning на spine point clouds. 

Четвертый этап – реализация `PointNeXt → VAE → SIREN-SDF`. Сначала обучается baseline с loss `SDF + KL`, затем добавляются surface, eikonal и normal regularizers. После проверки reconstruction запускается prior generation и extraction mesh через Marching Cubes.

Пятый этап – обучение MoGen с нуля. Сначала выполняется короткий pilot с random initialization и ограниченным числом шагов. Если loss, samples и validation metrics выглядят осмысленно, обучение продлевается.

Шестой этап – финальное сравнение. Оцениваются три финальные модели:

```text
A. MoGen pretrained → fine-tuned
B. MoGen random initialization → training on spines
C. PointNeXt → VAE → SIREN-SDF
```

Все три модели оцениваются на одном split, с одинаковым числом generated samples, одинаковыми generation seeds, одинаковыми морфометрическими метриками и единым отчетом.

### Структура реализации

Для дальнейшей реализации целесообразно разделить код на независимые блоки:

```text
src/models/spine_vae/
  pointnext_encoder.py
  latent.py
  siren_sdf_decoder.py
  losses.py
  model.py
  train.py
  generate.py

src/models/spine_mogen/
  dataset.py
  model_adapter.py
  train_finetune.py
  train_scratch.py
  infer.py
  checkpoint.py

src/reconstruction/
  marching_cubes.py
  estimate_normals.py
  screened_poisson.py
  mesh_validation.py

src/evaluation/
  geometry_metrics.py
  morphometrics.py
  distribution_metrics.py
  diversity.py
  memorization.py
  bootstrap.py
  report.py
```

Все параметры моделей, обучения, reconstruction и evaluation должны задаваться через конфигурационные файлы. Минимальный набор:

```text
configs/vae/baseline.yaml
configs/vae/final.yaml
configs/mogen/pretrained_finetune.yaml
configs/mogen/scratch_pilot.yaml
configs/mogen/scratch_final.yaml
configs/reconstruction/poisson.yaml
configs/evaluation/default.yaml
```

Каждый запуск сохраняет:

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
generated_samples
```

Структура `runs`:

```text
runs/<experiment_id>/
  config.yaml
  checkpoints/
  logs/
  samples/
  meshes/
  metrics/
```