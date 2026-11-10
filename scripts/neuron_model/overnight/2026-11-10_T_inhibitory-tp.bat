@echo off
rem 2026-11-10, chain T (CPU only, ~2-4 h; can run IN PARALLEL with chain S): inhibitory neurons into T0 / P0 / P1.
rem Decision of 2026-11-10 (docs/neuron-model/compartment-by-direction.md, section 9): the axon of a neuron WITHOUT an apical
rem   dendrite (BC, BPC, MC, NGC, Unsure I) is its thin and long limb (median branch radius < 180 nm, cable > 1.5 mm; checked
rem   against synapses: 97-99 %% of the presynaptic points of the 7 well-annotated inhibitory neurons); the WHOLE limb is
rem   axon_like (simple variant, ~14 %% of its cable are dendritic branches - accepted loss), the other limbs get the new
rem   compartment inhibitory_dendrite; interneurons without such a limb (6 of 72) stay not_applicable (= excluded).
rem   Excitatory neurons: unchanged (axis -y, axons = table OR branch rule, as 2026-11-03_101704).
rem T1 compartment table: --reuse of the features run 2026-11-02_195824 (no NAS reading, seconds) -> a NEW folder.
rem    Check printed: the excitatory labels must equal those of 2026-11-03_101704 (crosstab).
rem T2 branches: reapply_eligibility --refresh-metadata with the new table. REWRITES branches.parquet in place (the old
rem    labels come back by the same command with the 2026-11-03_101704 table); copy of the dataset table first.
rem T3 T0, T4 P0 (same configs, splits_v1, exclude [axon_like, None, nan] -> inhibitory dendrites now INCLUDED),
rem T5 P1 on the new P0, T6 generated set (T0 + P1b) on val + assembled-branch evaluation, also for the inhibitory requests.
rem T7 view_neuron_compartments.py: HTML with whole neurons (disputed excitatory, inhibitory) and branch close-ups (reads the NAS).
rem Log: scripts\neuron_model\overnight\2026-11-10_T_inhibitory-tp.bat > logs_T.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set B=C:\Users\Student\LipsEkaterina\data\minnie65\preprocessed_branches
set FEAT=runs\analysis\neuron-model\2026-11-02_195824_apical-by-direction-minnie65
set OLD=runs\analysis\neuron-model\2026-11-03_101704_apical-by-direction-minnie65
set NEW=runs\analysis\neuron-model\2026-11-10_apical-by-direction-minnie65-inhibitory
set SPLIT=data\processed\neuron-model\splits\splits_v1.parquet
set RUNS=runs\training\neuron-model
set TP=runs\analysis\neuron-model\tp_generation\inhibitory

cd /d %REPO%
if not exist %FEAT%\limb_features.parquet (echo MISSING %FEAT%\limb_features.parquet & goto :end)

echo ===== [T1] compartment table with inhibitory axons / dendrites ===== %date% %time%
python scripts\neuron_model\apical_by_direction.py --reuse %FEAT% --dataset minnie65 --pia-axis 0,-1,0 --branches-root %B% --out %NEW%
if errorlevel 1 goto :end
python -c "import pandas as pd; a=pd.read_parquet(r'%OLD%\limb_compartments_direction.parquet'); b=pd.read_parquet(r'%NEW%\limb_compartments_direction.parquet'); m=a.merge(b,on='limb',suffixes=('_old','_new')); print(pd.crosstab(m.compartment_old,m.compartment_new)); print('new labels', b.compartment.value_counts().to_dict())"

echo ===== [T2] refresh the compartments of the branches ===== %date% %time%
copy /y %B%\branches.parquet %B%\branches_before_2026-11-10_inhibitory.parquet
python scripts\neuron_model\reapply_eligibility.py --output-root %B% --refresh-metadata --compartment-table %NEW%\limb_compartments_direction.parquet
if errorlevel 1 goto :end
python -c "import pandas as pd; t=pd.read_parquet(r'%B%\branches.parquet', columns=['cell_type_binary','compartment','train_eligible_T','train_eligible_P']); print(t.groupby(['cell_type_binary','compartment'],dropna=False)[['train_eligible_T','train_eligible_P']].sum())"

echo ===== [T3] T0 ===== %date% %time%
python scripts\neuron_model\fit_t0.py --config configs\neuron-model\trunk\t0.yaml data.branches_root=%B% data.splits=%SPLIT%
if errorlevel 1 goto :end
for /f "delims=" %%d in ('dir /b /ad /o-n %RUNS%\*_t0') do (set T0=%RUNS%\%%d& goto :t0found)
:t0found
echo T0 run: %T0%

echo ===== [T4] P0 ===== %date% %time%
python scripts\neuron_model\fit_p0.py --config configs\neuron-model\placement\p0.yaml data.branches_root=%B% data.splits=%SPLIT%
if errorlevel 1 goto :end
for /f "delims=" %%d in ('dir /b /ad /o-n %RUNS%\*_p0') do (set P0=%RUNS%\%%d& goto :p0found)
:p0found
echo P0 run: %P0%

echo ===== [T5] P1 on the new P0 (P1a, P1b, P1b2 with two radii; chosen by the Monte Carlo check) ===== %date% %time%
python scripts\neuron_model\fit_p1.py --p0-run %P0% --n-eval 400 --n-sim 19
if errorlevel 1 goto :end
for /f "delims=" %%d in ('dir /b /ad /o-n %RUNS%\*_p1') do (set P1=%RUNS%\%%d& goto :p1found)
:p1found
echo P1 run: %P1%

echo ===== [T6] generated branches (T0 + chosen P1) on val, evaluation ===== %date% %time%
python scripts\neuron_model\generate_tp.py --t0-run %T0% --p0-run %P1% --mode generated --n 1000 --seed 1 --split val --out-root %TP%\generated
for /f "delims=" %%d in ('dir /b /ad /o-n %TP%\generated') do (set GEN=%TP%\generated\%%d& goto :genfound)
:genfound
python scripts\neuron_model\generate_tp.py --t0-run %T0% --p0-run %P1% --mode real_t --n 1000 --seed 1 --split val --out-root %TP%\real_t
for /f "delims=" %%d in ('dir /b /ad /o-n %TP%\real_t') do (set REALT=%TP%\real_t\%%d& goto :realtfound)
:realtfound
for %%s in (%GEN% %REALT%) do (
  echo ----- %%s: all / excitatory / inhibitory -----
  python scripts\neuron_model\evaluate_assembled_branches.py --tp-dir %%s
  python scripts\neuron_model\evaluate_assembled_branches.py --tp-dir %%s --cell-type-binary excitatory
  python scripts\neuron_model\evaluate_assembled_branches.py --tp-dir %%s --cell-type-binary inhibitory
)

echo ===== [T7] page with whole neurons and their compartments (for the report) ===== %date% %time%
python scripts\neuron_model\view_neuron_compartments.py --limbs %NEW%\limbs.csv --compartment-table %NEW%\limb_compartments_direction.parquet ^
  --features %FEAT%\limb_features.parquet --points runs\analysis\neuron-model\2026-11-10_030133_axon-rule-check\points.parquet ^
  --branches %B%\branches.parquet --dataset-root O:\Datasets\Minnie65

:end
echo ===== T done ===== %date% %time%
