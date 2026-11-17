@echo off
rem 2026-11-18, chain AB (CPU + GPU, ~10-14 h). Run after git pull, after chain AA has finished.
rem
rem AB1 MoGen v3 (conditional, trained on all of Minnie65 splits_v1) against MoGen A (dev split of 100 neurons): never
rem     compared by one protocol. Evaluation neurons = val of splits_v1 MINUS every neuron of the dev split (A saw 11 val
rem     neurons of v1 in training); train reference = train of splits_v1. Both through d8_pw8; v3 without a condition (its
rem     unconditional mode, condition dropout 0.10); floor = real spines of the same neurons through d8_pw8 (local cache).
rem     The untouched test is NOT used. Needs the NAS (real meshes of the evaluation).
rem AB2 H01 compartments: direction to the pia with the inhibitory axon rule (reuses the soma centres / limb directions of
rem     2026-11-03), written into the H01 branches (backup of branches.parquet first).
rem AB3 split of H01 (88 neurons -> ~70 / 9 / 9 by cell type) + merge with splits_v1 -> splits_v1_h01.
rem AB4 H01 spines whose neck stayed in the trunk (~21 %) listed and left out of the H01 cache (all train + 3000 val spines).
rem AB5 T0, P0, P1 on Minnie65 + H01 (species in the design), generation + evaluation per dataset on val, neuron CV.
rem AB6 MoGen v4: v3 continued on Minnie65 + H01 with the species condition (GPU, ~4-6 h).
rem Log: scripts\neuron_model\overnight\2026-11-18_AB_mogen-compare-h01.bat > logs_AB.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set B=C:\Users\Student\LipsEkaterina\data\minnie65\preprocessed_branches
set H01B=C:\Users\Student\LipsEkaterina\data\h01\preprocessed_branches
set H01MAN=C:\Users\Student\LipsEkaterina\data\h01\preprocessed_v2\H01_Seg\manifest.parquet
set SPLITS=data\processed\neuron-model\splits
set SPLIT=%SPLITS%\splits_v1.parquet
set SPLITH=%SPLITS%\splits_v1_h01.parquet
set RUNS=runs\training\neuron-model
set ANA=runs\analysis\neuron-model
set CFG=configs\neuron-model\reconstruction\poisson_sweep
set V3=%RUNS%\2026-11-03_015237_mogen-conditional-v3
set MA=%RUNS%\2026-10-20_213103_mogen-ft-dev
set FULL=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/manifest_minnie65_full.parquet
set MORPH=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/morphometrics_minnie65_full.parquet
set CMP=%ANA%\2026-11-18_v3-vs-A
set H01DIR=%ANA%\2026-11-18_apical-by-direction-h01-inhibitory
set TPH=%ANA%\tp_generation\minnie_h01

cd /d %REPO%

rem AB0 = AA3 again: in AA every job failed - the local cache minnie65_v1_60k has point clouds and SDF only, no local_sealed
rem meshes. Now the meshes come from the NAS (manifest with absolute paths). LabID Wt WITHOUT the z 0.7 correction (labid_raw_z1:
rem the 0.7 had no physical meaning, docs/neuron-model/labid-data-provenance.md). PSF narrow = NA 1.4 oil (the SpineTool protocol).
set CONF=%ANA%\2026-11-18_confocal
set META=cache\manifest_minnie65_meta.parquet
set LABIDZ1=C:\Users\Student\LipsEkaterina\data\labid\labid_raw_z1\labid\preprocessed\manifest.parquet
echo ===== [AB0] confocal imaging model: Minnie65 train spines (NAS meshes), LabID Wt (z 1) ===== %date% %time%
python scripts\neuron_model\make_real_controls.py --manifest %META% --path-mode absolute --splits %SPLIT% --split train --controls asis confocal ^
  --confocal-param sigma_xy_nm=55 --confocal-param sigma_z_nm=180 --confocal-param threshold=0.3 --n-samples 500 --workers 8 --out %CONF%\minnie_n_t0.3
for %%t in (0.08 0.15 0.5) do (
  python scripts\neuron_model\make_real_controls.py --manifest %META% --path-mode absolute --splits %SPLIT% --split train --controls confocal ^
    --confocal-param sigma_xy_nm=55 --confocal-param sigma_z_nm=180 --confocal-param threshold=%%t --n-samples 500 --workers 8 --out %CONF%\minnie_n_t%%t
)
python scripts\neuron_model\make_real_controls.py --manifest %META% --path-mode absolute --splits %SPLIT% --split train --controls confocal ^
  --confocal-param sigma_xy_nm=90 --confocal-param sigma_z_nm=270 --confocal-param threshold=0.15 --n-samples 500 --workers 8 --out %CONF%\minnie_w_t0.15
python scripts\neuron_model\make_real_controls.py --manifest %LABIDZ1% --where "neuron_id~^Wt_[0-9]" --controls asis --n-samples 250 --workers 8 --out %CONF%\labid_wt
python scripts\neuron_model\confocal_domain_shift.py --reference minnie=%CONF%\minnie_n_t0.3\asis\seed_1 --target labid_wt=%CONF%\labid_wt\asis\seed_1 ^
  --imaged n_t0.08=%CONF%\minnie_n_t0.08\confocal\seed_1 --imaged n_t0.15=%CONF%\minnie_n_t0.15\confocal\seed_1 --imaged n_t0.3=%CONF%\minnie_n_t0.3\confocal\seed_1 ^
  --imaged n_t0.5=%CONF%\minnie_n_t0.5\confocal\seed_1 --imaged w_t0.15=%CONF%\minnie_w_t0.15\confocal\seed_1 --out %CONF%\shift

echo ===== [AB1] MoGen v3 vs A on the val neurons of splits_v1 that A never saw ===== %date% %time%
python scripts\neuron_model\make_untouched_test_split.py --dev %SPLITS%\splits_dev_100n.parquet --final %SPLIT% --final-part val --train-source final --out %SPLITS%\splits_v1_val_minus_dev.parquet
python scripts\neuron_model\reconstruct_mogen_poisson.py --run %V3% --meshes-dir %V3%\meshes --set seed_1 --out %V3%\meshes_poisson_d8_pw8 --poisson-config %CFG%\d8_pw8.yaml --workers 8
python scripts\neuron_model\make_real_controls.py --manifest cache\minnie65_v1_60k\manifest.parquet --splits %SPLITS%\splits_v1_val_minus_dev.parquet --split test ^
  --controls poisson --poisson-config %CFG%\d8_pw8.yaml --n-samples 300 --seeds 1 --workers 8 --out %CMP%\real
python scripts\neuron_model\evaluate_generators.py ^
  --model mogen_v3=%V3%\meshes_poisson_d8_pw8 --model mogen_A=%MA%\meshes_poisson_d8_pw8 --model real_poisson=%CMP%\real\poisson ^
  "data.manifests=['%FULL%']" data.path_mode=absolute data.splits=%SPLITS%\splits_v1_val_minus_dev.parquet ^
  data.bbox=%SPLITS%\bbox_splits_v1.json data.drop_unassigned=true "protocol.generation_seeds=[1]" real.morphometrics_table=%MORPH% name=generator-evaluation-v3-vs-A
for /f "delims=" %%d in ('dir /b /ad /o-n %ANA%\*_generator-evaluation-v3-vs-A') do (set EVAL=%ANA%\%%d& goto :evalfound)
:evalfound
python scripts\neuron_model\classifier_ablation.py --eval-run %EVAL% --reference real_poisson --models mogen_v3 mogen_A
python scripts\neuron_model\morphometric_tails.py --eval-run %EVAL% --metrics Volume Area Length ConvexHullVolume
type %EVAL%\postprocessed\summary.md

echo ===== [AB2] H01 compartments with the inhibitory axon rule ===== %date% %time%
python scripts\neuron_model\apical_by_direction.py --reuse %ANA%\2026-11-03_015212_apical-by-direction-h01 --dataset h01 --branches-root %H01B% --out %H01DIR%
if errorlevel 1 (echo H01 COMPARTMENTS FAILED & goto :end)
copy /y %H01B%\branches.parquet %H01B%\branches_before_2026-11-18_compartments.parquet
python scripts\neuron_model\reapply_eligibility.py --output-root %H01B% --dataset H01_Seg --refresh-metadata --compartment-table %H01DIR%\limb_compartments_direction.parquet

echo ===== [AB3] split of H01 and the merged split ===== %date% %time%
if not exist %SPLITS%\splits_h01_v1.parquet python scripts\neuron_model\build_splits.py --config configs\neuron-model\data\splits_h01_v1.yaml "manifests=['%H01MAN%']"
if not exist %SPLITH% python scripts\neuron_model\merge_splits.py %SPLIT% %SPLITS%\splits_h01_v1.parquet --out %SPLITH%
if not exist %SPLITH% (echo NO MERGED SPLIT & goto :end)

echo ===== [AB4] H01 spines without the neck, H01 cache ===== %date% %time%
python scripts\neuron_model\detached_spine_keys.py --branches-root %H01B% --out %SPLITS%\h01_detached_spines.parquet
python scripts\neuron_model\cache_subset.py --manifest %H01MAN% --splits %SPLITH% --out cache\h01_v1 --n-train 60000 --n-val 3000 ^
  --exclude-keys %SPLITS%\h01_detached_spines.parquet

echo ===== [AB5] T0 / P0 / P1 on Minnie65 + H01 ===== %date% %time%
set ROOTS=%B%;%H01B%
python scripts\neuron_model\fit_t0.py --config configs\neuron-model\trunk\t0.yaml "data.branches_root=%ROOTS%" data.splits=%SPLITH% "model.mean_models=[parabola_mix]"
for /f "delims=" %%d in ('dir /b /ad /o-n %RUNS%\*_t0') do (set T0H=%RUNS%\%%d& goto :t0found)
:t0found
python scripts\neuron_model\fit_p0.py --config configs\neuron-model\placement\p0.yaml "data.branches_root=%ROOTS%" data.splits=%SPLITH% ^
  "model.families=[zinb]" "model.exposures=[length]" "model.random_effects=[true]" "model.sigma_by_cell_type=[true]"
for /f "delims=" %%d in ('dir /b /ad /o-n %RUNS%\*_p0') do (set P0H=%RUNS%\%%d& goto :p0found)
:p0found
python scripts\neuron_model\calibrate_p0_counts.py --p0-run %P0H%
python scripts\neuron_model\fit_p1.py --p0-run %P0H% --n-eval 400 --n-sim 19
for /f "delims=" %%d in ('dir /b /ad /o-n %RUNS%\*_p1') do (set P1H=%RUNS%\%%d& goto :p1found)
:p1found
echo T0 %T0H% / P0 %P0H% / P1 %P1H%
for %%k in (minnie65 h01) do (
  python scripts\neuron_model\generate_tp.py --t0-run %T0H% --p0-run %P1H% --mode generated --n 1000 --seed 1 --split val --dataset %%k --out-root %TPH%\%%k\generated
  python scripts\neuron_model\generate_tp.py --t0-run %T0H% --p0-run %P1H% --mode real_t --n 1000 --seed 1 --split val --dataset %%k --out-root %TPH%\%%k\real_t
)
for %%k in (minnie65 h01) do (
  for %%m in (generated real_t) do call :evalnewest %TPH%\%%k\%%m
)
python scripts\neuron_model\cv_tp_neurons.py --branches-root "%ROOTS%" --splits %SPLITH% --k 5 --repeats 1 --models p0 t0 ^
  --p0-variants zinb:length:re:sg2:t0.5 zinb:length:re:sg2

echo ===== [AB6] MoGen v4: Minnie65 + H01 with the species condition (GPU) ===== %date% %time%
if not exist cache\h01_v1\manifest.parquet (echo NO H01 CACHE & goto :end)
python scripts\neuron_model\train_mogen.py --config configs\neuron-model\mogen\conditional_v4_h01.yaml ^
  "data.manifests=['cache/minnie65_v1_60k/manifest.parquet','cache/h01_v1/manifest.parquet']" data.splits=%SPLITH% data.drop_unassigned=true ^
  "data.variants=[1,2]" "data.check_files=[]"

:end
echo ===== AB done ===== %date% %time%
goto :eof

:evalnewest
for /f "delims=" %%d in ('dir /b /ad /o-n %1') do (
  python scripts\neuron_model\evaluate_assembled_branches.py --tp-dir %1\%%d
  goto :eof
)
goto :eof
