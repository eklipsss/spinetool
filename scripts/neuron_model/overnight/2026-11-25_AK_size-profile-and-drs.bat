@echo off
rem 2026-11-25, chain AK. Size bias of generated spines: where the excess is, and does density-ratio selection remove it.
rem Run after git pull, AFTER the rerun of chain AJ (logs_AJ2.txt) has finished. CPU only, ~6-8 h (most of it AK3).
rem
rem AK0 (CPU, minutes) diagnose_tp_counts.py on the T+P sets of chain AH (Minnie65 and H01, full generation): splits the excess of
rem           spines per branch (11 vs 7) by thin / thick status of the generated and the real branch and by branch group
rem           -> <set>\evaluation*\count_diagnosis
rem AK1 (CPU, ~15 min) cloud_size_diagnostics.py with the new radius profile along the spine axis (height deciles):
rem           real clouds of the test spines vs A, v3, v4 and the AJ sweep (steps 0..16000)
rem           -> runs\analysis\neuron-model\2026-11-25_cloud-profile
rem AK2 (CPU, minutes) density_ratio_selection.py for A and v3: a classifier real vs generated on the 11 morphometrics is fitted
rem           on TRAINING real spines of splits_v1 (30 000 sampled) and generation seed 1, every seed-2 sample is kept with
rem           probability min(1, r / M), r = p_real / p_gen; checked on the held-out real neurons (val minus dev) as in chain AI.
rem           Prototype on the mac (AI tables, neurons split in halves): kept 34-48 %, significant metrics 6-8 -> 0 of 11,
rem           AUC vs real 0.73-0.79 -> 0.59-0.62; a random subset of the same size stays at 5-7 of 11.
rem           The kept seed-2 samples are copied as new sets -> runs\analysis\neuron-model\2026-11-25_drs\{A,v3}\seed_2
rem AK3 (CPU, ~5-6 h) evaluate_generators.py on generation seed 2: A, A + selection, v3, v3 + selection, real controls,
rem           same protocol as AI (Chamfer metrics, coverage, memorization, classifier), + paired neuron bootstrap
rem           -> <date>_generator-evaluation-drs-seed2
rem Log: scripts\neuron_model\overnight\2026-11-25_AK_size-profile-and-drs.bat > logs_AK.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set SPLITS=data\processed\neuron-model\splits
set RUNS=runs\training\neuron-model
set ANA=runs\analysis\neuron-model
set V3=%RUNS%\2026-11-03_015237_mogen-conditional-v3
set V4=%RUNS%\2026-11-17_085545_mogen-conditional-v4-h01
set MA=%RUNS%\2026-10-20_213103_mogen-ft-dev
set AC=%RUNS%\2026-11-04_095002_mogen-ft-dev-curves
set P3=%V3%\meshes_poisson_d8_pw8
set P4=%V4%\meshes_poisson_d8_pw8
set MAN=%MA%\meshes_poisson_d8_pw8_n500
set PAC=%AC%\meshes_poisson_d8_pw8
set CMP2=%ANA%\2026-11-21_v3-vs-A-seeds
set EVAI=%ANA%\2026-11-21_045820_generator-evaluation-v3-vs-A-n500-2seeds
set DRS=%ANA%\2026-11-25_drs
set TPH=%ANA%\tp_generation\minnie_h01_acfit
set FULL=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/manifest_minnie65_full.parquet
set MORPH=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/morphometrics_minnie65_full.parquet

cd /d %REPO%
if not exist %EVAI%\real_test_morphometrics.parquet (echo NO evaluation of chain AI & goto :end)
if not exist %CMP2%\real\poisson\seed_2\sample_00299 (echo NO real controls of chain AI & goto :end)
if not exist %MAN%\seed_2 (echo NO A seed_2 & goto :end)
if not exist %P3%\seed_2 (echo NO v3 seed_2 & goto :end)
if not exist %SPLITS%\splits_v1.parquet (echo NO splits_v1 & goto :end)

echo ===== [AK0] spines per branch: where the excess comes from ===== %date% %time%
for %%k in (minnie65 h01) do (
  for /f "delims=" %%d in ('dir /b /ad /o-n %TPH%\%%k\generated') do (
    python scripts\neuron_model\diagnose_tp_counts.py --tp-dir %TPH%\%%k\generated\%%d
  )
)

echo ===== [AK1] radius profile of the point clouds ===== %date% %time%
python scripts\neuron_model\cloud_size_diagnostics.py --manifest cache\minnie65_v1_60k\manifest.parquet --real real=%CMP2%\real\poisson\seed_1 ^
  --gen A-s1=%MAN%\seed_1 --gen A-s2=%MAN%\seed_2 --gen v3-s1=%P3%\seed_1 --gen v3-s2=%P3%\seed_2 ^
  --gen v4-23P-g1.5=%P4%\v4_mouse_23P_g1.5_seed_1 --gen v4-6P-CT-g1.5=%P4%\v4_mouse_6P-CT_g1.5_seed_1 ^
  --gen Acurves-step0=%PAC%\sweep_step0 --gen Acurves-step1000=%PAC%\sweep_step1000 --gen Acurves-step3000=%PAC%\sweep_step3000 ^
  --gen Acurves-step8000=%PAC%\sweep_step8000 --gen Acurves-step16000=%PAC%\sweep_step16000 ^
  --out %ANA%\2026-11-25_cloud-profile
if errorlevel 1 (echo CLOUD PROFILE FAILED)

echo ===== [AK2] density-ratio selection, A and v3 ===== %date% %time%
python scripts\neuron_model\density_ratio_selection.py --real-fit %MORPH% --real-fit-splits %SPLITS%\splits_v1.parquet --real-fit-split train ^
  --gen-fit %EVAI%\postprocessed\morphometrics_mogen_A_seed1.parquet --real-eval %EVAI%\real_test_morphometrics.parquet ^
  --gen-eval %EVAI%\postprocessed\morphometrics_mogen_A_seed2.parquet --floor %EVAI%\postprocessed\morphometrics_real_poisson_seed2.parquet ^
  --gen-eval-dir %MAN%\seed_2 --export %DRS%\A --out %DRS%\report_A
if errorlevel 1 (echo SELECTION A FAILED & goto :end)
python scripts\neuron_model\density_ratio_selection.py --real-fit %MORPH% --real-fit-splits %SPLITS%\splits_v1.parquet --real-fit-split train ^
  --gen-fit %EVAI%\postprocessed\morphometrics_mogen_v3_seed1.parquet --real-eval %EVAI%\real_test_morphometrics.parquet ^
  --gen-eval %EVAI%\postprocessed\morphometrics_mogen_v3_seed2.parquet --floor %EVAI%\postprocessed\morphometrics_real_poisson_seed2.parquet ^
  --gen-eval-dir %P3%\seed_2 --export %DRS%\v3 --out %DRS%\report_v3
if errorlevel 1 (echo SELECTION v3 FAILED & goto :end)

echo ===== [AK3] evaluation on generation seed 2 ===== %date% %time%
python scripts\neuron_model\evaluate_generators.py ^
  --model mogen_A=%MAN% --model mogen_A_drs=%DRS%\A --model mogen_v3=%P3% --model mogen_v3_drs=%DRS%\v3 --model real_poisson=%CMP2%\real\poisson ^
  "data.manifests=['%FULL%']" data.path_mode=absolute data.splits=%SPLITS%\splits_v1_val_minus_dev.parquet ^
  data.bbox=%SPLITS%\bbox_splits_v1.json data.drop_unassigned=true "protocol.generation_seeds=[2]" real.morphometrics_table=%MORPH% ^
  real.chamfer_memorization=false name=generator-evaluation-drs-seed2
if errorlevel 1 (echo EVALUATION FAILED & goto :end)
for /f "delims=" %%d in ('dir /b /ad /o-n %ANA%\*_generator-evaluation-drs-seed2') do (set EVAL=%ANA%\%%d& goto :evalfound)
:evalfound
echo evaluation run: %EVAL%
python scripts\neuron_model\paired_generator_bootstrap.py --eval-run %EVAL% --models mogen_A mogen_A_drs mogen_v3 mogen_v3_drs real_poisson ^
  --pairs mogen_A_drs:mogen_A mogen_A_drs:real_poisson mogen_v3_drs:mogen_v3 mogen_A_drs:mogen_v3_drs --n-boot 1000
python scripts\neuron_model\classifier_ablation.py --eval-run %EVAL% --reference real_poisson --models mogen_A mogen_A_drs mogen_v3 mogen_v3_drs
type %EVAL%\postprocessed\summary.md

:end
echo ===== chain AK done ===== %date% %time%
