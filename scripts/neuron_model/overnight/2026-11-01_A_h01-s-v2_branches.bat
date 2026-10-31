@echo off
rem Overnight 2026-11-01, chain A (CPU + NAS reads of H01_Seg). Run from the repository root, env neuron-model active.
rem   1. S preprocessing of H01 AGAIN, into a NEW folder preprocessed_v2 (~3 h): the first run lost 30 % of the spines at
rem      sealing (loose fragments + duplicated faces in the raw meshes) and applied the Minnie threshold 300 nm instead of
rem      h01: 0. Fixed in spine_preprocessing_h01.yaml (input_drop_duplicate_faces, input_min_component_face_fraction) and
rem      in SpinePreprocessingConfig.false_spine_threshold. The seal hash changed, so the old folder must not be resumed.
rem   2. branch preprocessing of H01 on the new S output (~2 h).
rem Chain B (2026-11-01_B_cache-v1_mogen-cond.bat) can run at the same time in a second window.

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set H01_S_ROOT=C:\Users\Student\LipsEkaterina\data\h01\preprocessed_v2
set H01_B=C:\Users\Student\LipsEkaterina\data\h01\preprocessed_branches

cd /d %REPO%

echo ===== [A1/2] H01 S preprocessing v2 -^> %H01_S_ROOT% ===== %date% %time%
python scripts\run_spine_preprocessing.py --raw-data-root O:\Datasets\H01 --dataset H01_Seg --metadata-root O:\Datasets ^
  --config data\processed\config\spine_preprocessing_h01.yaml --output-root %H01_S_ROOT% ^
  --workers 20 --batch-size 2048 --neurons-per-chunk 25 --stage all

echo ===== [A1/2] summary ===== %date% %time%
python scripts\neuron_model\summarize_s_tables.py --preprocessed-root %H01_S_ROOT%\H01_Seg

echo ===== [A2/2] H01 branch preprocessing -^> %H01_B% ===== %date% %time%
python scripts\neuron_model\run_branch_preprocessing.py --dataset H01_Seg --raw-data-root O:\Datasets\H01 --metadata-root O:\Datasets ^
  --s-preprocessed-root %H01_S_ROOT%\H01_Seg --output-root %H01_B% --only-s-neurons --workers 16 --io-threads 16

echo ===== A done ===== %date% %time%
