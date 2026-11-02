@echo off
rem Overnight 2026-11-03, chain C (GPU). Run from any folder, env neuron-model active. ~4-5 h in total.
rem   1. conditional MoGen v3: v2 + lr x100 for the condition layer Dense_0 (configs\neuron-model\mogen\conditional_v3.yaml),
rem      15 000 steps (~2.5 h)
rem   2. check_condition_use of v3 (binary type train / val, detailed type train)
rem   3. 200 spines per condition (excitatory / inhibitory) x guidance 1 and 3, for v2 and v3 (~8 sets)
rem   4. compare_conditional_generation: do generated inhibitory spines differ from excitatory ones as the real ones do?
rem Log: redirect the output, e.g.  scripts\neuron_model\overnight\2026-11-03_C_mogen-v3_condgen.bat > logs_C.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set CACHE=%REPO%\cache\minnie65_v1_60k
set SPLITS=data\processed\neuron-model\splits\splits_v1.parquet
set V2=runs\training\neuron-model\2026-11-02_023219_mogen-conditional-v2
set MORPH=%REPO%\cache\morphometrics_minnie65_full.parquet
set N=200

cd /d %REPO%

echo ===== [C1/4] MoGen v3 ===== %date% %time%
python scripts\neuron_model\train_mogen.py --config configs\neuron-model\mogen\conditional_v3.yaml ^
  "data.manifests=['%CACHE:\=/%/manifest.parquet']" data.splits=%SPLITS:\=/% data.drop_unassigned=true "data.variants=[1,2]" "data.check_files=[]"

rem the newest v3 run folder
for /f "delims=" %%d in ('dir /b /ad /o-d runs\training\neuron-model\*_mogen-conditional-v3') do (set V3=runs\training\neuron-model\%%d& goto :found)
:found
echo v3 run: %V3%

echo ===== [C2/4] check_condition_use v3 ===== %date% %time%
python scripts\neuron_model\check_condition_use.py --run %V3% --split train --block cell_type_binary
python scripts\neuron_model\check_condition_use.py --run %V3% --split val --block cell_type_binary
python scripts\neuron_model\check_condition_use.py --run %V3% --split train --block cell_type

echo ===== [C3/4] generation by condition (v2, v3) ===== %date% %time%
for %%r in (%V2% %V3%) do (
  for %%c in (excitatory inhibitory) do (
    for %%g in (1 3) do (
      python scripts\neuron_model\generate_mogen.py --run %%r --checkpoint latest --condition cell_type_binary=%%c --guidance %%g --n-samples %N% --seed 1
    )
  )
)

echo ===== [C4/4] generated vs real by condition ===== %date% %time%
for %%r in (%V2% %V3%) do (
  python scripts\neuron_model\compare_conditional_generation.py --block cell_type_binary --morphometrics-table %MORPH% ^
    --gen excitatory@g1=%%r\meshes\cond_cell_type_binary-excitatory_g1_seed_1 --gen inhibitory@g1=%%r\meshes\cond_cell_type_binary-inhibitory_g1_seed_1 ^
    --gen excitatory@g3=%%r\meshes\cond_cell_type_binary-excitatory_g3_seed_1 --gen inhibitory@g3=%%r\meshes\cond_cell_type_binary-inhibitory_g3_seed_1
)

echo ===== C done ===== %date% %time%
