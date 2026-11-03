@echo off
rem Chain E (GPU, ~3-4 h). Conditional MoGen v3 (2026-11-03_015237): the guidance that reproduces the real difference
rem inhibitory / excitatory, and whether the DETAILED cell type is used.
rem Result of chain C: w = 1 reproduces 0.3-1.1 of the real differences, w = 3 exaggerates them 1.5-4x -> try 1.5 and 2.
rem   1. binary type: excitatory / inhibitory x w 1.5, 2 (200 spines each)
rem   2. detailed type at w 1.5: 4P and 6P-CT (excitatory, Length differs by +27.5 %), BPC and MC (inhibitory)
rem   3. compare_conditional_generation for both
rem Log: scripts\neuron_model\overnight\2026-11-04_E_guidance_detailed-types.bat > logs_E.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set V3=runs\training\neuron-model\2026-11-03_015237_mogen-conditional-v3
set MORPH=%REPO%\cache\morphometrics_minnie65_full.parquet
set N=200

cd /d %REPO%

echo ===== [E1/3] binary type, guidance 1.5 and 2 ===== %date% %time%
for %%c in (excitatory inhibitory) do (
  for %%g in (1.5 2) do (
    python scripts\neuron_model\generate_mogen.py --run %V3% --checkpoint latest --condition cell_type_binary=%%c --guidance %%g --n-samples %N% --seed 1
  )
)

echo ===== [E2/3] detailed types, guidance 1.5 ===== %date% %time%
for %%t in (4P 6P-CT BPC MC) do (
  python scripts\neuron_model\generate_mogen.py --run %V3% --checkpoint latest --condition cell_type=%%t --guidance 1.5 --n-samples %N% --seed 1 --tag cond_cell_type-%%t_g1.5_seed_1
)

echo ===== [E3/3] generated vs real ===== %date% %time%
python scripts\neuron_model\compare_conditional_generation.py --block cell_type_binary --morphometrics-table %MORPH% ^
  --gen excitatory@g1.5=%V3%\meshes\cond_cell_type_binary-excitatory_g1.5_seed_1 --gen inhibitory@g1.5=%V3%\meshes\cond_cell_type_binary-inhibitory_g1.5_seed_1 ^
  --gen excitatory@g2=%V3%\meshes\cond_cell_type_binary-excitatory_g2_seed_1 --gen inhibitory@g2=%V3%\meshes\cond_cell_type_binary-inhibitory_g2_seed_1
python scripts\neuron_model\compare_conditional_generation.py --block cell_type --morphometrics-table %MORPH% ^
  --gen 4P@g1.5=%V3%\meshes\cond_cell_type-4P_g1.5_seed_1 --gen 6P-CT@g1.5=%V3%\meshes\cond_cell_type-6P-CT_g1.5_seed_1 ^
  --gen BPC@g1.5=%V3%\meshes\cond_cell_type-BPC_g1.5_seed_1 --gen MC@g1.5=%V3%\meshes\cond_cell_type-MC_g1.5_seed_1

echo ===== E done ===== %date% %time%
