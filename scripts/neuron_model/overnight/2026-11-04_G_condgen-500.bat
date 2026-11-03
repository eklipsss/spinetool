@echo off
rem Chain G (GPU + CPU meshing, ~4-5 h). Conditional MoGen v3 (2026-11-03_015237): 500 spines per condition for the HTML report of the
rem conditional generation (statistical tests need more than 200 samples: power of the generated-vs-generated test and the
rem noise floor of the generated-vs-real test). Guidance 1.25 is the working value between 1 and 1.5 (chains C, E:
rem w = 1 reproduces 0.3-1.1 of the real difference inhibitory / excitatory, w = 1.5 exaggerates up to 1.9x), w = 1 as a
rem second point of the dose-response, and the UNCONDITIONAL set (zero condition vector) as the baseline.
rem Sets: <run>\meshes\cond_cell_type_binary-<value>_g<w>_seed_1 and <run>\meshes\seed_1 (unconditional).
rem Log: scripts\neuron_model\overnight\2026-11-04_G_condgen-500.bat > logs_G.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set V3=runs\training\neuron-model\2026-11-03_015237_mogen-conditional-v3
set N=500

cd /d %REPO%

echo ===== [G1/3] unconditional baseline ===== %date% %time%
python scripts\neuron_model\generate_mogen.py --run %V3% --checkpoint latest --n-samples %N% --seed 1

echo ===== [G2/3] guidance 1.25 ===== %date% %time%
for %%c in (excitatory inhibitory) do (
  python scripts\neuron_model\generate_mogen.py --run %V3% --checkpoint latest --condition cell_type_binary=%%c --guidance 1.25 --n-samples %N% --seed 1
)

echo ===== [G3/3] guidance 1 ===== %date% %time%
for %%c in (excitatory inhibitory) do (
  python scripts\neuron_model\generate_mogen.py --run %V3% --checkpoint latest --condition cell_type_binary=%%c --guidance 1 --n-samples %N% --seed 1 --tag cond_cell_type_binary-%%c_g1_seed_1_n500
)

echo ===== G done ===== %date% %time%
