@echo off
rem 2026-11-25, chain AN. T0 with random effects of the NEURON on the mean radius profile (parabola_mix_neuron). CPU only,
rem ~1 h (T0 fit ~30 min, six generations + evaluations ~5 min each). Can run in parallel with AK (CPU, mostly the
rem generator evaluation) and AM (GPU).
rem
rem Why (chain AK0): in the full T+P generation about half of the spine excess comes from REAL THIN branches that get a
rem thick generated trunk: only 26 % (Minnie65) / 30 % (H01) of the real thin branches stay thin, three times the chance
rem rate. The share of thin branches differs strongly between neurons and the T0 conditions do not carry it.
rem Model (t0_statistical.ParabolicMixtureNeuronMean): the pair theta = (log r_L, log r_0/r_L) of a branch of neuron n
rem   level effect  theta = B_k' x + a_n + e,  a_n ~ N(0, T)           (all branches of the neuron thicker / thinner)
rem   gate effect   P(k | x, n) = softmax(G' x + b_n s_k), b_n ~ N(0, tau^2)  (more / fewer branches in the thin component)
rem fitted by ECM (grid over b_n, Gaussian posterior of a_n). fit_t0 prints T, tau, the thin share per neuron (real /
rem independent effects / one effect per neuron) and the pairing P(gen thin | real thin) with the posterior of the
rem neuron from its OTHER validation branches.
rem AN1 fit_t0 on Minnie65 + H01 (same data and settings as T0 of chain AH, only the variant parabola_mix_neuron)
rem AN2 generate_tp on val, Minnie65 and H01, --t0-neuron independent / per_neuron / posterior (P1 of chain AB as in AH),
rem     evaluate_assembled_branches (now paired: counts only where the real source has one), diagnose_tp_counts
rem     -> runs\analysis\neuron-model\tp_generation\minnie_h01_neuron\{minnie65,h01}\generated\*
rem The same sets of chain AH (T0 without the effect) are re-evaluated paired by AK0 (evaluation_paired).
rem Log: scripts\neuron_model\overnight\2026-11-25_AN_t0-neuron-effect.bat > logs_AN.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set B=C:\Users\Student\LipsEkaterina\data\minnie65\preprocessed_branches
set H01B=C:\Users\Student\LipsEkaterina\data\h01\preprocessed_branches
set ROOTS=%B%;%H01B%
set SPLITS=data\processed\neuron-model\splits
set SPLITH=%SPLITS%\splits_v1_h01.parquet
set RUNS=runs\training\neuron-model
set ANA=runs\analysis\neuron-model
set P1H=%RUNS%\2026-11-17_072620_p1
set TPN=%ANA%\tp_generation\minnie_h01_neuron

cd /d %REPO%
if not exist %SPLITH% (echo NO %SPLITH% & goto :end)
if not exist %P1H%\p0_bundle.json (echo NO P1 run %P1H% & goto :end)

echo ===== [AN1] T0 with the neuron effects on Minnie65 + H01 ===== %date% %time%
python scripts\neuron_model\fit_t0.py --config configs\neuron-model\trunk\t0.yaml "data.branches_root=%ROOTS%" data.splits=%SPLITH% "model.mean_models=[parabola_mix_neuron]" name=t0-neuron
if errorlevel 1 (echo T0 FAILED & goto :end)
for /f "delims=" %%d in ('dir /b /ad /o-n %RUNS%\*_t0-neuron') do (set T0N=%RUNS%\%%d& goto :t0found)
:t0found
echo new T0 run: %T0N%

echo ===== [AN2] generation with the three neuron modes ===== %date% %time%
for %%k in (minnie65 h01) do (
  for %%m in (independent per_neuron posterior) do (
    python scripts\neuron_model\generate_tp.py --t0-run %T0N% --t0-model parabola_mix_neuron --p0-run %P1H% --mode generated --n 1000 --seed 1 --split val --dataset %%k --t0-neuron %%m --out-root %TPN%\%%k\generated
    call :evalnewest %TPN%\%%k\generated
  )
)

:end
echo ===== chain AN done ===== %date% %time%
goto :eof

:evalnewest
for /f "delims=" %%d in ('dir /b /ad /o-n %1') do (
  python scripts\neuron_model\evaluate_assembled_branches.py --tp-dir %1\%%d
  python scripts\neuron_model\diagnose_tp_counts.py --tp-dir %1\%%d
  goto :eof
)
goto :eof
