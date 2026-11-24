@echo off
rem 2026-11-27, chain AP. VN + FoldingNet + normalizing flow (the architecture used in the lab for spine embeddings) trained
rem on the SAME spines as MoGen A and the VAE and evaluated by the SAME protocol. GPU ~4-6 h (training), CPU ~5 h (Poisson
rem meshes, evaluation). Run when the GPU is free (after AM); the CPU part can overlap AL / AO.
rem
rem Model (src\neuron_model\spine_generation\models\spine_foldnet, docs\neuron-model\24.11-bridge-matching.md):
rem   encoder Vector Neurons DGCNN (rotation-equivariant), invariant code (512) = global VN features projected onto an
rem   equivariant frame R; decoder FoldingNet (2-D 45 x 45 grid folded twice), Chamfer loss, 2048 points, no rotations.
rem   AP1 mode frame     - the decoder output is rotated back by R (keeps the orientation in the dendrite frame);
rem   AP2 mode invariant - the code alone (the decoder must guess the orientation: tests our argument).
rem   Generation: codes of the 15 000 training spines -> RealNVP prior (the flow of our VAE) -> 500 samples x 2 seeds ->
rem   decoded on 8192 points -> Poisson d8_pw8 -> meshes (MoGen layout); recon_val = validation codes decoded (ceiling).
rem AP3 evaluate_generators.py as chain AI (test = val minus dev, seeds 1 + 2): vn_frame, vn_invariant, MoGen A, real
rem   controls; then the quality report (make_quality_report.py) of that evaluation.
rem Log: scripts\neuron_model\overnight\2026-11-27_AP_vn-foldingnet.bat > logs_AP.txt 2>&1

set REPO=C:\Users\Student\LipsEkaterina\GitHub\spinetool
set SPLITS=data\processed\neuron-model\splits
set RUNS=runs\training\neuron-model
set ANA=runs\analysis\neuron-model
set MA=%RUNS%\2026-10-20_213103_mogen-ft-dev
set MAN=%MA%\meshes_poisson_d8_pw8_n500
set CMP2=%ANA%\2026-11-21_v3-vs-A-seeds
set DEV=cache\minnie65_dev\manifest.parquet
set FULL=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/manifest_minnie65_full.parquet
set MORPH=C:/Users/Student/LipsEkaterina/GitHub/spinetool/cache/morphometrics_minnie65_full.parquet

cd /d %REPO%
if not exist %DEV% (echo NO %DEV% & goto :end)
if not exist %SPLITS%\splits_dev_100n.parquet (echo NO splits_dev_100n & goto :end)

for %%m in (frame invariant) do (
  echo ===== [AP] train VN + FoldingNet, mode %%m ===== %date% %time%
  python scripts\neuron_model\train_vn_foldingnet.py --config configs\neuron-model\vn_foldingnet\%%m.yaml ^
    "data.manifests=['%DEV%']" data.splits=%SPLITS%\splits_dev_100n.parquet
  call :generate %%m
)

echo ===== [AP3] evaluation, seeds 1 + 2 ===== %date% %time%
if not defined VN_frame (echo NO frame model & goto :end)
set MODELS=--model vn_frame=%VN_frame%\meshes_flow_d8_pw8 --model mogen_A=%MAN% --model real_poisson=%CMP2%\real\poisson
if defined VN_invariant set MODELS=%MODELS% --model vn_invariant=%VN_invariant%\meshes_flow_d8_pw8
python scripts\neuron_model\evaluate_generators.py %MODELS% ^
  "data.manifests=['%FULL%']" data.path_mode=absolute data.splits=%SPLITS%\splits_v1_val_minus_dev.parquet ^
  data.bbox=%SPLITS%\bbox_splits_v1.json data.drop_unassigned=true "protocol.generation_seeds=[1,2]" real.morphometrics_table=%MORPH% ^
  real.chamfer_memorization=false name=generator-evaluation-vn-foldingnet
for /f "delims=" %%d in ('dir /b /ad /o-n %ANA%\*_generator-evaluation-vn-foldingnet') do (set EVAL=%ANA%\%%d& goto :evalfound)
:evalfound
echo evaluation run: %EVAL%
python scripts\neuron_model\make_quality_report.py --out %EVAL%\quality --s-eval %EVAL%
python scripts\neuron_model\classifier_ablation.py --eval-run %EVAL% --reference real_poisson --models vn_frame vn_invariant mogen_A
type %EVAL%\postprocessed\summary.md

:end
echo ===== chain AP done ===== %date% %time%
goto :eof

:generate
for /f "delims=" %%d in ('dir /b /ad /o-n %RUNS%\*_vn-foldingnet-%1') do (set VN_%1=%RUNS%\%%d& goto :genfound)
echo NO run of mode %1
goto :eof
:genfound
call set RUN=%%VN_%1%%
echo run of mode %1: %RUN%
python scripts\neuron_model\generate_vn_foldingnet.py --run %RUN% --mode reconstruction --n-samples 300
python scripts\neuron_model\generate_vn_foldingnet.py --run %RUN% --mode prior --n-samples 500 --seeds 1 2
goto :eof
