# Windows workstation specs (full-dataset preprocessing)

Collected via `scripts/collect_windows_specs.ps1`, first run 2026-09-27
(encoding-corrupted, recovered by reversing the transform), confirmed with a
clean re-run the same day after fixing the script's `Out-File` encoding
(`legacy/` is gitignored, so neither raw report is committed - both were only
ever local files). The two runs agree on every measurement, including the
slow NAS read throughput below.

## Context

- macOS (this machine): code + tests on a small local subset of Minnie65
  (`datasets/minnie65/<neuron_id>/...`, three neurons).
- Windows workstation: full-scale preprocessing of Minnie/H01/LabID
  (`docs/neuron-model/s-module-preprocessing.md`), with its own CGAL Python
  bindings build (Windows, Python 3.10 - see `README.md` "Build CGAL for
  other platforms"), `requirements/conda_win_requirements.txt` (also
  Python 3.10 to match).
- Raw datasets live on a NAS (`\\LabidNAS\NAS common`) on the same LAN as the
  Windows workstation, mounted as `O:\` through File Explorer / SMB.

## Characteristics

| Item | Value | Notes for the preprocessing pipeline |
| --- | --- | --- |
| OS / build | Windows 10 Pro for Workstations, build 19045, x64 (MSI MS-7E07) | Meets the README's "Windows 8+" minimum; long paths already enabled (see below). |
| CPU | Intel Core i9-14900KF, 24 cores / 32 logical processors, 3.2 GHz base | `SpinePreprocessingConfig.workers`: start around 20-28 (leave a few logical processors for OS/Defender/NAS I/O) and tune from actual throughput. |
| RAM | 31.8 GB total; 2.7 GB free on the first run, 12.2 GB free on the second (varies with what else is open) | Recheck free RAM right before a real run. Size `batch_size` so `workers x batch_size` meshes + their point clouds/SDF pools comfortably fit in free RAM; with ~32 GB total, keep per-worker memory modest or reduce `workers`. |
| GPU | NVIDIA GeForce RTX 4090, 24 GB VRAM (24,564 MiB via `nvidia-smi`), driver 610.88. (WMI's `AdapterRAM` reported "4 GB" - a known 32-bit-field wraparound for cards >4 GB, now bypassed by the script's `nvidia-smi` fallback.) | Not used by preprocessing (CPU-only); relevant later for PointNeXt/VAE/MoGen training (`spine_generation_technical_spec.md`) - 24 GB VRAM is generous for that. |
| Local disks | `ST2000DM008-2UB102` (2 TB, HDD, SATA); `Samsung SSD 980 PRO` (2 TB, SSD, NVMe) | Put the repo, conda env, and `output_root` for `preprocessed/` on the NVMe SSD volume. |
| Volumes | `C:` NTFS, 1862 GB (483 GB free); `D:` ("Data") NTFS, 1863 GB (355 GB free); one small unlabeled ~0.7 GB volume (EFI/recovery) | `D:\` looks like the intended data drive - good candidate for `output_root`. |
| NTFS long paths enabled | Yes (`LongPathsEnabled = 1`) | Good - `preprocessed/<neuron>/limb_*/branch_*/spines/spine_<id>/...` paths are deep enough that this matters. |
| Power plan | High performance | Already optimal for long batch runs. |
| Windows Defender real-time protection | Query failed both times (`HRESULT 0x800106ba`, `Get-MpPreference` CIM error) - inconclusive, not necessarily disabled | The Defender WMI provider fails like this when a third-party AV is installed instead, or the query needs elevation. Check manually (Windows Security app) whether real-time protection is on, and if so, add `output_root` (and the raw dataset path if not already covered) to its exclusions before a full run - this is a common, large slowdown source for many-small-file workloads. |
| Network adapters | Wi-Fi: TP-Link Wireless MU-MIMO USB Adapter, up to 866.7 Mbps; Ethernet: Intel I226-V, 1 Gbps (both reported "Up") | If both are active simultaneously, Windows' route selection decides which one actually carries NAS traffic - worth confirming Ethernet is the one in use, since the measured NAS throughput below is far below either link's theoretical ceiling. |
| NAS (`O:\`) | Reachable, mapped to `\\LabidNAS\NAS common`; ~1251 GB used, ~30.9 TB free | Plenty of capacity for the full datasets. |
| NAS single-file throughput (200 MB) | Write ~11.3 MB/s (~17.7 s); **read ~1.1 MB/s (~175 s)**, consistent across both runs | Confirmed, not a fluke: both write and read are far below what even the slower (866 Mbps Wi-Fi, ~108 MB/s theoretical) link should give, and read is ~10x slower than write - unusual for a NAS and points at a real bottleneck (NAS-side disk/CPU, SMB protocol version/signing overhead, or wrong adapter carrying the traffic), not local measurement noise. Worth investigating on the NAS/network side before relying on `O:\` for any bulk sequential IO. |
| NAS many-small-files (500 x 4 KB) | ~178 files/s (~2.8 s), consistent across both runs | At ~15 files/spine, this alone would need ~14 minutes of pure write time per 100k spines - a real bottleneck at full Minnie/H01 scale. |
| Local-disk many-small-files (same test, `C:\...\Temp`) | 8,658-11,068 files/s | **~50-60x faster** than the NAS for small-file writes. |
| Python / conda (base env, not yet `neuron-model`) | conda 25.7.0, base Python 3.13.5 | The `neuron-model` env itself will be created separately with `python=3.10` to match the Windows CGAL build. |

## Recommendations

- **`output_root` on a local disk (`D:\`) was the original recommendation**,
  but a real-data size measurement in `CLAUDE.md` ("Полный прогон Minnie65 на
  Windows") puts the full Minnie65 output at **~1.4 TB** (~0.76M spines) - too big for either
  local volume (`C:` 470 GB free, `D:` 355 GB free at the time of this
  report), so for the actual full run `output_root` has to be on `O:\`
  (~30 TB free) instead. The many-small-file write gap below is therefore a
  real cost of the full run, not just a theoretical risk - see the mitigations
  in `CLAUDE.md` (bigger `batch_size`, pilot timing run, network check below).
- The NAS read/write throughput is confirmed slow (not a one-off fluke) and
  far below both adapters' theoretical link speed. Check which adapter
  actually carries `O:\` traffic before doing any bulk NAS I/O:
  ```powershell
  Get-NetAdapter                                    # confirm Ethernet/Wi-Fi Up/Disconnected
  $nasIp = (Resolve-DnsName LabidNAS).IPAddress[0]   # or whatever O:\ actually resolves to
  Find-NetRoute -RemoteIPAddress $nasIp              # -> InterfaceAlias Windows will actually use
  ```
  Cross-check empirically (the route table can be right and actual traffic
  still end up elsewhere in edge cases): run `Get-NetAdapterStatistics`,
  copy one real file from `O:\` (e.g. `Copy-Item O:\Datasets\Minnie65\<...>\
  spine_000.off C:\temp\test.off`), run `Get-NetAdapterStatistics` again -
  whichever adapter's `ReceivedBytes` jumped is the one really carrying it.
  If it's Wi-Fi and Ethernet shows "Up" in the first command, forcing
  Ethernet (disable the Wi-Fi adapter for the duration of the run, or lower
  Ethernet's `InterfaceMetric` via `Set-NetIPInterface` - both need admin) is
  likely the single biggest speed win available, bigger than any
  `workers`/`batch_size` tuning.
- Confirm Windows Defender real-time protection status manually - the
  `Get-MpPreference` CIM query already failed once (see above); try
  `Get-MpComputerStatus | Select-Object RealTimeProtectionEnabled` first
  (sometimes works when `Get-MpPreference` doesn't), otherwise check via the
  Windows Security app (Virus & threat protection -> Virus & threat
  protection settings -> Manage settings). If enabled, exclude the raw
  dataset path and `output_root` (Add or remove exclusions -> Add an
  exclusion -> Folder; or `Add-MpPreference -ExclusionPath "O:\Datasets\
  Minnie65"` from an elevated PowerShell). On a managed/lab machine this may
  need an admin/IT - if the Security app shows "some settings are managed by
  your organization" the account can't change it itself.
- Start `workers` around 20-28 (of 32 logical processors) and `batch_size`
  small enough that `workers x batch_size` in-flight meshes/point
  clouds/SDF pools stay well under free RAM - re-check free RAM before the
  run (it ranged 2.7-12.2 GB free across the two reports here). Note this
  predates the ~0.76M-spine full-Minnie65 scale estimate in `CLAUDE.md`, which
  revises `batch_size` upward (to ~4096) for a different reason - the
  per-batch full-table parquet rewrite, not memory.
