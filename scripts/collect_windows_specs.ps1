<#
  Collects Windows workstation characteristics relevant to running the
  neuron-model preprocessing pipeline (docs/neuron-model/s-module-preprocessing.md)
  on the full Minnie/H01 datasets: CPU/RAM sizing for --workers/--batch-size,
  local-disk vs NAS (O:\) I/O throughput, small-file write performance,
  filesystem limits, and Python/CGAL compatibility.

  Usage (from an Anaconda PowerShell Prompt or a regular PowerShell window,
  run from the repository root):

      powershell -ExecutionPolicy Bypass -File scripts\collect_windows_specs.ps1

  The script writes a plain-text report to your Desktop and opens it in
  Notepad. Paste its full contents back into the chat so it can be recorded
  in docs/neuron-model/windows-workstation-specs.md.
#>

$report = Join-Path $env:USERPROFILE "Desktop\windows_specs_report.txt"
# Every Out-File call below explicitly uses -Encoding utf8 (never the cmdlet
# default). Mixing default-encoded and explicitly-utf8-encoded writes to the
# same file previously produced a mixed-encoding, unreadable report.
"Windows workstation specs report - $(Get-Date -Format o)" | Out-File $report -Encoding utf8

function Add-Section {
    param([string]$Title, [scriptblock]$Block)
    "`n" + ("=" * 80) | Out-File $report -Append -Encoding utf8
    "== $Title" | Out-File $report -Append -Encoding utf8
    ("=" * 80) | Out-File $report -Append -Encoding utf8
    try {
        & $Block 2>&1 | Out-String | Out-File $report -Append -Encoding utf8
    } catch {
        "ERROR: $_" | Out-File $report -Append -Encoding utf8
    }
}

Add-Section "OS" {
    Get-ComputerInfo | Select-Object OsName, OsVersion, OsBuildNumber, OsArchitecture, WindowsProductName, CsSystemType, CsManufacturer, CsModel
}

Add-Section "CPU" {
    Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed
}

Add-Section "RAM" {
    Get-CimInstance Win32_OperatingSystem |
        Select-Object @{n = 'TotalMemoryGB'; e = { [math]::Round($_.TotalVisibleMemorySize / 1MB, 1) } },
                       @{n = 'FreeMemoryGB'; e = { [math]::Round($_.FreePhysicalMemory / 1MB, 1) } }
    Get-CimInstance Win32_PhysicalMemory | Select-Object Manufacturer, @{n='CapacityGB';e={[math]::Round($_.Capacity/1GB,1)}}, Speed, DeviceLocator
}

Add-Section "GPU" {
    # Win32_VideoController.AdapterRAM is a 32-bit field and wraps/truncates
    # for cards with >4 GB VRAM (e.g. reports 4 GB for a 24 GB RTX 4090), so
    # prefer nvidia-smi for the real VRAM size when an NVIDIA GPU is present.
    Get-CimInstance Win32_VideoController | Select-Object Name, @{n='VRAM_GB_WMI_unreliable_above_4GB';e={[math]::Round($_.AdapterRAM/1GB,1)}}, DriverVersion
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    }
}

Add-Section "Physical disks" {
    Get-PhysicalDisk | Select-Object FriendlyName, MediaType, BusType, @{n='SizeGB';e={[math]::Round($_.Size/1GB,1)}}, HealthStatus
}

Add-Section "Volumes / drive letters" {
    Get-Volume | Select-Object DriveLetter, FileSystemLabel, FileSystem, @{n='SizeGB';e={[math]::Round($_.Size/1GB,1)}}, @{n='FreeGB';e={[math]::Round($_.SizeRemaining/1GB,1)}}
}

Add-Section "NTFS long paths enabled (registry)" {
    Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -ErrorAction Stop |
        Select-Object LongPathsEnabled
}

Add-Section "Power plan" {
    powercfg /getactivescheme
}

Add-Section "Windows Defender real-time protection" {
    Get-MpPreference | Select-Object DisableRealtimeMonitoring, ExclusionPath, ExclusionProcess
}

Add-Section "Network adapters" {
    Get-NetAdapter | Where-Object Status -eq 'Up' | Select-Object Name, InterfaceDescription, LinkSpeed
}

Add-Section "Mapped drives / net use (look for O:)" {
    Get-PSDrive -PSProvider FileSystem | Select-Object Name, Root, @{n='UsedGB';e={[math]::Round($_.Used/1GB,1)}}, @{n='FreeGB';e={[math]::Round($_.Free/1GB,1)}}
    net use
}

Add-Section "NAS path O:\ reachability" {
    if (Test-Path O:\) { "O:\ is reachable." } else { "O:\ is NOT reachable right now - map it first." }
}

Add-Section "NAS throughput test: O:\ single-file write+read (200 MB)" {
    if (Test-Path O:\) {
        $testDir = "O:\_speedtest_tmp"
        New-Item -ItemType Directory -Force -Path $testDir | Out-Null
        $testFile = Join-Path $testDir "speedtest.bin"
        $bytes = New-Object byte[] (200 * 1MB)
        (New-Object Random).NextBytes($bytes)
        $w = Measure-Command { [IO.File]::WriteAllBytes($testFile, $bytes) }
        $r = Measure-Command { [IO.File]::ReadAllBytes($testFile) | Out-Null }
        Remove-Item $testFile -Force
        Remove-Item $testDir -Force
        "Write: {0:N1} MB/s ({1:N2} s)" -f (200 / $w.TotalSeconds), $w.TotalSeconds
        "Read:  {0:N1} MB/s ({1:N2} s)" -f (200 / $r.TotalSeconds), $r.TotalSeconds
    } else { "O:\ not reachable, skipped." }
}

Add-Section "NAS throughput test: O:\ many small files (500 x 4 KB)" {
    # The preprocessing pipeline writes ~15 small files per spine
    # (sealed/local/local_sealed .off, transform/attachment/quality .json,
    # pointcloud_*.npz, sdf_samples.npz), so per-file overhead on the network
    # share matters as much as raw MB/s throughput.
    if (Test-Path O:\) {
        $dir = "O:\_speedtest_manyfiles"
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        $data = New-Object byte[] 4096
        $t = Measure-Command {
            1..500 | ForEach-Object { [IO.File]::WriteAllBytes("$dir\f_$_.bin", $data) }
        }
        Remove-Item $dir -Recurse -Force
        "500 x 4KB files written in {0:N2} s ({1:N0} files/s)" -f $t.TotalSeconds, (500 / $t.TotalSeconds)
    } else { "O:\ not reachable, skipped." }
}

Add-Section "Local-disk throughput test: many small files (same test, on the drive the repo will live on)" {
    $dir = Join-Path $env:TEMP "_speedtest_manyfiles"
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $data = New-Object byte[] 4096
    $t = Measure-Command {
        1..500 | ForEach-Object { [IO.File]::WriteAllBytes("$dir\f_$_.bin", $data) }
    }
    Remove-Item $dir -Recurse -Force
    "500 x 4KB files written in {0:N2} s ({1:N0} files/s) on {2}" -f $t.TotalSeconds, (500 / $t.TotalSeconds), $env:TEMP
}

Add-Section "Python / conda" {
    conda --version
    python --version
    where.exe python
}

Write-Host "Report written to $report"
notepad $report
