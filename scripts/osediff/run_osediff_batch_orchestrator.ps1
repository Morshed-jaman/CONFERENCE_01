param(
    [int]$ChunkSize = 10,
    [int]$ChunkTimeoutSeconds = 480,
    [int]$ProgressTimeoutSeconds = 240,
    [int]$CooldownSeconds = 30
)

$ErrorActionPreference = "Stop"
$project = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$conference = (Resolve-Path (Join-Path $project "..")).Path
$venvpy = Join-Path $conference "baselines\venvs\osediff\Scripts\python.exe"
$wrapper = Join-Path $PSScriptRoot "run_osediff_on_hc18.py"
$testCsv = Join-Path $project "data\processed\HC18_SR\test.csv"
$lqDir = Join-Path $project "data\processed\HC18_SR\LQ_mixed"
$outDir = Join-Path $project "results\eval\osediff\test\restored"
$logDir = Join-Path $project "results\logs"
$runLog = Join-Path $logDir "osediff_batch_run.log"
$diagLog = Join-Path $logDir "osediff_batch_diag.jsonl"
$suspectLog = Join-Path $logDir "osediff_suspect_files.txt"
$perImageLog = Join-Path $logDir "osediff_batch_perimage.csv"
$reproLog = Join-Path $logDir "osediff_reproduction_log.md"

$env:HF_HOME = "D:\hf_cache"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
New-Item -ItemType Directory -Force -Path $outDir, $logDir | Out-Null

function Write-RunLog([string]$Message) {
    $line = "$([DateTime]::Now.ToString('o')) $Message"
    Write-Host $line
    Add-Content -LiteralPath $runLog -Value $line
}

function Get-FreeDiskGiB {
    return [math]::Round(([System.IO.DriveInfo]::new("D:\")).AvailableFreeSpace / 1GB, 2)
}

function Get-FreeVramMiB {
    try {
        $values = & nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>$null
        return [int](($values | Select-Object -First 1).Trim())
    } catch {
        return 0
    }
}

function Get-LastDiag {
    if (-not (Test-Path -LiteralPath $diagLog)) { return $null }
    $line = Get-Content -LiteralPath $diagLog -Tail 1 -ErrorAction SilentlyContinue
    if (-not $line) { return $null }
    try { return $line | ConvertFrom-Json } catch { return $null }
}

function Stop-ProcessTree([int]$ProcessId) {
    & taskkill.exe /PID $ProcessId /T /F 2>$null | Out-Null
    Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
}

function Invoke-OSEDiffChunk(
    [string[]]$Names,
    [int]$ChunkNumber,
    [int]$ChopSize,
    [int]$TimeoutSeconds,
    [int]$NoProgressSeconds
) {
    $manifest = Join-Path $logDir "osediff_chunk_$ChunkNumber.txt"
    $stdoutLog = Join-Path $logDir "osediff_chunk_$ChunkNumber.out.log"
    $stderrLog = Join-Path $logDir "osediff_chunk_$ChunkNumber.err.log"
    [System.IO.File]::WriteAllLines(
        $manifest,
        $Names,
        [System.Text.UTF8Encoding]::new($false)
    )
    Set-Content -LiteralPath $stdoutLog -Value "" -Encoding UTF8
    Set-Content -LiteralPath $stderrLog -Value "" -Encoding UTF8

    $arguments = @(
        $wrapper,
        "--files", $manifest,
        "--skip_existing",
        "--out_dir", $outDir,
        "--log_file", $reproLog,
        "--per_image_log", $perImageLog,
        "--diag_log", $diagLog,
        "--chop_size", "$ChopSize",
        "--watchdog_seconds", "1200"
    )
    $started = [DateTime]::Now
    $lastProgress = $started
    $process = Start-Process -FilePath $venvpy -ArgumentList $arguments `
        -WorkingDirectory $project -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog -WindowStyle Hidden -PassThru
    Write-RunLog "chunk=$ChunkNumber pid=$($process.Id) images=$($Names.Count) chop=$ChopSize started"

    $timedOut = $false
    while (-not $process.HasExited) {
        Start-Sleep -Seconds 15
        $process.Refresh()
        if (Test-Path -LiteralPath $diagLog) {
            $diagWrite = (Get-Item -LiteralPath $diagLog).LastWriteTime
            if ($diagWrite -gt $lastProgress) { $lastProgress = $diagWrite }
        }
        $elapsed = ([DateTime]::Now - $started).TotalSeconds
        $idle = ([DateTime]::Now - $lastProgress).TotalSeconds
        if ($elapsed -ge $TimeoutSeconds -or $idle -ge $NoProgressSeconds) {
            $timedOut = $true
            $last = Get-LastDiag
            $suspect = if ($last -and $last.event -eq "image_start") { $last.filename } else { $null }
            Write-RunLog "chunk=$ChunkNumber timeout elapsed=$([math]::Round($elapsed,1))s idle=$([math]::Round($idle,1))s suspect=$suspect"
            Stop-ProcessTree $process.Id
            Start-Sleep -Seconds 30
            return [pscustomobject]@{ Success=$false; Suspect=$suspect; ExitCode=-1; Elapsed=$elapsed }
        }
    }
    $process.WaitForExit()
    $process.Refresh()
    $exitCode = $process.ExitCode
    $elapsed = ([DateTime]::Now - $started).TotalSeconds
    Write-RunLog "chunk=$ChunkNumber pid=$($process.Id) exit=$exitCode elapsed=$([math]::Round($elapsed,1))s"
    if ($exitCode -ne 0) {
        $last = Get-LastDiag
        $suspect = if ($last -and $last.event -eq "image_start") { $last.filename } else { $null }
        return [pscustomobject]@{ Success=$false; Suspect=$suspect; ExitCode=$exitCode; Elapsed=$elapsed }
    }
    return [pscustomobject]@{ Success=$true; Suspect=$null; ExitCode=0; Elapsed=$elapsed }
}

$testNames = @(
    Import-Csv -LiteralPath $testCsv |
        ForEach-Object { "$($_.image_id).png" } |
        Sort-Object
)
if ($testNames.Count -ne 150) {
    throw "Expected 150 test filenames, found $($testNames.Count)."
}
$missingLq = @($testNames | Where-Object { -not (Test-Path -LiteralPath (Join-Path $lqDir $_)) })
if ($missingLq.Count -gt 0) {
    throw "Missing LQ inputs: $($missingLq -join ', ')"
}

$queue = [System.Collections.Generic.List[string]]::new()
foreach ($name in $testNames) {
    if (-not (Test-Path -LiteralPath (Join-Path $outDir $name))) { $queue.Add($name) }
}
$suspects = [System.Collections.Generic.HashSet[string]]::new()
$chunkNumber = 0
$batchStarted = [DateTime]::Now
$lastHeartbeat = $batchStarted
Write-RunLog "resume_start completed=$($testNames.Count - $queue.Count)/150 remaining=$($queue.Count)"

while ($queue.Count -gt 0) {
    $chunkNumber++
    $take = [math]::Min($ChunkSize, $queue.Count)
    $names = @($queue.GetRange(0, $take))
    $queue.RemoveRange(0, $take)
    Write-RunLog "Chunk $chunkNumber, queued_remaining=$($queue.Count), free_vram_mib=$(Get-FreeVramMiB), free_D_gib=$(Get-FreeDiskGiB)"
    $result = Invoke-OSEDiffChunk $names $chunkNumber 256 $ChunkTimeoutSeconds $ProgressTimeoutSeconds

    if (-not $result.Success) {
        if ($result.Suspect) {
            [void]$suspects.Add($result.Suspect)
            Add-Content -LiteralPath $suspectLog -Value $result.Suspect
        }
        foreach ($name in $names) {
            if (
                -not (Test-Path -LiteralPath (Join-Path $outDir $name)) -and
                -not $suspects.Contains($name)
            ) {
                $queue.Add($name)
            }
        }
    }

    $completed = @($testNames | Where-Object { Test-Path -LiteralPath (Join-Path $outDir $_) }).Count
    Write-RunLog "chunk=$chunkNumber transition completed=$completed/150 queued=$($queue.Count) suspects=$($suspects.Count)"
    if (([DateTime]::Now - $lastHeartbeat).TotalMinutes -ge 5) {
        $elapsed = [DateTime]::Now - $batchStarted
        Write-RunLog "Elapsed: $($elapsed.ToString('m\\:ss')), completed: $completed/150, free D: $(Get-FreeDiskGiB) GiB, current chunk: $chunkNumber."
        $lastHeartbeat = [DateTime]::Now
    }
    if ($queue.Count -gt 0) { Start-Sleep -Seconds $CooldownSeconds }
}

if ($suspects.Count -gt 5) {
    Write-RunLog "HARD STOP: more than five suspects: $([string]::Join(', ', $suspects))"
    exit 2
}

$retryNumber = 1000
foreach ($suspect in @($suspects)) {
    if (Test-Path -LiteralPath (Join-Path $outDir $suspect)) { continue }
    $retryNumber++
    Write-RunLog "final_retry filename=$suspect chop=128"
    $retry = Invoke-OSEDiffChunk @($suspect) $retryNumber 128 180 180
    if ($retry.Success -and (Test-Path -LiteralPath (Join-Path $outDir $suspect))) {
        [void]$suspects.Remove($suspect)
        Write-RunLog "final_retry recovered=$suspect"
    } else {
        Write-RunLog "final_retry failed=$suspect"
    }
}

$finalCount = @($testNames | Where-Object { Test-Path -LiteralPath (Join-Path $outDir $_) }).Count
Write-RunLog "orchestrator_complete restored_count=$finalCount suspects_remaining=$($suspects.Count)"
if ($finalCount -ne 150 -or $suspects.Count -gt 0) { exit 3 }
exit 0
