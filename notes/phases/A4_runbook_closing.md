# A4 closing: runbook

This closes A4 (`notes/phases/A4_closing.md`). It checks the delivery `acee7c0` once more on your laptop, then
fast-forwards `master` to it, commits this runbook into `notes/phases/` and pushes.

- **The code is the build you accepted in stage 11** (`3944dbb`). Since then only records changed: the closing note and
  the delivery index, D129 and D130 numbered and approved, and the acceptance in `docs/a4-usability.md`. So there is no
  Unity build and no headset part.
- **Kept:** your trial branches and folders, and the stage 9 version (`a4-delivery1-checkpoint-7e5b1d3.zip`,
  `trial/a4-d1-stage9`).

**This is the delivery.** Parts 0 to 2 work on a new local branch, `a4-delivery1`; Part 3 moves `master` to it and
pushes.

| Part | Machine | What | About |
|---|---|---|---|
| 0 | Laptop, PowerShell | A clean `master` | 1 min |
| 1 | Laptop, PowerShell | Check the archive; every Python suite on `master` and on the delivery branch, compared | 10 to 25 min |
| 2 | Laptop, PowerShell | Thirteen C# builds; the perception folder; every C# check; the replays, your stage 11 scan included; A2.5's behaviour; A2.7's comparisons | 15 to 30 min |
| 3 | Laptop, PowerShell | `master` fast-forwarded to the delivery; this runbook committed; pushed | 2 min |

Save this file in the laptop's Downloads as `A4_runbook_closing.md`, beside `a4-delivery1-closing.zip`. Use one
PowerShell window. Unity can stay closed.

## Part 0 (LAPTOP): a clean master

```powershell
cd C:\Users\jubay\second-eyes
. {
    if ((git rev-parse --abbrev-ref HEAD) -ne "master") {
        git switch master
        if ($LASTEXITCODE -ne 0) { throw "STOP: git switch master (send me the output)" }
    }
    if ((git rev-parse HEAD) -ne "833ef8dbfe63795244c10ea8d4b56f5c4ce75b98") { throw "STOP: master is not at 833ef8d (send me: git log -3 --oneline)" }
    $ps = "quest-app/ProjectSettings/ProjectSettings.asset"
    if ((git status --porcelain -- $ps) -and -not (git diff -- $ps)) { git restore -- $ps; "restored ProjectSettings.asset (its line endings only had changed)" }
    if (git status --porcelain --untracked-files=no) { git status --short; throw "STOP: tracked files have uncommitted changes (send me the list above)" }
    if (-not (git ls-files tools/localize) -and (Test-Path tools\localize)) { Remove-Item -Recurse -Force tools\localize; "removed the rerun tool's leftover build files" }
    "on master at 833ef8d, clean"
}
```

## Part 1 (LAPTOP): check the checkpoint, then the Python suites on both branches

```powershell
cd C:\Users\jubay\second-eyes
$env:PATH = "C:\Android\platform-tools;$env:PATH"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force; .\.venv\Scripts\Activate.ps1
. {
    $D = "C:\Users\jubay\second-eyes-data"; $W = "$D\a4-delivery1-closing"
    $Z = "$HOME\Downloads\a4-delivery1-closing.zip"
    if (-not (Test-Path "$HOME\Downloads\A4_runbook_closing.md")) { throw "STOP: save this runbook in Downloads as A4_runbook_closing.md" }
    if (-not (Test-Path $Z)) { throw "STOP: save the checkpoint zip in Downloads" }
    if ((Get-FileHash $Z -Algorithm SHA256).Hash.ToLower() -ne "8618e2e91f5ee3f63dcbb3ba4781ea7ea42bbc47edbb50a68c6465221533ea04") { throw "STOP: not the checkpoint I sent" }
    if ((git rev-parse --abbrev-ref HEAD) -ne "master") { throw "STOP: run Part 0 first" }
    if ((git rev-parse HEAD) -ne "833ef8dbfe63795244c10ea8d4b56f5c4ce75b98") { throw "STOP: master is not at 833ef8d (send me: git log -3 --oneline)" }
    if (git status --porcelain --untracked-files=no) { throw "STOP: tracked files have uncommitted changes (run Part 0)" }
    if (git branch --list "a4-delivery1") { throw "STOP: a4-delivery1 already exists (send me: git log -3 --oneline a4-delivery1)" }
    if (Test-Path $W) { throw "STOP: $W already exists; rename it first" }
    New-Item -ItemType Directory -Force "$W\checkpoint" | Out-Null
    Expand-Archive $Z -DestinationPath "$W\checkpoint"
    $bad = 0
    foreach ($line in Get-Content "$W\checkpoint\INDEX.sha256") {
        $hash, $name = $line -split "  ", 2
        if ((Get-FileHash "$W\checkpoint\$name" -Algorithm SHA256).Hash.ToLower() -ne $hash) { $bad++; "mismatch: $name" }
    }
    if ($bad) { throw "STOP: $bad file(s) in the checkpoint differ from INDEX.sha256" }
    "checkpoint: every file matches its index"
    git bundle verify "$W\checkpoint\a4-delivery1.bundle"
    if ($LASTEXITCODE -ne 0) { throw "STOP: the bundle doesn't verify against this repository (send me the output)" }

    function Run-Suites([string]$tag) {
        $out = "$W\py-$tag"; New-Item -ItemType Directory -Force $out | Out-Null
        foreach ($t in Get-ChildItem grounding\tests\test_*.py | Sort-Object Name) {
            $file = "$out\$($t.BaseName).txt"
            for ($attempt = 1; $attempt -le 6; $attempt++) {
                python $t.FullName *>&1 | Out-File -Encoding utf8 $file
                $code = $LASTEXITCODE
                $flake = (Select-String -Path $file -SimpleMatch "WinError" -Quiet) -or ((Get-Content $file -Tail 1) -match 'was published$')
                if (-not $flake) { break }
                Add-Content "$out\retries.txt" ("{0}: attempt {1} met the Windows file-lock flake" -f $t.BaseName, $attempt)
                Start-Sleep -Seconds 10
            }
            Add-Content "$out\exit-codes.txt" ("{0} {1}" -f $t.BaseName, $code)
        }
        $retried = if (Test-Path "$out\retries.txt") { @(Get-Content "$out\retries.txt").Count } else { 0 }
        "{0}: {1} suites run, {2} rerun after the flake" -f $tag, (Get-Content "$out\exit-codes.txt").Count, $retried
    }
    Run-Suites "master"
    git fetch "$W\checkpoint\a4-delivery1.bundle" "a4-delivery1:a4-delivery1"
    if ($LASTEXITCODE -ne 0) { throw "STOP: fetching the bundle into the delivery branch (send me the output)" }
    git switch "a4-delivery1"
    if ($LASTEXITCODE -ne 0) { throw "STOP: git switch to the delivery branch" }
    if ((git rev-parse HEAD) -ne "acee7c0daab2b8dc51355a5662bb51d3489e076c") { throw "STOP: the delivery branch is not at acee7c0" }
    Run-Suites "trial"

    $R = "$W\python-compare.txt"; Set-Content $R "Every Python suite: master (833ef8d) against the delivery branch (acee7c0)"
    $cm = @{}; foreach ($l in Get-Content "$W\py-master\exit-codes.txt") { $n, $c = $l -split " "; $cm[$n] = $c }
    $ct = @{}; foreach ($l in Get-Content "$W\py-trial\exit-codes.txt") { $n, $c = $l -split " "; $ct[$n] = $c }
    $differ = 0
    foreach ($n in $cm.Keys | Sort-Object) {
        $fm = (Select-String -Path "$W\py-master\$n.txt" -Pattern "^(  )?FAIL|FAILED:" -CaseSensitive | ForEach-Object { $_.Line } | Sort-Object -Unique) -join "`n"
        $ft = (Select-String -Path "$W\py-trial\$n.txt" -Pattern "^(  )?FAIL|FAILED:" -CaseSensitive | ForEach-Object { $_.Line } | Sort-Object -Unique) -join "`n"
        $same = ($cm[$n] -eq $ct[$n]) -and ($fm -eq $ft)
        $note = "same"
        $mf = "$W\py-master\$n.txt"
        if (-not $same -and ((Select-String -Path $mf -SimpleMatch "WinError" -Quiet) -or ((Get-Content $mf -Tail 1) -match 'was published$'))) {
            foreach ($dir in "a4-trial-stage11", "a4-trial-stage10", "a4-trial-stage9", "a4-trial-stage8", "a4-trial-stage7", "a4-trial-stage6", "a4-trial-stage5", "a4-trial-stage4", "a4-trial-stage3c", "a4-trial-stage3b", "a4-trial-stage3", "a4-trial-stage2b", "a4-trial-stage2") {
                $m2 = "$D\$dir\py-master\$n.txt"
                if ((Test-Path $m2) -and -not (Select-String -Path $m2 -SimpleMatch "WinError" -Quiet) -and ((Get-Content $m2 -Tail 1) -notmatch 'was published$')) {
                    $f2 = (Select-String -Path $m2 -Pattern "^(  )?FAIL|FAILED:" -CaseSensitive | ForEach-Object { $_.Line } | Sort-Object -Unique) -join "`n"
                    if ($f2 -eq $ft) { $same = $true; $note = "same as the clean master run in $dir (this master run met the flake)" }
                    break
                }
            }
        }
        if (-not $same) {   # the trial's run may be the odd one: rerun it, up to three times, 10 seconds apart
            for ($k = 1; $k -le 3 -and -not $same; $k++) {
                Start-Sleep -Seconds 10
                $rr = "$W\py-trial\$n.rerun$k.txt"
                python "grounding\tests\$n.py" *>&1 | Out-File -Encoding utf8 $rr
                $rc = "$LASTEXITCODE"
                $fr = (Select-String -Path $rr -Pattern "^(  )?FAIL|FAILED:" -CaseSensitive | ForEach-Object { $_.Line } | Sort-Object -Unique) -join "`n"
                if ($rc -eq $cm[$n] -and $fr -eq $fm) { $same = $true; $note = "same on rerun $k (the trial's first run was the odd one)" }
            }
        }
        if (-not $same) { $differ++ }
        Add-Content $R ("{0,-32} master exit {1}, trial exit {2}: {3}" -f $n, $cm[$n], $ct[$n], $(if ($same) { $note } else { "DIFFERS" }))
    }
    foreach ($tag in "master", "trial") {
        if (Test-Path "$W\py-$tag\retries.txt") { foreach ($l in Get-Content "$W\py-$tag\retries.txt") { Add-Content $R ("rerun on {0}: {1}" -f $tag, $l) } }
    }
    $newOk = $true
    foreach ($pair in @(@("test_contract_v3", "^21 passed, 0 failed"), @("test_resolution_v2", "^26 passed, 0 failed"),
                        @("test_perception_folder", "^16 passed, 0 failed"), @("test_scan_device", "^21 passed, 0 failed"))) {
        $last = Get-Content "$W\py-trial\$($pair[0]).txt" -Tail 1
        Add-Content $R ("{0} (new, trial only): exit {1}, {2}" -f $pair[0], $ct[$pair[0]], $last)
        if ($ct[$pair[0]] -ne "0" -or $last -notmatch $pair[1]) { $newOk = $false }
    }
    Get-Content $R
    if ($differ) { throw "STOP: $differ suite(s) behave differently on the delivery branch (see $R); send me $R and the folder $W\py-trial" }
    if (-not $newOk) { throw "STOP: a new suite (see the last four lines of $R and its file in $W\py-trial)" }
}
```

**Expect** `master: 32 suites run, ...`, then `trial: 36 suites run, ...` (reruns after the flake are fine), and the
report: 32 lines ending `same` (or `same as the clean master run in ...` where this run's `master` met the flake), any `rerun on ...` lines, then the four new suites, each `exit 0`:
- `test_contract_v3`: 21 passed, 0 failed;
- `test_resolution_v2`: 26 passed, 0 failed;
- `test_perception_folder`: 16 passed, 0 failed;
- `test_scan_device`: 21 passed, 0 failed.

## Part 2 (LAPTOP): the C# projects, the perception folder, the checks and the PC rerun

Same window, on the delivery branch. It now also stops if your perception folder differs from the one I built: with Git's
line endings in the folder, the same commit must give the same folder on every machine.

```powershell
. {
    if (-not (Get-Command dotnet -ErrorAction SilentlyContinue)) { throw "STOP: the .NET 8 SDK isn't on PATH" }
    if ((git rev-parse HEAD) -ne "acee7c0daab2b8dc51355a5662bb51d3489e076c") { throw "STOP: run Part 1 first (HEAD is not the delivery branch)" }
    $D = "C:\Users\jubay\second-eyes-data"; $W = "$D\a4-delivery1-closing"; $N = "$W\dotnet.txt"
    Set-Content $N "A4 trial (acee7c0): the C# projects, the checks and the PC rerun"
    foreach ($pr in "tools\csharp-tests\UnityApi", "tools\csharp-tests\UnityGlue", "tools\csharp-tests\PanelPartial", "tools\csharp-tests\Behavior",
                    "tools\csharp-tests\Resolution", "tools\csharp-tests\Frames", "tools\csharp-tests\Contract", "tools\csharp-tests\Localization",
                    "tools\csharp-tests\Scan", "tools\csharp-tests\ScanGlue", "tools\csharp-tests\FrameSourceGlue", "tools\csharp-tests\AutoScan", "tools\localize") {
        $out = dotnet build $pr -nologo -v q 2>&1
        Add-Content $N ("== build " + $pr); $out | Add-Content $N
        if (-not ($out | Select-String " 0 Error\(s\)" -Quiet)) { throw "STOP: $pr doesn't build (see $N)" }
        "{0,-36} builds" -f $pr
    }
    $PF = "$W\perception-folder"
    $pb = python -m grounding.quest.perception_device build --out $PF
    if ($LASTEXITCODE -ne 0) { throw "STOP: building the perception folder (send me the output)" }
    $pb
    if (-not ($pb -match "3a4ce1a888fe59a0ce9a2a51ece03cab9d3b7527fd9f956b3982debcc6067fb5")) { throw "STOP: your perception folder differs from mine (send me the line above)" }
    $f = dotnet run --no-build --project tools\csharp-tests\Frames 2>&1
    Add-Content $N "== Frames"; $f | Add-Content $N; "Frames:       " + ($f | Select-Object -Last 1)
    if (-not ($f | Select-String "^40 passed, 0 failed" -Quiet)) { throw "STOP: the frame checks (see $N)" }
    $c = dotnet run --no-build --project tools\csharp-tests\Contract -- grounding\tests\fixtures schemas $PF 2>&1
    Add-Content $N "== Contract"; $c | Add-Content $N; "Contract:     " + ($c | Select-Object -Last 1)
    if (-not ($c | Select-String "^129 passed, 0 failed, 1 known difference" -Quiet)) { throw "STOP: the contract checks (see $N)" }
    $lz = dotnet run --no-build --project tools\csharp-tests\Localization -- grounding\tests\fixtures schemas 2>&1
    Add-Content $N "== Localization"; $lz | Add-Content $N; "Localization: " + ($lz | Select-Object -Last 1)
    if (-not ($lz | Select-String "^76 passed, 0 failed" -Quiet)) { throw "STOP: the localization and scan session checks (see $N)" }
    $sc = dotnet run --no-build --project tools\csharp-tests\Scan -- schemas 2>&1
    Add-Content $N "== Scan"; $sc | Add-Content $N; "Scan:         " + ($sc | Select-Object -Last 1)
    if (-not ($sc | Select-String "^43 passed, 0 failed" -Quiet)) { throw "STOP: the scan rule, camera model and guidance checks (see $N)" }
    $sg = dotnet run --no-build --project tools\csharp-tests\ScanGlue 2>&1
    Add-Content $N "== ScanGlue"; $sg | Add-Content $N; "ScanGlue:     " + ($sg | Select-Object -Last 1)
    if (-not ($sg | Select-String "^21 passed, 0 failed" -Quiet)) { throw "STOP: the scanner's stubs against the real classes (see $N)" }
    $fsg = dotnet run --no-build --project tools\csharp-tests\FrameSourceGlue 2>&1
    Add-Content $N "== FrameSourceGlue"; $fsg | Add-Content $N; "FrameSourceGlue: " + ($fsg | Select-Object -Last 1)
    if (-not ($fsg | Select-String "^1 passed, 0 failed" -Quiet)) { throw "STOP: the frame source, compiled against its stubs (see $N)" }
    $AS = "$W\autoscan-sample"
    $asr = dotnet run --no-build --project tools\csharp-tests\AutoScan -- schemas $AS 2>&1
    Add-Content $N "== AutoScan"; $asr | Add-Content $N; "AutoScan:     " + ($asr | Select-Object -Last 1)
    if (-not ($asr | Select-String "^106 passed, 0 failed" -Quiet)) { throw "STOP: the automatic scan's checks (see $N)" }
    $rp = dotnet run --no-build --project tools\localize -- --memory "$AS\frames.jsonl" "perception\bedroom\category-map.bedroom.v2.json" "perception\bedroom\size-priors.bedroom.v1.json" `
          "perception\autoscan\autoscan.default.v1.json" "$W\autoscan-replay" schemas --compare "$AS\scene.json" 2>&1
    $rpCode = $LASTEXITCODE
    Add-Content $N "== the automatic scan's replay on the depth-only sample"; $rp | Add-Content $N
    if ($rpCode -ne 0) { throw "STOP: the automatic scan's replay, depth only (see $N)" }
    "Replay:       " + ($rp | Select-Object -Last 1)
    $rm = dotnet run --no-build --project tools\localize -- --memory "$AS\room\frames.jsonl" "perception\bedroom\category-map.bedroom.v3.json" "perception\bedroom\size-priors.bedroom.v1.json" `
          "perception\autoscan\autoscan.default.v2.json" "$W\autoscan-replay-room" schemas --compare "$AS\room\scene.json" 2>&1
    $rmCode = $LASTEXITCODE
    Add-Content $N "== the automatic scan's replay on the sample with Meta's room"; $rm | Add-Content $N
    if ($rmCode -ne 0) { throw "STOP: the automatic scan's replay with Meta's room (see $N)" }
    "Replay, room: " + ($rm | Select-Object -Last 1)
    $rb = dotnet run --no-build --project tools\localize -- --memory "$AS\boxes\frames.jsonl" "perception\bedroom\category-map.bedroom.v4.json" "perception\bedroom\size-priors.bedroom.v1.json" `
          "perception\autoscan\autoscan.default.v3.json" "$W\autoscan-replay-boxes" schemas --compare "$AS\boxes\scene.json" 2>&1
    $rbCode = $LASTEXITCODE
    Add-Content $N "== the automatic scan's replay on the sample with 3D boxes"; $rb | Add-Content $N
    if ($rbCode -ne 0) { throw "STOP: the automatic scan's replay with 3D boxes (see $N)" }
    "Replay, boxes: " + ($rb | Select-Object -Last 1)
    $rs = dotnet run --no-build --project tools\localize -- --memory "$AS\support\frames.jsonl" "perception\bedroom\category-map.bedroom.v4.json" "perception\bedroom\size-priors.bedroom.v1.json" `
          "perception\autoscan\autoscan.default.v4.json" "$W\autoscan-replay-support" schemas --compare "$AS\support\scene.json" 2>&1
    $rsCode = $LASTEXITCODE
    Add-Content $N "== the automatic scan's replay on the sample with v4"; $rs | Add-Content $N
    if ($rsCode -ne 0) { throw "STOP: the automatic scan's replay with v4 (see $N)" }
    "Replay, v4:   " + ($rs | Select-Object -Last 1)
    $rh = dotnet run --no-build --project tools\localize -- --memory "$AS\mesh\frames.jsonl" "perception\bedroom\category-map.bedroom.v4.json" "perception\bedroom\size-priors.bedroom.v1.json" `
          "perception\autoscan\autoscan.default.v5.json" "$W\autoscan-replay-mesh" schemas --compare "$AS\mesh\scene.json" 2>&1
    $rhCode = $LASTEXITCODE
    Add-Content $N "== the automatic scan's replay on the sample with Meta's mesh"; $rh | Add-Content $N
    if ($rhCode -ne 0) { throw "STOP: the automatic scan's replay with Meta's mesh (see $N)" }
    "Replay, mesh: " + ($rh | Select-Object -Last 1)
    $rd = dotnet run --no-build --project tools\localize -- --memory "$AS\detected\frames.jsonl" "$PF\maps\category-map.bedroom.v5.json" "perception\bedroom\size-priors.bedroom.v1.json" `
          "perception\autoscan\autoscan.default.v6.json" "$W\autoscan-replay-detected" schemas --compare "$AS\detected\scene.json" 2>&1
    $rdCode = $LASTEXITCODE
    Add-Content $N "== the automatic scan's replay on the sample with v6"; $rd | Add-Content $N
    if ($rdCode -ne 0) { throw "STOP: the automatic scan's replay with v6 (see $N)" }
    "Replay, v6:   " + ($rd | Select-Object -Last 1)
    $rv = dotnet run --no-build --project tools\localize -- --memory "$AS\corrections\frames.jsonl" "$PF\maps\category-map.bedroom.v5.json" "perception\bedroom\size-priors.bedroom.v1.json" `
          "perception\autoscan\autoscan.default.v7.json" "$W\autoscan-replay-corrections" schemas --compare "$AS\corrections\scene.json" 2>&1
    $rvCode = $LASTEXITCODE
    Add-Content $N "== the automatic scan's replay on the sample with v7"; $rv | Add-Content $N
    if ($rvCode -ne 0) { throw "STOP: the automatic scan's replay with v7 (see $N)" }
    "Replay, v7:   " + ($rv | Select-Object -Last 1)
    $rw = dotnet run --no-build --project tools\localize -- --memory "$AS\worldlock\frames.jsonl" "$PF\maps\category-map.bedroom.v5.json" "$PF\priors\size-priors.bedroom.v2.json" `
          "$PF\configs\autoscan.default.v8.json" "$W\autoscan-replay-worldlock" schemas --compare "$AS\worldlock\scene.json" 2>&1
    $rwCode = $LASTEXITCODE
    Add-Content $N "== the automatic scan's replay on the sample with v8"; $rw | Add-Content $N
    if ($rwCode -ne 0) { throw "STOP: the automatic scan's replay with v8 (see $N)" }
    "Replay, v8:   " + ($rw | Select-Object -Last 1)

    $IN = @("grounding\tests\fixtures\capture\valid\capture.bedroom.json", "grounding\tests\fixtures\contract\valid\category-map.bedroom.json",
            "grounding\tests\fixtures\localization\size-priors.test.json", "perception\localization\localization.default.v2.json")
    $LO = "$W\localize-fixture"
    $r1 = dotnet run --no-build --project tools\localize -- @IN "$LO\run1" schemas 2>&1
    $code1 = $LASTEXITCODE
    $r2 = dotnet run --no-build --project tools\localize -- @IN "$LO\run2" schemas --compare "$LO\run1\scene.json" 2>&1
    $code2 = $LASTEXITCODE
    Add-Content $N "== the PC rerun on the capture fixture"; $r1 | Add-Content $N; $r2 | Add-Content $N
    if ($code1 -ne 0 -or $code2 -ne 0) { throw "STOP: the PC rerun (see $N)" }
    if ((Get-FileHash "$LO\run1\scene.json").Hash -ne (Get-FileHash "$LO\run2\scene.json").Hash) { throw "STOP: two reruns wrote different scenes (send me the folder $LO)" }
    "PC rerun: two runs wrote the same scene, and the comparison agrees"

    $GOLD25 = "$D\a25-goldens\goldens-20261009-053729\headset"
    if (-not (Test-Path $GOLD25)) { throw "STOP: $GOLD25 is missing" }
    $b = dotnet run --no-build --project tools\csharp-tests\Behavior -- $GOLD25 2>&1
    Add-Content $N "== A2.5's behaviour checks"; $b | Add-Content $N; "Behaviour:    " + ($b | Select-Object -Last 1)
    if (-not ($b | Select-String "passed, 0 failed" -Quiet)) { throw "STOP: A2.5's behaviour checks (see $N)" }

    $S = "$D\iref-release-samples"; $O = "$D\iref-release\official"; $T = "$D\tmp"
    New-Item -ItemType Directory -Force $T | Out-Null; $env:TEMP = $T; $env:TMP = $T
    $LIB = "native\out\host\se_llama.dll"; $TINY = "$T\a27-tiny.gguf"
    foreach ($p in $S, $O) { if (-not (Test-Path $p)) { throw "STOP: $p is missing (send me: dir $D)" } }
    if (-not (Test-Path $LIB)) { throw "STOP: $LIB is missing: in a Developer PowerShell for VS 2022, run python tools\build_llama.py host, then run Part 2 again" }
    if (-not (Test-Path $TINY)) { python native\tests\make_tiny_model.py third_party\llama.cpp $TINY; if ($LASTEXITCODE -ne 0) { throw "STOP: the tiny test model" } }
    $G = "$W\a27-goldens"
    python -m grounding.a27 goldens --samples $S --official $O --out $G --lib $LIB --gguf $TINY
    if ($LASTEXITCODE -ne 0) { throw "STOP: the A2.7 goldens (send me the output)" }
    $env:PATH = "$PWD\native\out\host;$env:PATH"
    $REL = "grounding\relations\relations.v1.json"; $DIR = "grounding\relations\directions.v1.json"; $LANG = "grounding\a24\language.v1.json"
    $modes = @(@("relations", "$G\relations.json", $REL), @("schemas", "$G\schemas.json", "schemas"), @("validation", "$G\validation.json", "schemas"),
               @("resolution", "$G\resolution.json", "schemas", $REL, $DIR), @("reader", "$G\reader.json", $LANG),
               @("pipeline", "$G\pipeline.json", "schemas", $REL, $DIR, $LANG, "grounding\a24\settings.v1.json"), @("native", "$G\native.json", $TINY))
    foreach ($m in $modes) {
        $out = dotnet run --no-build --project tools\csharp-tests\Resolution -- @m 2>&1
        Add-Content $N ("== " + $m[0]); $out | Add-Content $N
        $last = ($out | Select-String "differ" | Select-Object -Last 1)
        "{0,-11} {1}" -f $m[0], $last
        if (-not ($out | Select-String " 0 differ" -Quiet)) { throw "STOP: the $($m[0]) comparison (see $N)" }
    }

    $S11 = "$D\a4-trial-stage11\scans\scan.d421455f.1"   # the scan you accepted, rebuilt once more from its log
    if (-not (Test-Path "$S11\frames.jsonl")) { throw "STOP: $S11 is missing (send me: dir $D\a4-trial-stage11\scans)" }
    $r11 = dotnet run --no-build --project tools\localize -- --memory "$S11\frames.jsonl" "$PF\maps\category-map.bedroom.v5.json" "$PF\priors\size-priors.bedroom.v2.json" `
           "$PF\configs\autoscan.default.v8.json" "$W\replay-stage11" schemas --compare "$S11\scene.json" 2>&1
    $r11Code = $LASTEXITCODE
    Add-Content $N "== your stage 11 scan"; $r11 | Add-Content $N
    if ($r11Code -ne 0) { throw "STOP: your stage 11 scan's replay (see $N)" }
    "Your stage 11 scan: " + ($r11 | Select-Object -Last 1)
}
```

**Expect:**
- thirteen `builds` lines;
- `built ...\perception-folder: 22 files, 153793 bytes; manifest SHA-256 3a4ce1a8...`;
- the checks:

  | Check | Expected |
  |---|---|
  | `Frames` | 40 passed, 0 failed |
  | `Contract` | 129 passed, 0 failed, 1 known difference(s) reported |
  | `Localization` | 76 passed, 0 failed |
  | `Scan` | 43 passed, 0 failed |
  | `ScanGlue` | 21 passed, 0 failed |
  | `FrameSourceGlue` | 1 passed, 0 failed |
  | `AutoScan` | 106 passed, 0 failed |
  | `Replay` | compared with ...: the complete scene is the same, byte for byte once canonical (4 objects; every field, sizes, rotations and evidence included) |
  | `Replay, room` | compared with ...: the complete scene is the same, byte for byte once canonical (7 objects; every field, sizes, rotations and evidence included) |
  | `Replay, boxes` | compared with ...: the complete scene is the same, byte for byte once canonical (3 objects; every field, sizes, rotations and evidence included) |
  | `Replay, v4` | compared with ...: the complete scene is the same, byte for byte once canonical (4 objects; every field, sizes, rotations and evidence included) |
  | `Replay, mesh` | compared with ...: the complete scene is the same, byte for byte once canonical (5 objects; every field, sizes, rotations and evidence included) |
  | `Replay, v6` | compared with ...: the complete scene is the same, byte for byte once canonical (2 objects; every field, sizes, rotations and evidence included) |
  | `Replay, v7` | compared with ...: the complete scene is the same, byte for byte once canonical (5 objects; every field, sizes, rotations and evidence included) |
  | `Replay, v8` | compared with ...: the complete scene is the same, byte for byte once canonical (2 objects; every field, sizes, rotations and evidence included) |

- `PC rerun: two runs wrote the same scene, and the comparison agrees`;
- `Behaviour:    64 passed, 0 failed`;
- seven A2.7 mode lines, every one ending `0 differ`.
- `Your stage 11 scan: compared with ...: the complete scene is the same, byte for byte once canonical (11 objects; ...)`.

## Part 3 (LAPTOP): master to the delivery, this runbook committed, pushed

Only after Parts 1 and 2 ended without a STOP.

```powershell
. {
    $D = "C:\Users\jubay\second-eyes-data"; $W = "$D\a4-delivery1-closing"
    if ((git rev-parse HEAD) -ne "acee7c0daab2b8dc51355a5662bb51d3489e076c") { throw "STOP: run Parts 1 and 2 first (HEAD is not the delivery branch at acee7c0)" }
    foreach ($f in "$W\python-compare.txt", "$W\dotnet.txt") { if (-not (Test-Path $f)) { throw "STOP: $f is missing: run Parts 1 and 2 first" } }
    if (-not (Select-String -Path "$W\dotnet.txt" -SimpleMatch "== your stage 11 scan" -Quiet)) { throw "STOP: Part 2 didn't reach its end (see $W\dotnet.txt)" }
    $ps = "quest-app/ProjectSettings/ProjectSettings.asset"
    if ((git status --porcelain -- $ps) -and -not (git diff -- $ps)) { git restore -- $ps }
    if (git status --porcelain --untracked-files=no) { git status --short; throw "STOP: tracked files have changed (send me the list above)" }
    git switch master
    if ($LASTEXITCODE -ne 0) { throw "STOP: git switch master (send me the output)" }
    if ((git rev-parse HEAD) -ne "833ef8dbfe63795244c10ea8d4b56f5c4ce75b98") { throw "STOP: master moved (send me: git log -3 --oneline)" }
    git merge --ff-only a4-delivery1
    if ($LASTEXITCODE -ne 0) { throw "STOP: the fast-forward (send me the output)" }
    if ((git rev-parse HEAD) -ne "acee7c0daab2b8dc51355a5662bb51d3489e076c") { throw "STOP: master isn't at acee7c0 after the fast-forward" }
    Copy-Item "$HOME\Downloads\A4_runbook_closing.md" "notes\phases\A4_runbook_closing.md"
    git add -- notes/phases/A4_runbook_closing.md
    if ($LASTEXITCODE -ne 0) { throw "STOP: git add" }
    git commit -m "A4 closing: the closing runbook (a4-delivery1-closing.zip, SHA-256 8618e2e91f5ee3f63dcbb3ba4781ea7ea42bbc47edbb50a68c6465221533ea04)"
    if ($LASTEXITCODE -ne 0) { throw "STOP: the commit" }
    if (git status --porcelain --untracked-files=no) { git status --short; throw "STOP: something is left uncommitted (send me the list above)" }
    git push origin master
    if ($LASTEXITCODE -ne 0) { throw "STOP: the push (send me the output)" }
    git log --oneline -3
    "A4 is closed: master at " + (git rev-parse --short HEAD) + ", pushed to origin"
}
```

**Send me:**
- `python-compare.txt` and `dotnet.txt` from `C:\Users\jubay\second-eyes-data\a4-delivery1-closing`;
- everything Part 3 printed: the last three commits and the closing line;
- if any block stopped: its STOP line and the file it names.

Your trial branches stay local; nothing here deletes them.
