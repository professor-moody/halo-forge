# Get detailed compile errors for failing problems
param([string]$DatasetFile = "windows_curriculum_rlvr.jsonl")

. $env:USERPROFILE\init-msvc.ps1

$problems = Get-Content $DatasetFile | ForEach-Object { ConvertFrom-Json $_ }

Write-Host "Testing $($problems.Count) problems..."

foreach ($p in $problems) {
    $src = "test_compile.cpp"
    $exe = "test_compile.exe"
    
    Set-Content -Path $src -Value $p.solution -Encoding UTF8
    
    $output = cl /nologo /EHsc /W3 /Fe:$exe $src 2>&1 | Out-String
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "=== $($p.id) ===" -ForegroundColor Red
        $lines = $output -split "`n" | Where-Object { $_ -match "error" } | Select-Object -First 3
        foreach ($line in $lines) {
            Write-Host $line
        }
    }
    
    Remove-Item $src -ErrorAction SilentlyContinue
    Remove-Item $exe -ErrorAction SilentlyContinue
    Remove-Item "test_compile.obj" -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "Done"

