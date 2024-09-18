# Find all .py files in the docs directory as this script and convert them to .ipynb
# This excludes the deployment directory

param (
    [Parameter(Position=0,mandatory=$true)]
    [string]$runId,

    [Parameter(Position=1,mandatory=$false)]
    [string]$kernelName = "pyrit-kernel"
)

$scriptDir = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
$docDir = Split-Path -Parent -Path $scriptDir

# Define the cache file path using the given ID
$cacheFile = "$docDir\generate_docs\cache\$runId"
if (-not (Test-Path $cacheFile)) {
    New-Item -ItemType File -Path $cacheFile | Out-Null
}

# Load processed files into a hash set for quick lookup
$processedFiles = @{}
if (Test-Path $cacheFile) {
    Get-Content $cacheFile | ForEach-Object { $processedFiles[$_] = $true }
}

# Find all .py files excluding the deployment directory
$files = Get-ChildItem -Path $docDir -Recurse -Include *.py -Exclude *_helpers.py -File |
         Where-Object { -not $_.FullName.ToLower().Contains("\deployment\") } |
         Where-Object { -not $_.FullName.ToLower().Contains("\generate_docs\") }


foreach ($file in $files) {
    if ($processedFiles.ContainsKey($file.FullName)) {
        Write-Host "Skipping already processed file: $file"
        continue
    }

    Write-Host "Processing $file"

    $stderr = $null
    $result = jupytext --execute --set-kernel $kernelName --to notebook $file.FullName 2>&1

    $stderr = $result | Where-Object {
        $_ -is [System.Management.Automation.ErrorRecord] -and
        $_.ToString() -notmatch "RuntimeWarning" -and
        $_.ToString() -notmatch "self._get_loop()"
    }


    if ($stderr) {
        Write-Host "Error processing $file"
        $stderr | ForEach-Object {
             Write-Host "$_" -ForegroundColor Red
            }
    } else {
        Write-Host "Successfully processed $file"
        # Log to cache file
        $file.FullName | Out-File -Append -FilePath $cacheFile
    }
}
