# Find all .ipynb files in the docs directory as this script and convert them to .py
# This excludes the deployment directory


$scriptDir = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
$docDir = Split-Path -Parent -Path $scriptDir

# Find all .ipynb files excluding the deployment directory
$files = Get-ChildItem -Path $docDir -Recurse -Include *.ipynb -File |
         Where-Object { -not $_.FullName.ToLower().Contains("\deployment\") } |
         Where-Object { -not $_.FullName.ToLower().Contains("\generate_docs\") }


foreach ($file in $files) {

    Write-Host "Processing $file"

    $stderr = $null
    $result = jupytext --to py:percent $file.FullName 2>&1

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
    }
}
