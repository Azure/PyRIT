$currDir = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
$files = Get-ChildItem -Path $currDir -Recurse -Include *.py -File|
Where-Object { -not $_.FullName.ToLower().Contains("\deployment\") }

foreach ($file in $files) {
    write-host "Processing $file"
    jupytext --execute --to notebook $file.FullName
}
