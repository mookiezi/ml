$folder = "D:\ml"
$filter = "*.*"
$debounceSeconds = 10
$lastChange = Get-Date

$fsw = New-Object IO.FileSystemWatcher $folder, $filter
$fsw.IncludeSubdirectories = $true
$fsw.EnableRaisingEvents = $true

Register-ObjectEvent $fsw Changed -SourceIdentifier FileChanged -Action {
    $global:lastChange = Get-Date
}

Write-Host "Watching for changes in $folder..."
while ($true) {
    Start-Sleep -Seconds 5
    $elapsed = (Get-Date) - $global:lastChange
    if ($elapsed.TotalSeconds -ge $debounceSeconds) {
        $global:lastChange = (Get-Date).AddYears(10) # prevent immediate rerun
        try {
            cd $folder
            git add .
            git commit -m "Auto commit $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
            git push origin master
            Write-Host "Auto-pushed at $(Get-Date)"
        } catch {
            Write-Host "Git push failed: $_"
        }
    }
}