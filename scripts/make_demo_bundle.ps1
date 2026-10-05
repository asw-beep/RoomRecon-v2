# Package a walkable room as a self-contained backup demo: viewer build + room + local server.
# Usage: .\scripts\make_demo_bundle.ps1 -SceneDir E:\roomrecon-scenes\vr_room\dns-30k-lite `
#            -OutDir E:\roomrecon-scenes\demo-vr_room [-ViewOnlyHtml]
# Needs: collision_room.ps1 already run in SceneDir; the patched viewer built (npm run build).
# Writes OutDir and OutDir.zip; refuses to touch an existing OutDir.
param(
    [Parameter(Mandatory)][string]$SceneDir,
    [Parameter(Mandatory)][string]$OutDir,
    [string]$ViewerDir = "E:\roomrecon-scenes\supersplat-viewer",
    [switch]$ViewOnlyHtml
)
$ErrorActionPreference = "Stop"
if ((Test-Path $OutDir) -or (Test-Path "$OutDir.zip")) { throw "$OutDir or its .zip exists; pick a new -OutDir" }
$demo = Join-Path (Split-Path $PSScriptRoot -Parent) "demo"
New-Item -ItemType Directory "$OutDir\room" | Out-Null

Get-ChildItem "$ViewerDir\public" -File | Where-Object { $_.Extension -ne ".map" } | Copy-Item -Destination $OutDir
Copy-Item "$ViewerDir\LICENSE" "$OutDir\LICENSE-supersplat-viewer.txt"
foreach ($f in "aligned.ply", "room.collision.glb", "settings.json", "scene.json") {
    Copy-Item "$SceneDir\$f" "$OutDir\room\"
}
Copy-Item "$demo\serve.py" $OutDir
# CRLF line endings for cmd.exe and Notepad
foreach ($f in "start-demo.cmd", "README.txt") { Get-Content "$demo\$f" | Set-Content -Encoding ascii "$OutDir\$f" }

if ($ViewOnlyHtml) {
    # upstream viewer with the room embedded (SOG-compressed): opens by double-click, no collision
    npx -y @playcanvas/splat-transform -w "$SceneDir\aligned.ply" --viewer-settings "$SceneDir\settings.json" `
        "$OutDir\room-view-only.html"
}
Compress-Archive -Path "$OutDir\*" -DestinationPath "$OutDir.zip"
Get-ChildItem $OutDir -Recurse -File | Select-Object @{n = "File"; e = { $_.FullName.Substring($OutDir.Length + 1) } }, Length
Get-Item "$OutDir.zip" | Select-Object FullName, Length
