# Package a walkable room as a self-contained backup demo: viewer build + room + local server.
# Usage: .\scripts\make_demo_bundle.ps1 -SceneDir E:\roomrecon-scenes\vr_room\dns-30k-lite `
#            -OutDir E:\roomrecon-scenes\demo-vr_room [-ViewOnlyHtml] -Data @"
#   Room: "vr_room" from the MuSHRoom dataset (Ren et al.), CC BY 4.0.
#   Trained on the GPU machine, run dns-30k-lite (2026-10-05); see results/ in the repo.
# "@
# -Data is the README's DATA section (indented two spaces); the Gaussian count and file size
# come from the room's aligned.ply.
# Needs: collision_room.ps1 already run in SceneDir; the patched viewer built (npm run build).
# Writes OutDir and OutDir.zip; refuses to touch an existing OutDir.
param(
    [Parameter(Mandatory)][string]$SceneDir,
    [Parameter(Mandatory)][string]$OutDir,
    [Parameter(Mandatory)][string]$Data,
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
# README facts about this room: Gaussian count from the PLY header, size, the DATA section
$ply = "$SceneDir\aligned.ply"
$fs = [IO.File]::OpenRead($ply)
try { $buf = New-Object byte[] 4096; $n = $fs.Read($buf, 0, $buf.Length) } finally { $fs.Close() }
$header = [Text.Encoding]::ASCII.GetString($buf, 0, $n)
if ($header -notmatch "element vertex (\d+)") { throw "no vertex count in the header of $ply" }
$gaussians = ([long]$Matches[1]).ToString("N0", [Globalization.CultureInfo]::InvariantCulture)
$sizeMb = [math]::Round((Get-Item $ply).Length / 1e6)
# CRLF line endings for cmd.exe and Notepad
Get-Content "$demo\start-demo.cmd" | Set-Content -Encoding ascii "$OutDir\start-demo.cmd"
(Get-Content "$demo\README.txt") | ForEach-Object {
    if ($_ -eq "@@DATA@@") { $Data.TrimEnd() -split "\r?\n" }   # one line each, so all get CRLF
    else { $_.Replace("@@GAUSSIANS@@", $gaussians).Replace("@@SIZE_MB@@", "$sizeMb") }
} | Set-Content -Encoding ascii "$OutDir\README.txt"
if (Select-String -Path "$OutDir\README.txt" -Pattern "@@" -Quiet) { throw "README placeholder left unfilled" }

if ($ViewOnlyHtml) {
    # upstream viewer with the room embedded (SOG-compressed): opens by double-click, no collision
    npx -y @playcanvas/splat-transform -w "$SceneDir\aligned.ply" --viewer-settings "$SceneDir\settings.json" `
        "$OutDir\room-view-only.html"
}
Compress-Archive -Path "$OutDir\*" -DestinationPath "$OutDir.zip"
Get-ChildItem $OutDir -Recurse -File | Select-Object @{n = "File"; e = { $_.FullName.Substring($OutDir.Length + 1) } }, Length
Get-Item "$OutDir.zip" | Select-Object FullName, Length
