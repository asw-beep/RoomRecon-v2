# Align the splat to a Y-up, metric-ish frame, keep the room cluster, build walk collision.
# Usage: .\scripts\collision_room.ps1 -Dir E:\roomrecon-scenes\vr_room\dns-30k
# scene.json is in PlayCanvas' frame (see roomrecon/scene.py), which is the frame splat-transform
# applies -r/-s and --seed-pos in. Comma arguments are quoted: unquoted, PowerShell turns
# "30,40,50" into an array and the tool silently ignores the rotation.
param([Parameter(Mandatory)][string]$Dir, [double]$VoxelSize = 0.05)
$ErrorActionPreference = "Stop"
$s = Get-Content "$Dir\scene.json" -Raw | ConvertFrom-Json
$inv = [Globalization.CultureInfo]::InvariantCulture
$rot = ($s.euler_xyz_deg | ForEach-Object { $_.ToString("F6", $inv) }) -join ","
$seed = ($s.spawn_position | ForEach-Object { $_.ToString("F4", $inv) }) -join ","
$scale = ([double]$s.scale).ToString("F6", $inv)
Write-Host "rotate '$rot'  scale $scale  seed '$seed'"
npx -y @playcanvas/splat-transform "$Dir\splat.ply" -r "$rot" -s "$scale" "$Dir\aligned.ply"
npx -y @playcanvas/splat-transform "$Dir\aligned.ply" --seed-pos "$seed" --filter-cluster `
    --voxel-size "$VoxelSize" --voxel-external-fill --voxel-carve --collision-mesh "$Dir\room.voxel.json"
Get-ChildItem $Dir | Select-Object Name, Length
