# Collision accuracy study (G8): build walk collision from the splat (S), DN-Splatter's mesh (M),
# the phone's LiDAR (L) and the laser scan (GT) with exactly the demo's splat-transform pipeline
# (scripts/collision_room.ps1), then compare them with roomrecon.collision.
# Usage: .\scripts\collision_study.ps1 -Study E:\roomrecon-scenes\collision-study -SceneDir E:\roomrecon-scenes\vr_room\dns-30k-lite
#   -Study holds gt_points.npz, mesh_points.npz, lidar_points.npz (sampled on the GPU machine).
#   -SceneDir is only read (splat.ply, scene.json); every output goes under -Study.
param([Parameter(Mandatory)][string]$Study, [Parameter(Mandatory)][string]$SceneDir)
$ErrorActionPreference = "Stop"
$s = Get-Content "$SceneDir\scene.json" -Raw | ConvertFrom-Json
$inv = [Globalization.CultureInfo]::InvariantCulture
$rot = ($s.euler_xyz_deg | ForEach-Object { $_.ToString("F6", $inv) }) -join ","
$seed = ($s.spawn_position | ForEach-Object { $_.ToString("F4", $inv) }) -join ","
$scale = ([double]$s.scale).ToString("F6", $inv)
Write-Host "rotate '$rot'  scale $scale  seed '$seed'"
foreach ($n in "mesh", "lidar", "gt") {
    python -m roomrecon.collision.prep "$Study\${n}_points.npz" "$Study\${n}_points.ply"
}
$src = @{ S = "$SceneDir\splat.ply"; M = "$Study\mesh_points.ply"; L = "$Study\lidar_points.ply"; GT = "$Study\gt_points.ply" }
foreach ($k in "S", "M", "L", "GT") {
    $d = "$Study\$k"
    New-Item -ItemType Directory -Force $d | Out-Null
    npx -y @playcanvas/splat-transform -w $src[$k] -r "$rot" -s "$scale" "$d\aligned.ply"
    npx -y @playcanvas/splat-transform -w "$d\aligned.ply" --seed-pos "$seed" --filter-cluster `
        --voxel-size 0.05 --voxel-external-fill --voxel-carve --collision-mesh "$d\room.voxel.json"
}
python -m roomrecon.collision --glb "S=$Study\S\room.collision.glb" --glb "M=$Study\M\room.collision.glb" `
    --glb "L=$Study\L\room.collision.glb" --glb "GT=$Study\GT\room.collision.glb" `
    --gt-points "$Study\gt_points.npz" --lidar-points "$Study\lidar_points.npz" `
    --scene "$SceneDir\scene.json" --out "$Study\results"
