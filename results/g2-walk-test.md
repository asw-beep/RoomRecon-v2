# G2 — Walk test: `vr_room` in the SuperSplat Viewer (PC)

Machine: PC (Windows 11, RTX 3050 Laptop 6 GB), browser, WebGPU renderer.
Date: 2026-10-05. Run `dns-30k-lite` (see `g1-vr_room.md`, `g2-export.md`).
Viewer: local SuperSplat Viewer 1.37.0 build (`E:\roomrecon-scenes\supersplat-viewer`), with
our UI-size patch (`viewer/supersplat-viewer-1.37.0-roomrecon.patch`).

## How it was run

```powershell
.\scripts\collision_room.ps1 -Dir E:\roomrecon-scenes\vr_room\dns-30k-lite   # voxel 0.05 m
npx -y http-server E:\roomrecon-scenes\supersplat-viewer\public -p 3000 -c-1
npx -y http-server E:\roomrecon-scenes\vr_room\dns-30k-lite -p 8081 --cors -c-1
```

URL: `http://localhost:3000/?content=http://localhost:8081/aligned.ply&collision=http://localhost:8081/room.collision.glb&settings=http://localhost:8081/settings.json&ministats`

- `aligned.ply`: `splat.ply` rotated by `euler_xyz_deg` (−86.59, 0.31, 0.32) and scaled ×1.0378.
- Collision: `room.collision.glb`, 15.2k triangles (268 KB), from splat-transform 3.9.0 voxels at
  0.05 m with exterior fill and carve. Scene bounds 12.5 × 7.06 × 9.43 m, so the viewer's
  walk-mode minimum (5 m horizontally) is met without changing the viewer.
- The `.glb` collision worked; the `room.voxel.json` alternative was not needed.

## Checks (answered by the user, unless marked)

| # | Check | Result |
|---|---|---|
| 1 | Start inside the room at eye height | Yes. Starts facing the yellow chairs at about eye height (also seen by Claude in Chrome) |
| 2 | Walls stop you | **Yes**, "the walls stop me realistically" |
| 3 | Never fall through the floor | Yes, covered by the user's "everything is working" (not asked separately) |
| 4 | Collision matches the visible walls | **Yes**, stops at the visible surface |
| 5 | No large floaters in the walkable area | Yes, covered by "everything is working" (not asked separately) |
| 6 | Smooth movement | MiniStats in the user's screenshot: **frame 18.0 ms (~55 FPS), GPU 3.5 ms**, 0.908M splats |

**G2 exit criterion met: the room is walkable with collision.** 908k splats ran at ~55 FPS on the
laptop, so the ~400–500k comfort limit measured in v1 does not apply to this viewer and machine.
No pruning was needed.

## Problems found and fixed

| Problem | Fix |
|---|---|
| Camera could not move: the viewer threw `Cannot read properties of undefined (reading 'sharpness')` on startup and never attached its controls. Our `settings.json` lacked `postEffectSettings`, which the v2 schema requires | `scene_cli.viewer_settings()` writes the viewer's own defaults, all effects off (`e8ca60f`, test added) |
| `collision_room.ps1` silently kept a stale `aligned.ply` on rerun | splat-transform `-w` on both calls (`a613bc6`) |
| Toolbar and panels too small to use comfortably | Local viewer patch: `?uiscale=N`, default 1.6 (`297447b`). MiniStats is drawn into the canvas and cannot be scaled this way |

## Not done

- No screen recording of the walk yet (`walk.mp4`, outside git).
- FPS is one reading from a screenshot, not a measured walk over the whole room.
