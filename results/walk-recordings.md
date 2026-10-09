# Walk recordings (G2, G4 evidence)

Recorded by Aswin on the PC (Windows 11, RTX 3050 Laptop, Chrome, local SuperSplat Viewer 1.37.0
with the RoomRecon UI patch, `&ministats`) on 2026-10-09 with the Windows screen recorder. Videos are
too large for git; they live outside the repo:

| Room | File (copy) | Original | Length | MiniStats read from the video |
|---|---|---|---|---|
| `vr_room` (MuSHRoom, run `dns-30k-lite`) | `E:\roomrecon-scenes\recordings\g2-vr_room-walk.mp4` | `Videos\Screen Recordings\Screen Recording 2026-10-09 185254.mp4` (68 MB) | 81 s | frame 10.0 ms (~100 FPS), 0.908 M splats, GPU 4.3 ms (at 20 s) |
| `sejal-cap2` (own room, run `dns-30k-lite`) | `E:\roomrecon-scenes\recordings\g4-sejal-cap2-walk.mp4` | `Videos\Screen Recordings\Screen Recording 2026-10-09 185405.mp4` (29 MB) | 34 s | frame 18.0 ms (~55 FPS), GPU 7.2 ms at 5 s; 0.746 M splats, GPU 5.0 ms at 20 s |

FPS values are single readings of the on-screen overlay, not averages. The `vr_room` recording's
collision source (default splat-based or the mesh-based study version) was not noted.
