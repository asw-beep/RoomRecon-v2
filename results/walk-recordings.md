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

The user's own copies are in the repo's `Recordings/` folder (git-ignored, `*.mp4`): `room-1.mp4` is
the `vr_room` walk; the room-2 (own room) walk was still named `Screen Recording 2026-10-09
185405.mp4` there on 2026-10-09 (to be renamed `room-2.mp4`).

## Walk test of the collision study (S vs M), 2026-10-09

Aswin walked `vr_room` with the default splat-based collision (S) and with the mesh-based collision
(M) in the local viewer: **"personally I didn't feel anything different."** The measured differences
(`g8c-collision-accuracy.md`: 17.8 % vs 1.7 % of real walls missing at body height) were not felt in
a free walk; they are concentrated in places such a walk may not reach (furniture and wall stretches
off the usual path). This is reported as is: the offline measurement and the felt experience differ.
