# Walk recordings (G2, G4 evidence)

Recorded by Aswin on the PC (Windows 11, RTX 3050 Laptop, Chrome, local SuperSplat Viewer 1.37.0
with the RoomRecon UI patch, `&ministats`) on 2026-10-09 with the Windows screen recorder. Videos are
too large for git; they live outside the repo:

| Room | File (copy) | Original | Length | MiniStats read from the video |
|---|---|---|---|---|
| `vr_room` (MuSHRoom, run `dns-30k-lite`) | `E:\roomrecon-scenes\recordings\g2-vr_room-walk.mp4` | `Videos\Screen Recordings\Screen Recording 2026-10-09 185254.mp4` (68 MB) | 81 s | frame 10.0 ms (100 FPS) at all 14 samples (every 6 s); GPU 2.5-8.3 ms; 0.908 M splats |
| `sejal-cap2` (own room, run `dns-30k-lite`) | `E:\roomrecon-scenes\recordings\g4-sejal-cap2-walk.mp4` | `Videos\Screen Recordings\Screen Recording 2026-10-09 185405.mp4` (29 MB) | 34 s | frame 10.0 ms (100 FPS) at all 11 samples (every 3 s); GPU 2.9-7.1 ms; 0.746 M splats |

FPS values are readings of the on-screen overlay at regular samples (ffmpeg frames, 2026-10-09), not averages.
The frame time never changed while the GPU time varied, so it was probably limited by the display's refresh, not by rendering
(not checked). **Correction 2026-10-09:** the first reading of the `sejal-cap2` video said frame 18.0 ms (~55 FPS); zoomed
in, the overlay reads 10.0 ms (the font's dotted zero looks like an 8). The G2 reading of 18.0 ms for `vr_room`
(`g2-walk-test.md`, from a screenshot that is no longer available) may be the same misreading; it could not be re-checked. The `vr_room` recording's
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
