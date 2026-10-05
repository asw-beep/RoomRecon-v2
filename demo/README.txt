RoomRecon - backup demo (walkable 3D room)
==========================================

A room captured with an iPhone (ARKit poses + LiDAR depth), reconstructed as a 3D Gaussian
splat with DN-Splatter, and walkable with collision in the SuperSplat Viewer.
Course project, IIIT Kottayam, Sem 7 Computer Vision. Team: Aswin M, Sejal Satheesh.

HOW TO RUN (Windows)
  1. Unzip this folder anywhere (do not run it from inside the zip).
  2. Double-click start-demo.cmd.
     A server window opens (keep it open) and the room opens in your browser.
  3. To stop: close the server window.

  Other systems: run "python serve.py" in this folder, then open
  http://localhost:8000/?content=room/aligned.ply&collision=room/room.collision.glb&settings=room/settings.json

  Needs: Python 3.7+ and a recent Chrome or Edge. No internet, Node or install needed.
  The room file is 225 MB, so the first load takes a few seconds.

CONTROLS
  Pick walk mode in the toolbar (bottom right), click into the view, then
  W A S D = move, mouse = look, Shift = run, R = reset camera, Esc = release the mouse.
  Add &uiscale=2 to the address for bigger buttons (&uiscale=1 = original size).

FALLBACK
  room-view-only.html (if present) opens on its own by double-click: same room, no
  collision walking, just look and fly around.

CONTENTS
  index.html, index.js, index.css   SuperSplat Viewer 1.37.0 (MIT, see
                                    LICENSE-supersplat-viewer.txt), with a RoomRecon patch
                                    for bigger buttons
  room/aligned.ply                  the room: 908,249 Gaussians, metric scale, Y up
  room/room.collision.glb           collision mesh (walls and floor)
  room/settings.json                start position (inside the room, eye height)
  room/scene.json                   how the room was aligned
  serve.py, start-demo.cmd          local web server and launcher

DATA
  Room: "vr_room" from the MuSHRoom dataset (Ren et al.), CC BY 4.0.
  Trained on the GPU machine, run dns-30k-lite (2026-10-05); see results/ in the repo.
