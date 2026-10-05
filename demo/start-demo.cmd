@echo off
rem RoomRecon backup demo: serve this folder on http://localhost:8000 and open the walkable room.
cd /d "%~dp0"
set PY=python
where python >nul 2>nul || set PY=py
start "RoomRecon demo server - close to stop" %PY% serve.py
timeout /t 3 /nobreak >nul
start "" "http://localhost:8000/?content=room/aligned.ply&collision=room/room.collision.glb&settings=room/settings.json"
