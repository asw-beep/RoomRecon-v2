"""Serve this folder on http://localhost:8000 for the RoomRecon backup demo (stdlib only)."""
import http.server
import os

PORT = 8000
os.chdir(os.path.dirname(os.path.abspath(__file__)))
Handler = http.server.SimpleHTTPRequestHandler
# Windows can map .js to text/plain from the registry, and browsers refuse that for scripts
Handler.extensions_map.update({".js": "text/javascript", ".css": "text/css",
                               ".json": "application/json", ".html": "text/html",
                               ".glb": "model/gltf-binary", ".ply": "application/octet-stream",
                               ".wasm": "application/wasm"})
print(f"RoomRecon demo on http://localhost:{PORT}/  (close this window to stop)")
http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
