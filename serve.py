"""Servidor web EN VIVO del terminal (FastAPI). Sirve el dashboard con datos frescos
de la cuenta y una API JSON; la pagina se auto-refresca. Para desplegar en Render/
Railway/Fly (ver docs/desplegar_web.md).

  uvicorn serve:app --host 0.0.0.0 --port 8000      # local
  # en produccion el host pone $PORT (ver Procfile)

Seguridad: las claves de Alpaca se leen de variables de entorno (APCA_API_KEY_ID /
APCA_API_SECRET_KEY) — NUNCA van en el repo. Si defines DASH_USER y DASH_PASS, la web
pide usuario/contraseña (recomendado: el panel muestra tu cuenta). PAPER siempre.
"""
import os
import secrets
import threading
import time

import json

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

import dashboard

app = FastAPI(title="Ilegalísimo Terminal")
_BASE = os.path.dirname(os.path.abspath(__file__))
_PLANTILLA = open(os.path.join(_BASE, "web", "plantilla.html"), encoding="utf-8").read()

# caché de datos: reunir() es pesado (Alpaca+Yahoo+noticias); se refresca cada TTL s
_cache = {"data": None, "t": 0.0}
_lock = threading.Lock()
TTL = float(os.environ.get("DASH_TTL", "45"))


def datos(ttl=TTL):
    with _lock:
        if _cache["data"] is None or (time.time() - _cache["t"]) > ttl:
            try:
                _cache["data"] = dashboard.reunir()
                _cache["t"] = time.time()
            except Exception as e:
                if _cache["data"] is None:
                    raise
                print(f"[serve] reunir fallo, sirvo caché: {str(e)[:80]}")
        return _cache["data"]


# autenticación básica opcional (si hay DASH_USER/DASH_PASS en el entorno)
_sec = HTTPBasic(auto_error=False)
_U, _P = os.environ.get("DASH_USER"), os.environ.get("DASH_PASS")


def auth(cred: HTTPBasicCredentials = Depends(_sec)):
    if not (_U and _P):
        return True                                   # sin credenciales configuradas -> abierto
    ok = cred and secrets.compare_digest(cred.username, _U) and secrets.compare_digest(cred.password, _P)
    if not ok:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "No autorizado",
                            {"WWW-Authenticate": "Basic"})
    return True


@app.get("/", response_class=HTMLResponse)
def home(_=Depends(auth)):
    html = _PLANTILLA.replace("__DATOS__", json.dumps(datos(), ensure_ascii=False))
    # en la web viva la pagina recarga sola (en el Artifact lo republica el bucle)
    html += "\n<script>setTimeout(function(){location.reload()}," + str(int(TTL * 1000 + 15000)) + ")</script>"
    return html


@app.get("/api/data")
def api(_=Depends(auth)):
    return JSONResponse(datos())


@app.get("/health")
def health():
    return {"ok": True, "cache_age_s": round(time.time() - _cache["t"], 1) if _cache["t"] else None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
