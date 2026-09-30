"""Cliente HTTP con reintentos, cache en disco y cabeceras de navegador.

La cache evita machacar las APIs al repetir pruebas y hace reproducibles los
experimentos: dos ejecuciones el mismo dia ven exactamente los mismos datos.
"""
import hashlib
import json
import os
import time

import requests

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/122.0 Safari/537.36")


class Cliente:
    def __init__(self, cache_dir="datos/cache", ttl=3600, reintentos=4, pausa=1.0,
                 timeout=30):
        self.cache_dir = cache_dir
        self.ttl = ttl
        self.reintentos = reintentos
        self.pausa = pausa
        self.timeout = timeout
        self.sesion = requests.Session()
        self.sesion.headers.update({"User-Agent": UA, "Accept": "application/json, text/plain, */*"})
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def _ruta_cache(self, clave):
        h = hashlib.sha1(clave.encode()).hexdigest()[:20]
        return os.path.join(self.cache_dir, f"{h}.json") if self.cache_dir else None

    def _leer_cache(self, ruta):
        if ruta and os.path.exists(ruta) and (time.time() - os.path.getmtime(ruta) < self.ttl):
            try:
                with open(ruta, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def get_json(self, url, params=None, headers=None, cache=True, metodo="GET", body=None):
        clave = f"{metodo} {url} {json.dumps(params, sort_keys=True)} {json.dumps(body, sort_keys=True)}"
        ruta = self._ruta_cache(clave) if cache else None
        pega = self._leer_cache(ruta)
        if pega is not None:
            return pega
        ultimo = None
        for intento in range(self.reintentos):
            try:
                if metodo == "POST":
                    r = self.sesion.post(url, params=params, json=body, headers=headers,
                                         timeout=self.timeout)
                else:
                    r = self.sesion.get(url, params=params, headers=headers, timeout=self.timeout)
                if r.status_code == 200:
                    datos = r.json()
                    if ruta:
                        with open(ruta, "w", encoding="utf-8") as f:
                            json.dump(datos, f)
                    return datos
                ultimo = f"HTTP {r.status_code}"
                # 429 (rate limit) y 5xx: merece la pena reintentar con espera creciente
                if r.status_code not in (429, 500, 502, 503, 504):
                    break
            except Exception as e:
                ultimo = f"{type(e).__name__}: {e}"
            time.sleep(self.pausa * (2 ** intento))
        raise RuntimeError(f"No se pudo obtener {url} -> {ultimo}")
