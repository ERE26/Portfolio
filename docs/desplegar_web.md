# Desplegar el terminal como web EN VIVO (gratis)

El terminal (`web/plantilla.html`) se puede ver de dos formas:

1. **Artifact** (lo que ya tienes): una foto que el bucle **republica cada 5 min**.
   Cero infraestructura. Enlace privado de claude.ai. Ya funciona.
2. **Web en vivo** (`serve.py`, FastAPI): un servidor que sirve el terminal con datos
   **frescos de la cuenta** y una API JSON; la página se **auto-refresca** sola. Esto
   es lo que montamos aquí. Necesita un sitio donde correr el backend Python.

Ambas usan la MISMA plantilla y los MISMOS datos. La web en vivo solo añade un
servidor que llama a `dashboard.reunir()` bajo demanda (con caché de ~45 s).

## Qué hace `serve.py`
- `GET /` → el terminal con los datos del momento (y recarga sola).
- `GET /api/data` → el JSON completo (por si quieres construir otra cosa encima).
- `GET /health` → estado.
- **Seguridad**: las claves de Alpaca se leen de variables de entorno
  (`APCA_API_KEY_ID` / `APCA_API_SECRET_KEY`); **nunca** van en el repo. Si defines
  `DASH_USER` y `DASH_PASS`, la web pide usuario/contraseña (recomendado, porque el
  panel muestra tu cuenta). Siempre PAPER (la clave empieza por `PK`).

## Probarlo en local (opcional)
```bash
pip install -r requirements.txt
export APCA_API_KEY_ID=PK...        # tus claves PAPER
export APCA_API_SECRET_KEY=...
export DASH_USER=eduardo DASH_PASS=unaclave   # opcional
uvicorn serve:app --host 0.0.0.0 --port 8000
# abre http://localhost:8000
```

## Desplegar en Render (plan gratuito) — pasos

Lo que necesitas de ti: una cuenta gratis en https://render.com (puedes entrar con
GitHub). El repo ya trae `render.yaml` y `Procfile`, así que es casi automático.

1. **Sube el repo a tu GitHub** (si no lo está ya) y entra en Render con esa cuenta.
2. En Render: **New → Blueprint**, elige este repositorio. Render lee `render.yaml`
   y crea el servicio `ilegalisimo-terminal` (Python, plan free).
3. Te pedirá las **variables de entorno** (están marcadas `sync: false` = secretas):
   - `APCA_API_KEY_ID` = tu clave PAPER de Alpaca (empieza por `PK`).
   - `APCA_API_SECRET_KEY` = tu secreto de Alpaca.
   - `DASH_USER` y `DASH_PASS` = usuario/contraseña para proteger el panel (ponlos).
4. **Create / Deploy**. En 2-3 min tendrás una URL tipo
   `https://ilegalisimo-terminal.onrender.com`. Ábrela, mete usuario/contraseña, y
   verás el terminal en vivo actualizándose solo.

> Alternativas equivalentes: **Railway** (railway.app) o **Fly.io** — mismo
> `serve.py`, mismas variables de entorno. Render es el más simple con `render.yaml`.

## Notas honestas
- **Plan gratuito de Render**: el servicio se "duerme" tras ~15 min sin visitas y
  tarda unos segundos en despertar en la siguiente visita. Para un panel personal es
  suficiente; si quieres que esté siempre despierto, es un plan de pago (no hace falta).
- **Paneles de histórico** (tear-sheet y la red del Modelo) necesitan los datos
  cacheados (`datos/mercado/1d`, `datos/modelo_cache.json`), que **no** van al repo
  (gitignore). En un despliegue limpio esos paneles aparecen como "muestra
  insuficiente" / "ejecuta modelo.py" y **degradan con elegancia**; el resto (cuenta,
  riesgo, posiciones, velas, mercado, calendario, noticias) va 100% en vivo.
  Si los quieres en la web, hay que subir esos ficheros al servidor (o generar el
  histórico allí) — te lo monto si decides ir por esa vía.
- **Seguridad**: nunca subas las claves al repo. Van SOLO como variables de entorno
  en Render. Y es cuenta PAPER: nunca dinero real.
- **Nada obligatorio**: si no quieres montar servidor, el Artifact que ya republica el
  bucle cada 5 min te da el terminal en vivo sin desplegar nada.
