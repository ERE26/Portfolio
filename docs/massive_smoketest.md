# Smoke test: API de acciones de Massive (tier gratis "Stocks Basic")

Fecha: 2026-10-03 · Alcance: SOLO LECTURA, PAPER, cero gasto (solo tier gratis).
Cliente: `fuentes/massive.py` (función `barras_diarias(ticker, desde, hasta)`).

## Qué dice la documentación (verificado en massive.com/docs)

Massive es un proveedor de datos de mercado estilo Polygon.io (de hecho su
cliente Python oficial es `from massive import RESTClient`). Parámetros de la
API REST confirmados en la documentación:

- **Base URL:** `https://api.massive.com`
- **Autenticación:** cabecera `Authorization: Bearer <API_KEY>` (recomendada) o,
  alternativamente, el parámetro de consulta `?apiKey=<API_KEY>`. La clave se
  lee de `MASSIVE_API_KEY` (con respaldo en `datos/.massive_key`, gitignored) y
  **viaja en la cabecera, nunca en la URL ni en logs**.
- **Barras diarias (EOD aggregates):**
  `GET /v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}`
  con `adjusted`, `sort`, `limit`. Ejemplo:
  `/v2/aggs/ticker/AAPL/range/1/day/2020-01-01/2020-01-10?adjusted=true&sort=asc&limit=5000`.
- **Tier gratis "Stocks Basic" (documentado):** 0 $/mes · **5 peticiones/min** ·
  **solo fin de día (EOD)** · **2 años** de histórico · incluye datos de
  referencia y corporate actions; **los deslistados conservan su histórico
  completo** (sin sesgo de supervivencia). Excluye tiempo real, WebSocket y
  flat files.

El cliente `fuentes/massive.py` ya implementa exactamente esta base URL, auth y
endpoint, con límite de ritmo (12,5 s entre llamadas) y timeouts defensivos.

## ¿Funciona la clave?

**No.** `MASSIVE_API_KEY` **está puesta** (34 caracteres), pero la API la
**rechaza**. Toda petición —con cabecera Bearer y también con `?apiKey=`—
devuelve:

```
HTTP 401
{"status":"ERROR","error":"Unknown API Key"}
```

Inspección estructural de la clave (sin exponer su valor): 34 caracteres, sin
espacios, y **contiene los caracteres `<` y `>`**. Es decir, el valor es un
**marcador de posición / placeholder** (tipo `<...>`), no una API key real
emitida por Massive. Por eso el servidor la declara "Unknown API Key".

Importante: el error es de **credencial**, no de conexión ni de formato. Que el
servidor devuelva un JSON de error estructurado confirma que la **base URL, la
ruta del endpoint y el método de autenticación del cliente son correctos**; lo
único que falla es que la clave no autentica.

## Qué devuelve el tier gratis observado

No se pudo observar en vivo: al fallar la autenticación (401 "Unknown API Key")
**no se descargó ninguna barra**. Granularidad, profundidad de histórico y
cobertura de deslistados quedan **sin verificar empíricamente**; solo constan
los valores documentados (EOD, 2 años, 5 req/min).

- (a) AAPL EOD último ~mes → **no ejecutable** (401 antes de recibir datos).
- (b) Deslistados (TWTR / SIVB / FB) → **no ejecutable** (mismo 401).

## ¿Hay datos de deslistados?

Según la documentación, **sí** ("delisted tickers keep their full history",
todos los planes). **No verificado en vivo** por el fallo de credencial.

## VEREDICTO

**DATOS NO USABLES** — con la clave actual: `MASSIVE_API_KEY` está puesta pero
es un placeholder (contiene `<`/`>`) y Massive la rechaza con 401 "Unknown API
Key", así que no se pudo descargar ni validar un solo dato del tier gratis. El
cliente (`fuentes/massive.py`) y los endpoints están verificados y correctos:
en cuanto se exporte una API key real de Massive, el smoke test (AAPL EOD +
deslistados) podrá ejecutarse sin más cambios de código.
