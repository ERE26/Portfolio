# Bitácora de operaciones: memoria del *porqué* (no solo del resultado)

El sistema ya recordaba si un nombre ganaba o perdía (memoria por ticker). Pero
eso no explica **por qué**. La bitácora (`bitacora.py`) guarda, por cada entrada,
el contexto completo para aprender de los **patrones que ganan y los que pierden**.

## Qué se guarda en cada operación

En `datos/snapshots/<id>.json` (+ un PNG), al abrir la posición:

- **MOTIVO** — descripción legible de por qué se operó. Ej.:
  *"RVOL 2.19 (volumen muy alto, stock in play); sobre VWAP +1.67%; sesión
  +1.22%; convicción +1.87; noticias negativas (−0.27)"*.
- **PATRÓN de velas** — etiqueta del detector de figuras (`patrones.figuras`):
  figura principal, tendencia y rotura. Ej.: *"rotura_resistencia_alcista"*.
- **FOTO** — un PNG del candlestick de la entrada (`<id>.png`).
- **VENTANA OHLC** — las últimas 40 velas (para redibujar o re-analizar luego).
- **FEATURES** — RVOL, dist_VWAP, ret_sesión, ATR%, sentimiento, convicción.
- **RESULTADO** — se rellena al cerrarse (win/loss, P&L) por reconciliación.

## Cómo aprende

`python bitacora.py` (o `bitacora.aprender()`) une cada entrada cerrada con su
patrón y calcula el **win-rate y el P&L medio por patrón de velas** y por
tendencia. Responde: *"¿qué formas de gráfico y qué condiciones de entrada me
funcionan de verdad?"*. Con eso se puede, más adelante, subir la prioridad de los
patrones ganadores y bajar la de los perdedores — igual que la memoria por ticker,
pero a nivel de **patrón**, que generaliza a nombres nuevos.

## Cómo encaja con el resto de la memoria

El selector tiene ahora varias capas de memoria, de corto a largo plazo:

1. **Cooldown** tras SL (no re-entrar en un whipsaw) — minutos.
2. **Ficha por ticker** (win-rate realizado encogido) — días.
3. **Bitácora por patrón** (porqué + foto + figura → win-rate por patrón) — aprende
   qué *setups* funcionan, no solo qué nombres. ← esto.
4. **Umbral de RVOL** afinado con los resultados globales (≥25 ops).
5. **Watchlist dinámica** (exploración de nombres nuevos prometedores).

## Dónde vive

`datos/snapshots/` está en `.gitignore` (es estado de la sesión, no código). El
enlace entre el resultado y su contexto se hace por el `snapshot` id guardado en
`datos/trades_intradia.csv`; `registro.reconciliar` marca el desenlace también en
la bitácora.
