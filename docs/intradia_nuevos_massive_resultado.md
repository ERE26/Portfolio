# Ángulos intradía NUEVOS sobre datos limpios (Massive, point-in-time)

**Qué es esto.** El ORB largo confirmado murió en limpio (`docs/orb_massive_resultado.md`:
el edge se lo comía la supervivencia). La búsqueda de familias nuevas sobre **ETFs**
(`docs/intradia_massive_resultado.md`) dejó solo el *overnight de QQQ filtrado por
tendencia* como superviviente. Este documento ataca ángulos **complementarios y aún no
medidos**, sobre el **universo de ACCIONES point-in-time** (no solo ETFs) y sin sesgo de
supervivencia. Honestidad > resultado: si un ángulo muere, se dice.

Estos cuatro ángulos **no** solapan con los de `intradia_massive_resultado.md`:
- allí el ORB fue a nivel de **índice** (SPY/QQQ/IWM); aquí es el **lado corto sobre
  acciones** (universo in-play, simétrico al ORB de stock que murió).
- allí el gap-fade fue sobre **ETFs** (huecos pequeños, ~0,5–1,5%); aquí es sobre **gaps
  grandes de acciones** (>3%), un fenómeno distinto.
- allí la mean-reversion/última hora se midió sobre ETFs vía overnight/última hora; aquí
  es **mean-reversion intradía al VWAP** sobre ETFs **+ large-caps** y una
  **caracterización horaria** sobre acciones.

## Metodología común (`backtest_nuevos_massive.py`)

Reutiliza la infra ya commiteada (`fuentes.massive.barras_intradia`, `universo_pit`), no
la duplica.

- **Fuente**: Massive (Polygon-compatible), barras de **5 min de sesión regular**
  (09:30–16:00 ET).
- **Universo point-in-time** (`universo_pit.universo_asof`): para cada día, el universo es
  el top por dólar-volumen del GROUPED del mes anterior (acciones comunes activas +
  deslistadas; mega-caps excluidas por proxy). Incluye deslistados ⇒ **sin sesgo de
  supervivencia**. Para la mean-reversion se usa ETFs líquidos (sin supervivencia por
  construcción) + top-30 large-caps PIT.
- **Sin look-ahead**: VWAP de sesión y rango de apertura solo usan el pasado de la sesión;
  la entrada se rellena al **cierre de la vela que confirma + slippage**.
- **Costes**: `SLIP = 4 bps`/lado, `SLIP_STOP = 8 bps` en stops, `+ BORROW = 3 bps` por
  operación corta.
- **Robustez**: **3 semillas** (42 / 7 / 123) = 3 muestras independientes. Bootstrap
  **IC95** de la expectancy por operación. *Edge robusto* **solo** si IC95 > 0 en **las 3
  semillas** con ≥20 ops por semilla.

Muestra ≈ 30–40 tickers × 15 días × ~20 meses por semilla (muestras pequeñas: la descarga
de minuto es pesada).

---

## Estrategia 1 — LADO CORTO (breakdown, simétrico al ORB) · `backtest_short_massive.py`

Espejo exacto del ORB confirmado que murió, pero a la baja: 1ª vela de 5 m **bajista**
(dirección = color de la 1ª vela); entrada cuando una vela **cierra por debajo del mínimo**
del rango de apertura (ruptura confirmada a la baja); stop = **máximo** del rango; salida
EOD y variante **trailing** (2·ATR tras −1 R a favor). Coste de borrow incluido. Universo
PIT de acciones comunes in-play (mega-caps excluidas).

**Muestra**: 30 tickers × 15 días × 20 meses, semillas 42/7/123 (≈450 pares/semilla).

| modo | semilla | ops | acierto | expectancy (R) | IC95 | PF | veredicto |
|---|---|---:|---:|---:|---|---:|---|
| EOD | 42 | 135 | 49% | +0.042 | [-0.152, +0.251] | 1.10 | sin edge |
| EOD | 7 | 175 | 30% | -0.200 | [-0.412, +0.057] | 0.66 | sin edge |
| EOD | 123 | 162 | 40% | -0.078 | [-0.252, +0.116] | 0.85 | sin edge |
| EOD | **pooled** | 472 | 39% | **-0.089** | [-0.206, +0.046] | 0.83 | sin edge |
| EOD+trail2ATR | 42 | 135 | 50% | -0.042 | [-0.191, +0.114] | 0.90 | sin edge |
| EOD+trail2ATR | 7 | 175 | 35% | -0.259 | [-0.419, -0.089] | 0.55 | **PIERDE** |
| EOD+trail2ATR | 123 | 162 | 43% | -0.135 | [-0.281, +0.012] | 0.72 | sin edge |
| EOD+trail2ATR | **pooled** | 472 | 42% | **-0.154** | [-0.242, -0.064] | 0.69 | **PIERDE** |

**Veredicto Estrategia 1: NO hay edge en corto — es más, el trailing PIERDE.**
- EOD: expectancy *pooled* **negativa** (−0,089 R) y el IC95 de cada semilla cruza cero
  (42 en verde flojo, 7 y 123 en rojo). Sin edge.
- EOD+trailing: **pierde de forma significativa** (pooled IC95 [−0,242, −0,064] **excluye el
  cero por abajo**; la semilla 7 también). El trailing corta las ganadoras y deja correr las
  perdedoras en un patrón sin drift.
- Lectura honesta: el breakdown confirmado sobre acciones **no resucita** la familia ORB. El
  lado corto es tan inútil como el largo (o peor con trailing). Simétrico al resultado del
  ORB largo: no era una cuestión de dirección, era que **no hay edge de ruptura** una vez
  quitada la supervivencia.

---

## Estrategia 2 — GAP-FADE de gaps grandes de acción · `backtest_gapfade_massive.py`

En gaps de apertura grandes (|open/cierre_previo − 1| > umbral) se **fadea** con pequeña
confirmación de giro:
- gap UP → **corto** cuando una vela cierra por debajo del mínimo de la 1ª vela (giro bajo).
- gap DOWN → **largo** cuando una vela cierra por encima del máximo de la 1ª vela.

Stop = extremo del día hasta la entrada. Dos salidas: **objGapFill** (cierra al rellenar
el hueco = tocar el cierre previo, si el objetivo queda en el lado rentable; si no, EOD) y
**EOD** (aguanta al cierre). El cierre previo se toma del GROUPED ajustado del día anterior
(sin falsos gaps por splits). Universo PIT de acciones. Coste de borrow en cortos.

**Muestra**: 30 tickers × 15 días × 20 meses; los gaps grandes son **raros** en nombres
líquidos (mega-caps excluidas), de ahí las muestras pequeñas.

### Umbral >3%

| modo | semilla | ops | acierto | expectancy (R) | IC95 | PF | veredicto |
|---|---|---:|---:|---:|---|---:|---|
| objGapFill | 42 | 16 | 75% | +0.200 | [-0.155, +0.544] | 1.96 | sin edge |
| objGapFill | 7 | 10 | 80% | +0.111 | [-0.222, +0.411] | 1.79 | sin edge |
| objGapFill | 123 | 23 | 70% | +0.242 | [-0.100, +0.605] | 1.91 | sin edge |
| objGapFill | **pooled** | 49 | 73% | **+0.201** | [-0.013, +0.420] | 1.91 | sin edge |
| EOD | 42 | 16 | 75% | +0.375 | [-0.055, +0.796] | 2.80 | sin edge |
| EOD | 7 | 10 | 60% | +0.009 | [-0.437, +0.458] | 1.03 | sin edge |
| EOD | 123 | 23 | 70% | +0.456 | [+0.027, +0.914] | 2.71 | EDGE+ |
| EOD | **pooled** | 49 | 69% | **+0.338** | [+0.074, +0.615] | 2.33 | EDGE+ |

A >3% **todas las semillas dan expectancy positiva** y acierto alto (69–73%), y el *pooled*
de ambas salidas es positivo (EOD con IC95 que excluye el 0). **Pero** cada semilla tiene
10–23 ops: por debajo del mínimo de 20 para el veredicto de robustez. Sugerente, no probado.

### Umbral >2% (más eventos, más potencia)

| modo | semilla | ops | acierto | expectancy (R) | IC95 | PF | veredicto |
|---|---|---:|---:|---:|---|---:|---|
| objGapFill | 42 | 38 | 58% | -0.092 | [-0.330, +0.138] | 0.74 | sin edge |
| objGapFill | 7 | 19 | 74% | +0.108 | [-0.200, +0.415] | 1.50 | sin edge |
| objGapFill | 123 | 54 | 67% | +0.145 | [-0.053, +0.357] | 1.61 | sin edge |
| objGapFill | **pooled** | 111 | 65% | **+0.057** | [-0.080, +0.201] | 1.21 | sin edge |
| EOD | 42 | 38 | 53% | +0.078 | [-0.237, +0.406] | 1.21 | sin edge |
| EOD | 7 | 19 | 53% | +0.011 | [-0.405, +0.493] | 1.03 | sin edge |
| EOD | 123 | 54 | 57% | +0.207 | [-0.031, +0.475] | 1.72 | sin edge |
| EOD | **pooled** | 111 | 55% | **+0.129** | [-0.046, +0.317] | 1.39 | sin edge |

Al bajar a >2% (duplica la muestra), el *edge* **se diluye**: la expectancy *pooled* cae de
+0,34→+0,13 R (EOD) y +0,20→+0,06 R (objGapFill), **una semilla se vuelve negativa**, y
ningún IC95 excluye el 0. Clásico de muestra pequeña: lo "bonito" del >3% era el extremo
poco poblado.

**Veredicto Estrategia 2: NO hay edge robusto, pero hay una señal sugerente honesta.**
- El gap-fade de acciones tiene el perfil típico de reversión: **acierto alto** (55–73%),
  PF > 1 casi siempre, expectancy positiva en el *pooled* de las dos salidas y umbrales.
- No pasa el listón de robustez: a >3% las muestras son demasiado pequeñas (10–23 ops/semilla)
  y a >2% el *edge* se encoge hacia cero y una semilla se vuelve negativa. El IC95>0 **no se
  da en las 3 semillas** en ninguna configuración.
- Lectura honesta: es el ángulo **menos muerto** de los tres de entrada/salida direccional.
  Merecería re-testearse con más historia o un universo algo menos líquido (que gapea más)
  para ver si el lean positivo es real; con los datos de aquí es **sugerente pero no probado**.

<!-- SIGUIENTES SECCIONES: mean-reversion ETF/large-cap, time-of-day -->
