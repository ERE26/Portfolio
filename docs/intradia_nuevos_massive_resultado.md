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

---

## Estrategia 3 — MEAN-REVERSION al VWAP en ETFs + large-caps · `backtest_meanrev_massive.py`

Distinta a la mean-reversion ya descartada (aquella era sobre mid-caps "in play"). Aquí el
universo es **líquido y sin problema de supervivencia**: ETFs de índice/sector (SPY, QQQ,
IWM, DIA, XLK…) + top-30 large-caps por dólar-volumen point-in-time. Setup: extensión
`z = (close − VWAP) / ATR_bar`; `z ≤ −k` → **largo** (objetivo VWAP), `z ≥ +k` → **corto**
(objetivo VWAP); stop a `m·ATR`; una entrada por día y lado; se evita la última media hora.
`k = 1.5`, `m = 1.0`.

**Muestra**: 40 tickers × 15 días × 20 meses (≈600 pares/semilla → muestra grande).

| modo | semilla | ops | acierto | expectancy (R) | IC95 | PF | veredicto |
|---|---|---:|---:|---:|---|---:|---|
| LARGO | 42 | 442 | 38% | -0.368 | [-0.478, -0.252] | 0.54 | PIERDE |
| LARGO | 7 | 470 | 39% | -0.399 | [-0.505, -0.293] | 0.49 | PIERDE |
| LARGO | 123 | 416 | 41% | -0.329 | [-0.441, -0.217] | 0.56 | PIERDE |
| LARGO | **pooled** | 1328 | 39% | **-0.367** | [-0.429, -0.301] | 0.53 | PIERDE |
| CORTO | 42 | 423 | 38% | -0.474 | [-0.590, -0.358] | 0.43 | PIERDE |
| CORTO | 7 | 402 | 44% | -0.394 | [-0.508, -0.273] | 0.49 | PIERDE |
| CORTO | 123 | 412 | 46% | -0.279 | [-0.404, -0.153] | 0.62 | PIERDE |
| CORTO | **pooled** | 1237 | 42% | **-0.383** | [-0.451, -0.316] | 0.51 | PIERDE |
| L+S | **pooled** | 2565 | 41% | **-0.375** | [-0.423, -0.326] | 0.52 | PIERDE |

**Veredicto Estrategia 3: PIERDE, robusto y con muestra grande.**
- Largo y corto **pierden en las 3 semillas**, con IC95 que **excluye el 0 por abajo** en
  todas (≈ −0,37 R por operación). Con >1.200 ops por lado y semilla, esto no es ruido: es
  un resultado sólido y negativo.
- Acierto **< 46%**: la extensión respecto al VWAP **no revierte** — continúa (momentum
  intradía), y el stop a 1·ATR salta antes de que el precio vuelva al VWAP. Fadear
  extensiones de ETFs/large-caps líquidos es perder dinero a mano.
- Confirma y **extiende a universo limpio** el hallazgo previo del repo (la mean-reversion
  a VWAP pierde): no era un artefacto del universo mid-cap; tampoco funciona en índices y
  large-caps. Puerta cerrada.

---

## Estrategia 4 — TIME-OF-DAY (caracterización horaria) · `backtest_timeofday_massive.py`

Descriptivo (no una estrategia con stops): mide el **retorno medio bruto por tramo horario**
sobre el universo PIT de acciones, y dos tests direccionales: **momentum de última hora**
(¿la dirección de la mañana continúa 15:00–16:00?) y **reversión de mediodía** (¿se fadea la
mañana entre 12:00–14:00?). Retornos **brutos** (sin coste): si no hay sesgo bruto, menos aún
neto. 30 tickers × 15 días × 20 meses, ≈1.346 ticker-días/semilla.

### Retorno medio por tramo (pooled de las 3 semillas, bps)

| tramo ET | media pooled | IC95 | signif.? | ¿consistente entre semillas? |
|---|---:|---|---|---|
| 09:30–10:00 | +2.3 | [-6.5, +10.9] | no | no |
| 10:00–11:00 | -0.2 | [-6.8, +6.5] | no | no |
| 11:00–12:00 | **-11.5** | [-16.4, -6.7] | SÍ | 2/3 (42 no sig.) |
| 12:00–13:00 | +1.9 | [-2.3, +6.0] | no | no |
| 13:00–14:00 | +15.6 | [+7.0, +25.5] | SÍ | **no** (lo lleva seed 42: +41.6 vs +8.6/−3.4) |
| 14:00–15:00 | **-11.4** | [-15.3, -7.6] | SÍ | 2/3 (seed 7 no sig.) |
| 15:00–16:00 | -0.9 | [-5.3, +3.3] | no | no |

### Tests direccionales (pooled, bps brutos)

| test | pooled | IC95 | signif.? | por semilla |
|---|---:|---|---|---|
| momentum última hora (signo mañana × ret 15–16h) | **+7.5** | [+3.4, +11.6] | SÍ | 42 +19.4 SÍ · 7 −3.2 no · 123 +6.5 SÍ |
| reversión mediodía (−signo mañana × ret 12–14h) | -8.3 | [-19.6, +2.4] | no | 42 −16.7 no · 7 +10.2 no · 123 −18.3 SÍ |

**Veredicto Estrategia 4: no hay un sesgo horario limpio y robusto explotable.**
- Los tramos con media *pooled* significativa (débil mañana 11–12h, debilidad 14–15h, fuerza
  13–14h) **no son consistentes entre semillas**: el "+15,6 bps de 13–14h" lo produce casi
  entero la semilla 42 (+41,6 vs +8,6 / −3,4). Son artefactos de muestra, no estructura.
- **Reversión de mediodía: no existe** (pooled −8,3 bps, IC95 incluye 0; de hecho el signo
  dice que fadear la mañana *pierde*, es decir hay más continuación que reversión).
- **Momentum de última hora: señal débil pero la más coherente del estudio.** La dirección
  del día tiende a **continuar** de 15:00 a 16:00: +7,5 bps *pooled* (IC95 excluye 0) y
  significativa en 2 de 3 semillas. Pero: (1) es **bruta** — un round-trip de ~2–4 bps se
  come la mitad; (2) **falla la semilla 7** (−3,2 bps), así que no supera el listón de las 3
  semillas; (3) es justo lo **contrario** a lo que encontró el estudio de ETFs (allí la
  última hora de SPY/QQQ/IWM *perdía*). En acciones hay un rastro de continuación; en índices
  revierte. Nada que cablear: sesgo débil, no robusto y marginal neto de coste.

---

## Síntesis — los 4 ángulos nuevos

| estrategia | muestra | resultado | ¿edge robusto (IC95>0 en 3 semillas)? |
|---|---|---|---|
| **1. Lado corto (breakdown)** | ~450 ops/seed | EOD −0,09 R (IC95 cruza 0); trailing **PIERDE** | **NO** |
| **2. Gap-fade (>3% / >2%)** | 10–54 ops/seed | acierto 55–73%, lean positivo que se diluye con más potencia | **NO** (sugerente, infra-potenciado) |
| **3. Mean-reversion VWAP (ETF+LC)** | ~1.300 ops/seed por lado | **PIERDE** robusto (−0,37 R, IC95 excluye 0, los 2 lados y 3 semillas) | **NO (pierde)** |
| **4. Time-of-day** | ~1.346 días/seed | solo un momentum débil de última hora (+7,5 bps bruto, 2/3 semillas) | **NO** |

**Conclusión honesta:** ninguno de los cuatro ángulos nuevos es un edge robusto sobre datos
limpios point-in-time. Es información valiosa:

1. **El lado corto no resucita el ORB** — simétrico al largo: no era la dirección, es que no
   hay edge de ruptura una vez quitada la supervivencia (el trailing incluso pierde).
2. **La mean-reversion a VWAP pierde también en índices/large-caps**, no solo en mid-caps: la
   extensión intradía **continúa**, no revierte. Puerta cerrada del todo.
3. **No hay estructura horaria explotable** en acciones; el único rastro (continuación de
   última hora) es débil, bruto y no robusto, y contradice al de ETFs.
4. El **gap-fade de acciones** es el único con perfil de reversión real (acierto alto, PF>1,
   lean positivo), pero **sin potencia** para probarse: los gaps grandes son raros en nombres
   líquidos. Es el único candidato que merecería re-test con más historia / universo que
   gapee más — no para cablear hoy.

Esto **concuerda** con el mapa del repo: casi todo el intradía direccional muere en limpio.
El único superviviente de toda la campaña Massive sigue siendo el **overnight de QQQ filtrado
por tendencia** (`docs/intradia_massive_resultado.md`), que **no es intradía**. El intradía
limpio, por ahora, no da un edge nuevo robusto.

### Avisos honestos

- **Muestras pequeñas a propósito** (la descarga de minuto es pesada): el veredicto de "no
  edge" es sólido donde la muestra es grande (mean-reversion: >1.300 ops) y más tentativo
  donde es pequeña (gap-fade >3%: 10–23 ops/semilla). Por eso el gap-fade queda como
  "sugerente", no como "muerto".
- **Solo research/paper.** Nada de esto está cableado en vivo ni toca la operativa.
- **Costes incluidos**: SLIP 4 bps/lado, SLIP_STOP 8 bps, borrow 3 bps en cortos. El
  time-of-day se reporta en bruto (y aún así no hay edge).
