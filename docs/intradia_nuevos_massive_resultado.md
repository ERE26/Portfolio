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

<!-- SIGUIENTES SECCIONES: gap-fade, mean-reversion ETF/large-cap, time-of-day -->
