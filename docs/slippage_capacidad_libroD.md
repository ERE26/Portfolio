# Slippage y capacidad del Libro D (rotación Europa)

*2 de octubre de 2026.* Reproducible: `python stress_slippage_libroD.py`.

El Libro D está en **observación en paper**. Antes de que pueda siquiera plantearse
dinero real hay que medir dos cosas que un backtest ignora: cuánto cuesta **cruzar el
spread** y si la estrategia **cabe** en los ETFs que opera. Sin feed de bid/ask ni fills
reales todavía, estimo el spread con **Corwin-Schultz (2012)** desde el high-low diario y
la capacidad con el **volumen diario** (ADV).

## Spread efectivo estimado (Corwin-Schultz)

| ETF | País | Spread est. | ADV ($/día) | Liquidez |
|---|---|---:|---:|---|
| EWG | Alemania | 17 bps | $28,4M | líquido |
| EWU | Reino Unido | 16 bps | $41,0M | líquido |
| EWL | Suiza | 14 bps | $20,7M | líquido |
| EWP | España | 17 bps | $14,5M | medio |
| EWI | Italia | 19 bps | $14,6M | medio |
| EWN | P. Bajos | 16 bps | $6,7M | medio |
| EWQ | Francia | 17 bps | $10,9M | medio |
| EWD | Suecia | 18 bps | $5,5M | medio |
| **EWO** | **Austria** | 13 bps | **$1,0M** | **fino** |
| **EWK** | **Bélgica** | 11 bps | **$0,4M** | **fino** |

**Ojo con esto:** Corwin-Schultz da a EWK (11 bps) un spread *menor* que a SPY (23 bps).
Eso es **irreal** — un ETF que mueve $0,4M/día no tiene mejor spread que el instrumento
más líquido del mundo. El estimador mide el rango high-low *realizado*, que en nombres
finos se comprime porque hay pocas operaciones. **Los bps de arriba son un SUELO, no el
coste real** de los ETFs finos.

## Impacto del spread sobre el Sharpe: pequeño

| Escenario | Sharpe | CAGR | Drag/año |
|---|---:|---:|---:|
| Sin coste (bruto) | 0,54 | 7,1% | — |
| 10 bps (supuesto actual) | 0,49 | 6,2% | 0,83 pp |
| Spread estimado | 0,50 | 6,4% | 0,66 pp |
| 2× spread (stress) | 0,46 | 5,8% | 1,31 pp |

El turnover mensual bajo (top-3, rebalanceo mensual) hace que el spread **apenas duela**
(~0,7 pp/año). Esto es real y tranquilizador *por el lado del spread*.

## La restricción que de verdad ata: capacidad

El Libro D pone ~14,7% del equity en 3 ETFs → ~4,9% por ETF. ¿Cuánto es eso frente al
volumen diario?

| Equity | $/ETF | Posiciones que superan el 1% del ADV |
|---|---:|---|
| $100.000 | $4.900 | EWK (1% ADV) — tragable |
| $1.000.000 | $49.000 | EWO (5%), **EWK (12%)** — impacto |
| $10.000.000 | $490.000 | **EWO (47%), EWK (117%)** + resto — intragable |

**El Libro D tiene un techo de capacidad de ~$1M**, impuesto por los ETFs finos
(Austria, Bélgica), no por su edge. A $10M, una posición en EWK sería *más de un día
entero* de su volumen: imposible de ejecutar sin mover el precio.

## Qué se ha hecho con esto (análisis → acción)

- **Guard de capacidad en `operativa_rotacion.py`:** ningún objetivo supera el **1% del
  ADV** del ETF (`ADV_FRAC = 0.01`). Si el momentum elige un ETF fino, la posición se capa
  y el resto queda en liquidez. Es un control de **riesgo**, no una re-optimización de la
  estrategia (no cambia qué se elige, solo evita el impacto de mercado). A la cuenta actual
  (~$100k) apenas muerde; protege automáticamente si la cuenta crece.

## Veredicto (consistente y honesto)

El Libro D sigue siendo un **candidato marginal en observación**: edge ya justo
(DSR-vs-VGK 0,89), el spread lo recorta poco pero el estimador infraestima los nombres
finos, y tiene un **techo de capacidad de ~$1M**. Conclusión firme:

> **Paper SÍ** (genera track OOS a riesgo cero, con cuenta pequeña donde la capacidad no
> ata). **Dinero real NO** —y menos a escala— hasta: DSR-vs-VGK ≥ 0,95 + slippage medido
> con **fills reales** + track OOS. El guard de capacidad queda puesto de todas formas.
