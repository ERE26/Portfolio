# Presupuesto de riesgo entre libros — el "SO del pod-shop"

*2 de octubre de 2026.* Reproducible: `python presupuesto_cross_book.py`.

Con cuatro libros ya vivos en paper (A intradía, B swing/momentum US, C TSMOM
cross-asset, D rotación Europa), la pregunta deja de ser *"¿qué estrategia?"* y pasa a
ser **"¿cuánto riesgo a cada una?"**. Repartir a ojo (20% aquí, 15% allá) es exactamente
lo que separa a un aficionado de una plataforma multi-estrategia. La respuesta
profesional es repartir por **riesgo y correlación** (HRP), no por corazonada.

## Lo que mide cada libro (mensual, 2008–2026, neto de costes)

| Libro | Qué es | Sharpe | Vol | CAGR | maxDD |
|---|---|---:|---:|---:|---:|
| B swing | momentum transversal US (long/short) | 0,34 | 11,9% | 3,4% | −33,0% |
| C TSMOM | time-series momentum cross-asset | 0,51 | 14,3% | 6,4% | −28,7% |
| D rotación | rotación por países Europa | 0,57 | 15,2% | 7,9% | −30,1% |

Ninguno es heroico en solitario (Sharpe 0,3–0,6). Eso **no es el punto**.

## La joya está en la correlación

Matriz de correlación mensual (213 meses comunes):

| | B swing | C TSMOM | D rotación |
|---|---:|---:|---:|
| **B swing** | 1,00 | 0,25 | **−0,02** |
| **C TSMOM** | 0,25 | 1,00 | 0,41 |
| **D rotación** | −0,02 | 0,41 | 1,00 |

**B swing y D rotación son casi ortogonales (−0,02):** el momentum de acciones US y la
rotación entre países europeos ganan y pierden en momentos distintos. Ese cero es oro —
es lo que permite sumar libros sin sumar riesgo.

## Qué pasa al combinarlos (pesos HRP)

| | Mejor libro solo | Cartera multi-libro (HRP) |
|---|---:|---:|
| Sharpe | 0,59 | **0,61** |
| Volatilidad | 14–15% | **9,0%** |
| Max drawdown | −28% a −33% | **−14,2%** |
| Ratio de diversificación | 1,00 | **1,45** |

El Sharpe apenas sube (0,61 vs 0,59). **El premio no es más retorno: es que el drawdown
se parte a la mitad** (−14% vs −30%) y la volatilidad baja a 9%. Para un mandato de
**preservación de capital**, eso es exactamente lo que se busca — ganar parecido
sufriendo la mitad.

## El presupuesto resultante

Pesos de **riesgo** HRP entre los libros largos → traducidos a **capital** reservando un
*sleeve* fijo del 35% para el intradía (horizonte distinto, va aparte):

| Libro | Peso de riesgo (HRP) | Fracción de capital |
|---|---:|---:|
| A intradía (sleeve fijo) | — | 35,0% |
| B swing | 51,9% | 33,7% |
| C TSMOM | 25,5% | 16,6% |
| D rotación | 22,6% | 14,7% |

**Honestidad — por qué el swing, con el peor Sharpe, se lleva el mayor peso:** HRP reparte
por **riesgo y diversificación, no por retorno**. El swing es la pata de menor vol (11,9%)
y la mejor diversificadora (−0,02 con rotación), así que el algoritmo le carga más para
minimizar el drawdown de la cartera. Es deliberado y correcto para este objetivo; quien
quiera maximizar retorno inclinaría por Sharpe, pero ese no es nuestro mandato.

## Cómo alimenta esto a la operativa (análisis → acción)

No es un informe que se archiva. Vive en **`patrones/presupuesto.py`**, la **fuente única
de verdad** de la asignación por libro:

- `operativa_tsmom.py` y `operativa_rotacion.py` ya **leen su `capital_frac` de ahí** (antes
  llevaban 20% y 15% hardcodeados sueltos; ahora 16,6% y 14,7% del presupuesto).
- Hoy el reparto es un **prior de backtest**. En cuanto se acumule historia **viva** por
  libro (ya atribuida vía `client_order_id`), `objetivo()` detecta ≥24 meses reales de los
  tres libros y **recalcula el HRP con retornos reales** automáticamente — sin tocar código.
- *Fail-safe:* ante cualquier fallo, cae al prior. Nunca deja a un libro sin presupuesto.

## Límites reconocidos

- Es un **prior de backtest**, no la verdad viva. Su valor es ser un punto de partida
  honesto y auto-actualizable, no una cifra sagrada.
- Las tres series son backtests con sus propios supuestos de coste (10 bps + turnover);
  el slippage real de los ETFs finos europeos (Libro D) hará su Sharpe neto algo menor.
- HRP asume que la estructura de correlación es razonablemente estable. En un crash de
  correlación-a-1 (todo cae junto) la diversificación se encoge justo cuando más se
  necesita; por eso **encima del presupuesto siguen mandando los kill-switches** (Tier 1):
  el budget dice *cuánto* a cada libro en régimen normal, el kill-switch dice *para* todo
  cuando el régimen se rompe.
