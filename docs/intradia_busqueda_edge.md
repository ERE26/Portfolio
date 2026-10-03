# Búsqueda de edge intradía — backtest Monte-Carlo OOS (oct 2026)

Tras el forense del 2026-10-02 (37% acierto, −$550, expectancy −0,48R) se construyó un
backtest Monte-Carlo (`backtest_intradia.py`, `backtest_estrategias.py`) que coge tickers y
días **al azar** y simula la estrategia **real** barra a barra, point-in-time (sin
look-ahead), con el bracket TP/SL, break-even, time-stop y slippage. Se probaron varias
lógicas de entrada sobre la misma muestra para aislar el efecto de la entrada.

## Resultado (pooled, ~4.750 ticker-días, 2 semillas independientes)

| estrategia | ops | exp TP fijo | exp + trailing | acierto | consistente |
|---|---|---|---|---|---|
| **orb** (ruptura rango apertura, stop al mín. del rango) | 213 | +0,016R | **+0,038R** | 51% | **sí (+0,047/+0,030)** |
| gap_orb (gap≥2% + ruptura) | 52 | +0,011R | +0,026R | 56% | no (+0,137/−0,056) |
| gap_and_go (gap≥3% + continuación) | 71 | +0,093R | +0,352R | 33% | no (ruidoso) |
| pullback_vwap (retroceso al VWAP) | 312 | −0,265R | −0,286R | 23% | pierde |
| momentum genérico (la original) | 456 | −0,275R | −0,377R | 22% | pierde |

## Conclusiones

1. **El momentum genérico sobre VWAP (la estrategia original) NO tiene edge** — pierde
   −0,28R de forma fiable en 456 trades OOS, las dos semillas. Confirmado y descartado.
2. **El pullback al VWAP tampoco** (−0,27R, 312 trades).
3. **El catalizador (gap) añade señal pero es ruidoso solo** (gap_and_go / gap_orb: positivos
   de media pero inconsistentes entre semillas y con poca muestra). No fiable por sí mismo.
4. **El ÚNICO edge positivo y consistente es el ORB** (Opening-Range Breakout) con dos
   ingredientes clave que el original no tenía:
   - **stop al MÍNIMO del rango de apertura** (riesgo propio del setup, no un ATR genérico):
     subió el ORB de −0,63R (ATR) a ~0R.
   - **trailing stop** (dejar correr la ganadora en vez de caparla): de +0,016R a +0,038R.
5. **Magnitud honesta:** +0,038R/op es pequeño pero real y robusto (neto de slippage). Con
   sesgo de supervivencia (muestreo del universo líquido actual), en real será
   ~break-even-a-ligeramente-positivo. Es un salto cualitativo frente al −0,3R del original.

## Qué es el ORB operable (para cablear)

- Universo: mid-caps líquidas en juego (<$20B), como ya filtra el pool.
- Rango de apertura = máx/mín de los primeros 30 min (2 velas de 15m).
- Entrada: el precio rompe el MÁXIMO del rango, dentro de las primeras ~2h, con RVOL alto,
  fuerza relativa > mercado y momentum reciente > 0.
- Stop inicial: el MÍNIMO del rango de apertura.
- Salida: trailing a ~2·ATR bajo el máximo una vez en +1R (dejar correr); time-stop duro a
  2,5h; cierre en EOD. Sin TP fijo.

## Avisos (por qué no es una máquina de imprimir)

Sesgo de supervivencia; datos IEX (no SIP); el suelo de convicción (z-score de lote) no se
replica por-barra; una entrada por ticker-día. El edge es fino: la ejecución y el slippage
importan. Apto para paper; en real exigiría ejecución ajustada y vigilancia de la auditoría
de stops en vivo.
