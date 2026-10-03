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

## Fase 2 (oct 2026): meta-labeling, features y cartera — el salto cuant

Sobre 10.074 breakouts ORB-5min reales (150 tickers, 24m):
- **Meta-modelo ML (López de Prado)** — NO mejora de forma robusta el ORB (logística peor;
  HGB +0.02R pero DSR 0.93/PBO 0.29, no concluyente). Confirma a **Carver**: la complejidad
  no paga cuando la señal base ya captura el edge.
- **Lo que SÍ descubrió el análisis (importancia de features): la ruptura CONFIRMADA.**
  Exigir que la vela CIERRE sobre el máximo del rango (ext_breakout>0): +0.61R vs +0.27R del
  baseline (entrada al máximo); las falsas rupturas (mecha) pierden -0.37R. Aguanta OOS.
  En el simulador realista (entrada al CIERRE de la vela confirmada) el edge es +0.26R/trade,
  CONSISTENTE en 3 semillas (+0.30/+0.23/+0.24), frente al no-confirmado +1.4R pero salvaje.
- **Cartera (vol-targeting + half-Kelly, top-N in-play/día):** Sharpe MUY sensible al coste de
  entrada. Con el haircut REALISTA (0.37R, = la diferencia entre entrada al máximo y entrada
  confirmada real): **Sharpe ~1.8, DSR 0.80 (no robusto al 0.95)**. El Sharpe 4 del haircut
  optimista es un espejismo. Antes de sesgo de supervivencia.

### Conclusión honesta
El ORB-5min-confirmado sobre stocks-in-play es un edge REAL y validado OOS (+0.26R/trade),
pero MODESTO y FRÁGIL a la ejecución: el Sharpe de cartera plausible es ~1.5-2 (no 4), y
cae por debajo de lo robusto si la ejecución es mala. Coincide con la literatura (Zarattini
"stocks in play" ~2.8 con sus propias salvedades; réplicas: break-even a ~2.2c/acción). Es un
salto enorme frente al momentum muerto (-0.3R), pero NO es una máquina de imprimir: su
viabilidad depende de la calidad de ejecución. Siguiente: modelar ejecución (limit en el
máximo), universo point-in-time (quitar supervivencia), filtro de régimen, y paper-trading
para medir el slippage REAL antes de nada.

## Fase 3 (oct 2026): régimen y otras familias — qué NO añade edge

Opciones 2 y 3 del plan, testeadas con el mismo rigor OOS:
- **Filtro de régimen (HMM/Hurst/SPY-trend):** NINGÚN corte mejora el ORB de forma robusta
  (ningún subconjunto supera DSR 0.95). 'SPY<MA50 mejor' (Sharpe 2.88) es específico del
  periodo, no fiable. Hurst del valor no separa. -> ORB se queda SIN filtro de régimen (Carver).
- **Mean-reversion intradía (fade de extensiones bajo VWAP en in-play):** PIERDE en 2 configs
  (k=1.5: -0.32R, 31%; k=2.5: -0.35R, 25%). Un in-play extendido bajo VWAP suele estarlo por
  mala noticia que continúa (navaja), no por sobreventa. El edge está en el momentum, no en
  contrariarlo.

### Mapa final del edge intradía (todo validado OOS)
  ORB-5min confirmado, stocks-in-play, stop OR-low, hold EOD ...... EDGE (+0.26R/trade; cartera Sharpe ~2 realista)
  momentum genérico 15m .......................................... muerto (-0.3R)
  pullback a VWAP ................................................ muerto
  gap-and-go solo ............................................... ruidoso/inestable
  meta-modelo ML (filtro) ....................................... sin lift robusto
  filtro de régimen (SPY/Hurst) ................................. sin lift robusto
  mean-reversion a VWAP ......................................... pierde

CONCLUSIÓN: el ORB-confirmado es el ÚNICO edge intradía robusto; todo lo demás testeado no
funciona o no lo mejora de forma fiable. Es el resultado típico de la metodología honesta
(LdP/Carver): la mayoría de ideas no sobreviven la validación OOS, y el superviviente es una
regla simple. El muro ahora es EJECUCIÓN (solo medible en paper) y SUPERVIVENCIA (datos de
pago). El backtest ha dado lo que podía dar.
