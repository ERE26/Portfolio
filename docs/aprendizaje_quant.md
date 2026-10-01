# Aprendizaje quant: cómo seleccionar mejor (calidad > cantidad)

Síntesis honesta de la investigación sobre estrategias intradía de alta
selectividad, trading con noticias y sizing por convicción. Principio rector,
que coincide con la evidencia: **el edge defendible no es la velocidad ni los
datos exóticos, es la SELECTIVIDAD disciplinada** — elegir mejor qué operar y
cuánto, y no operar casi nada el resto del tiempo.

> Aviso: casi todos los retornos de 4 cifras de los papers asumen costes
> optimistas (sin slippage/spread/locate). El edge real es una fracción. Lo que
> SÍ es robusto está abajo.

## Lo más robusto (hazlo primero, alto impacto / bajo coste)

1. **Costes realistas + tope diario de operaciones.** Modela spread (½ del
   bid-ask) + slippage (función de ATR y tamaño vs ADV) + comisión, y mide el EV
   por trade NETO. La mayoría de "edges" de backtest mueren aquí. Un tope duro de
   operaciones/día fuerza a jerarquizar. Gratis y de máximo impacto.
2. **Ranking DURO top‑N por RVOL de apertura, no umbral blando.** El hallazgo
   robusto del paper de Zarattini-Barbon-Aziz (2024, SSRN 4729284): en el ORB de
   acciones, **~95% del alfa viene de QUÉ acción eliges** (relative volume /
   "stock in play"), no de la regla de entrada. ORB sin filtrar: Sharpe 0.48.
   ORB sobre el top‑20 por RVOL de apertura: Sharpe 2.81. El escalado por RVOL es
   monótono. → Pasar de "¿supera el umbral?" a "¿es de las K mejores hoy?". Esto
   ES el principio calidad>cantidad.
3. **Noticias como GATE de catalizador, NO como señal direccional.** El
   sentimiento de titular ya está descontado (latencia) y es ambiguo ("beat but
   lowers guidance"). Uso honesto: responder "¿hay una razón real para este
   movimiento?". Si RVOL alto SIN catalizador → desconfía (ruido/manipulación).
   Fuentes: Alpaca News (gratis, streaming, Benzinga) + Finnhub (earnings
   calendar). Scorer propio Loughran-McDonald / FinBERT (el genérico tipo VADER
   falla en finanzas).
4. **Time-stops y revisar que el management no corta la cola derecha.** Los
   sistemas de momentum (skew positivo) viven de DEJAR CORRER. Breakeven agresivo
   y scale-outs tempranos suben el win-rate pero REDUCEN la esperanza. Backtest
   SIEMPRE cada regla contra su ausencia.

## Medio plazo (impacto moderado-alto)

5. **Ensemble de confluencia + score EV, sizing por convicción (½-Kelly capado).**
   Exigir que varias señales POCO correlacionadas coincidan (RVOL + sobre-VWAP +
   catalizador fresco + fuerza relativa vs SPY + hora adecuada). Operar top‑K por
   EV neto. Es selectividad por construcción. Ojo: pocas señales, pesos simples,
   validar walk-forward (RVOL y gap están correlacionados → cuentan casi como una).
6. **ORB de 5m** (stop 10% ATR, salida EOD) sobre el watchlist YA filtrado, como
   complemento/alternativa a la entrada VWAP.
7. **Filtro de régimen simple** (día tendencial vs rango vía realized vol / ADX /
   breadth): apagar momentum en días malos. Selectividad a nivel de día.
8. **Trailing Chandelier/ATR** (máx − 3·ATR) para dejar correr ganadores en vez
   de TP fijo: captura el skew positivo.

## Cuando haya muestra de trades (mayor payoff en selectividad)

9. **Meta-labeling (López de Prado, triple-barrier, purged CV + embargo).** El
   modelo primario (VWAP+RVOL/ORB) da la DIRECCIÓN; un segundo modelo decide
   SÍ/NO operar y cuánto. Es literalmente "operar menos pero mejor" hecho modelo.
   **Ya está en el sistema** (`patrones/meta.py`, calibrado isotónico OOF).

## Escéptico sobre (poco robusto / sobreajustado)

- OFI con solo OHLCV (proxy pobre; el alfa de microestructura necesita L2).
- Sentimiento de titular como señal direccional (descontado, latencia).
- Breakeven agresivo y scale-outs tempranos (suelen bajar la esperanza).
- PEAD multi-día (erosionado; el intradía algo más vivo pero edge neto modesto).
- Cualquier resultado de 4 cifras (costes irreales; el short side es casi
  inejecutable en los small-caps que más se mueven).

## Cómo lo aplica este repo

- Selección intradía (`gestionar.py`, `operativa_intradia.py`): ranking por
  convicción con gate de catalizador (noticias) y reinversión SOLO en candidatos
  fuertes; umbral afinado por el aprendizaje real (`registro.py`).
- Aprendizaje de lo real (`registro.py`): cada entrada se anota con sus rasgos
  (RVOL, dist_vwap, ret_sesión, ATR, sentimiento, hora) y su desenlace; con ≥25
  ops se estima el umbral de RVOL que maximiza el acierto y se aplica.
- Robustez (`patrones/robustez.py`): DSR + PBO (CSCV) para no auto-engañarse.
- Costes realistas (`patrones/costes.py`): el coste en riesgo (ATR) escala inverso
  a la volatilidad; integrado en el backtest con `aplicar_sistema(coste_atr=
  "realista")`. Diagnóstico empírico en `docs/analisis_costes_intradia.md`
  (figura 26): en blue chips el edge 15m no sobrevive a costes salvo por
  CONFLUENCIA (RVOL alto **y** meta alto) — el edge está en los *stocks in play*,
  no en las large-caps.

Referencias: Zarattini-Barbon-Aziz (2024, SSRN 4729284 y 4416622); López de
Prado, *Advances in Financial Machine Learning* (2018); Bernard & Thomas
(PEAD); Barber & Odean (2000, "Trading Is Hazardous to Your Wealth"); Tetlock
(2007); Loughran-McDonald; FinBERT (Araci 2019); Cont-Kukanov-Stoikov (2014, OFI).
