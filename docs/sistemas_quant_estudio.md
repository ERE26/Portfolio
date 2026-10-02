# Estudio de sistemas cuant: StatArb, volatilidad (GARCH), trend following y ML

Estudio honesto de los cuatro bloques de métodos cuantitativos, qué aporta cada uno,
y **cómo está (o no) aplicado en este sistema**. Criterio de siempre: no basta con
que una técnica sea famosa; tiene que sobrevivir a costes, a walk-forward y a
DSR/PBO, y aportar algo que no tengamos ya.

---

## 1. Arbitraje estadístico (StatArb) — *market-neutral, reversión a la media*

**Qué es.** Busca dos (o más) activos cuyos precios se mueven juntos a largo plazo
(**cointegrados**): aunque cada uno sea no estacionario, una combinación lineal
(el *spread*) sí revierte a su media. Cuando el spread se desvía (z-score alto), se
vende el caro y se compra el barato apostando a que vuelve. Modelo del spread:
Ornstein-Uhlenbeck (reversión con una *media-vida*).

**Por qué interesa.** No por su Sharpe (decaído por saturación), sino por su
**correlación ≈ 0 con el mercado**: es un flujo market-neutral. Igual que el TSMOM,
su valor es diversificar, no heroísmo.

**Estado en nuestro sistema.** Probado (scan rápido de pares): **no hay pares con
cointegración limpia** en el universo actual con una media-vida operable. Decisión
honesta: **aparcado**, no descartado. El scan ciego de pares es trampa de
sobreajuste (con 100 activos pruebas ~5.000 pares → algunos "cointegran" por azar);
hay que hacerlo con **lógica económica** (mismo sector/subsector) y validar
out-of-sample. Es el candidato market-neutral natural, pero solo si aparece un par
con cointegración estable y media-vida de días, no meses.

**Riesgo clave.** La cointegración se **rompe** (cambia un fundamental, una fusión):
el spread que "siempre revierte" deja de revertir y la pérdida es ilimitada si no
hay stop. Por eso StatArb serio lleva stop por ruptura + límite de tiempo.

---

## 2. Modelos de volatilidad (GARCH, EWMA) — *el riesgo no es constante*

**Qué es.** La volatilidad llega en **rachas** (clustering: a un día agitado le
sigue otro). GARCH(1,1) modela la varianza de hoy como ω + α·(shock de ayer)² +
β·(varianza de ayer), capturando esa persistencia y la reversión de la vol a su
media. EWMA (RiskMetrics, λ=0,94) es el caso restringido sin media de largo plazo:
barato y robusto. Black-Scholes usa la vol para valorar opciones (no operamos
opciones, así que esa parte no aplica a nuestra ejecución).

**Por qué interesa — y es DIRECTAMENTE útil para nosotros.** La vol condicional
alimenta tres cosas que ya hacemos o queremos hacer mejor:
1. **Sizing (vol-targeting):** tamaño ∝ 1/σ̂. Mejor σ̂ (condicional) = mejor control
   de riesgo que la vol histórica de ventana fija, que reacciona tarde.
2. **VaR (Value at Risk):** cuánto puedo perder en el 5% peor de los días. El
   VaR-GARCH es **reactivo al régimen**; el histórico se queda "pegado" al pasado.
3. **Selección (meta-modelo):** el régimen de volatilidad cambia la fiabilidad de un
   setup. Un breakout en vol alta y subiendo no vale lo mismo que en calma.

**Estado en nuestro sistema — NUEVO (`patrones/volatilidad.py`).** Implementado en
dos niveles a propósito:
- **Producción (numpy puro, sin dependencias):** `rasgos_vol_condicional()` da vol
  EWMA, ratio vs largo plazo, si la vol sube, y el retorno estandarizado. **Cableado
  como features del meta-modelo** (sección 4). Apto para el contenedor que arranca
  limpio cada ciclo.
- **Análisis/riesgo (`arch`):** GARCH(1,1) por máxima verosimilitud, pronóstico de
  vol y VaR paramétrico con t de Student (colas gordas). Ejemplo real (SPY, hoy):
  persistencia α+β ≈ **0,99** (media-vida ~86 días, típico de renta variable),
  VaR 95% GARCH-t **0,9%** vs histórico **1,65%** — GARCH ve el régimen tranquilo
  actual, el histórico no.

Referencias: Engle (1982, ARCH, Nobel 2003); Bollerslev (1986, GARCH); J.P. Morgan
RiskMetrics (1996); Harvey et al. (2018, vol targeting).

---

## 3. Seguimiento de tendencias (Trend Following / CTA) — *ya lo tenemos*

**Qué es.** Captura impulso direccional a medio/largo plazo en muchas clases de
activo con medias móviles y filtros; es el motor de los grandes CTA (Man AHL,
Winton, AQR Managed Futures).

**Estado en nuestro sistema — IMPLEMENTADO (Libro C, `operativa_tsmom.py` +
`patrones/tsmom.py`).** Es exactamente esto: **time-series momentum** sobre una
cesta cross-asset (renta variable, bonos, oro, crédito), con vol targeting de
cartera y rebalanceo mensual. Validado (DSR 0,995, corr 0,27 con el libro diario) y
**desplegado en paper**. Además, la **rotación por países europeos** (estudio de
ayer, `analisis_europa_2026-10.md`) es trend/momentum transversal — candidato a
Libro D. El trend following es el bloque donde estamos más avanzados.

---

## 4. Machine Learning — *mejorado AHORA, y en su sitio correcto*

**La honestidad primero (importa).** Las redes neuronales profundas como **predictor
de precio primario** son, para nosotros, una **trampa de sobreajuste**: con solo
OHLCV el R² real es diminuto, el *alpha decay* es rápido, y una red con millones de
parámetros "memoriza" ruido que no se repite. Casi toda la literatura seria (y
nuestra propia experiencia: HRP, low-vol, pairs → negativos honestos) apunta a que
lo que gana no es un modelo más grande, es **mejor selección y gestión de riesgo**.
No voy a montar una red neuronal que prometa predecir el mercado; sería venderte humo.

**El ML que SÍ funciona: meta-labeling (López de Prado).** El modelo primario
(memoria de casos tipo k-NN + GBM) da la **dirección**; un segundo modelo (GBM) mira
las características de esa señal y predice **P(la señal acierta)** → solo operamos
cuando esa probabilidad supera un umbral. Es "operar menos pero mejor" hecho modelo.
Ya estaba (`patrones/meta.py`) con calibración isotónica out-of-fold.

**La mejora de hoy: volatilidad condicional como features del meta.** Añadidas 4
variables de régimen de vol (EWMA, familia GARCH; sección 2) a las features del
meta: `vol_cond`, `vol_ratio`, `vol_subiendo`, `ret_estand`. Hipótesis: la
fiabilidad de un setup depende del régimen de volatilidad, y el meta puede
aprenderlo. Validado en walk-forward de dos etapas (memoria→meta) con pesos de
unicidad, comparando contra la baseline SIN estas features:

**Resultados (walk-forward 2 etapas, 5 años diario, 3 pliegues, multi-semilla):**

| Variante del meta | AUC (test) | ret. seleccionado (ATR) |
|---|---:|---:|
| Sin features de vol | 0,5123 | 0,0070 |
| Con las 4 de vol | 0,5158 | 0,0071 |
| **Solo vol_ratio + ret_estand (elegida)** | **0,5171** | **0,0074** |

- El efecto es **pequeño pero real y con mecanismo validado**: la mejora se concentra
  en **régimen de volatilidad ALTA** (AUC +0,010) y es nula en vol baja (+0,001) —
  exactamente la hipótesis (el régimen de vol cambia la fiabilidad de un setup).
- La ablación feature-a-feature mostró que **solo 2 de las 4 aportan**: `vol_ratio`
  (+0,0054) y `ret_estand` (+0,0023); `vol_cond` y `vol_subiendo` eran ruido.
- **Decisión parsimoniosa:** quedarse con las 2 que aportan y descartar las 2 de
  ruido. KEEP-2 bate tanto a "todas" como a "ninguna" en AUC y en retorno
  seleccionado, con menor varianza. Más señal, menos sobreajuste.

**Por qué este camino y no una red neuronal.** Porque es **medible, interpretable y
honesto**: cada feature nueva se valida contra su ausencia; si no sube el AUC/Brier
y el Sharpe de lo seleccionado, se descarta. Una red profunda daría un número bonito
en backtest y nos engañaría. El ML correcto aquí es el que **decide mejor cuándo NO
operar**, no el que promete adivinar el precio.

Referencias: López de Prado (2018, *AFML*, meta-labeling, cap. 3); Bailey &
López de Prado (2014, DSR); calibración isotónica (Zadrozny-Elkan 2002).

---

## Resumen: dónde estamos en cada bloque

| Bloque | Estado | Dónde |
|---|---|---|
| Trend following (CTA) | ✅ en vivo (paper) | Libro C TSMOM; rotación Europa (candidato D) |
| Volatilidad (GARCH/EWMA) | ✅ nuevo | `volatilidad.py`: features del meta + VaR/pronóstico |
| Machine Learning (meta-labeling) | ✅ mejorado | `meta.py` + features de vol condicional |
| StatArb (pairs) | 🔸 aparcado (honesto) | sin pares limpios hoy; candidato market-neutral |

La dirección es la correcta: **varios flujos descorrelacionados, riesgo controlado,
y ML en su sitio (selección, no adivinación)**. No hace falta nada exótico ni de
pago — hace falta disciplina y medir sin autoengaño.
