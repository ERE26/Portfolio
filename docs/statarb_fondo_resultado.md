# Stat-arb intradía "como los fondos": ¿edge real o spread bid-ask?

**Pregunta central.** El intradía direccional ya murió en este repo (todo sub-coste).
El pairs/stat-arb dio *señal* pero sub-coste a mercado. Los fondos ganan justo ahí vía
escala, ejecución pasiva (cobrar el spread + rebate) y neutralidad de factores. Así que:

> ¿Existe un stat-arb intradía con **Sharpe robusto** que **sobreviva al coste de una
> cuenta retail (TAKER, cruza a mercado)**, o solo vive con **ejecución de fondo
> (MAKER, provee liquidez y cobra rebate)**?

**Respuesta corta (honesta): NO hay edge accesible.** Lo que parece alfa de reversión
intradía es, medido con rigor, el **rebote bid-ask** (microestructura), no capacidad
predictiva. "Sobrevive" en el escenario MAKER solo porque el escenario MAKER *es,
literalmente, cobrar el spread*. Para nosotros (TAKER) es catastrófico. El pairs en
cesta no tiene ni siquiera señal bruta significativa.

---

## Metodología (idéntico rigor en ambas estrategias)

- **Datos:** Massive (plan Starter): barras de 5-min (recortadas a sesión regular
  09:30–16:00 ET) + `grouped` diario de TODO el mercado US. Sin sesgo de supervivencia.
- **Universo POINT-IN-TIME** (`universo_pit.py`): top por dólar-volumen *as-of* la
  ventana de formación, filtrado a **acciones comunes** (type=CS: activas + deslistadas
  posteriores) para quitar ETFs, apalancados y símbolos de prueba sin reintroducir
  survivorship.
- **Sin look-ahead:** beta vs SPY estimada SOLO con la ventana de formación (anterior a
  la de prueba); la señal en la barra `t` usa retornos que terminan en `t` y cobra de
  `t`→`t+k` (no se solapan: la barra que decide no es la que paga); z-score rolling
  causal.
- **Ventanas:** formación 2024-01-02→2024-02-29 (41 sesiones); prueba
  2024-03-01→2024-05-24 (60 sesiones). **3 semillas** = submuestras de 40 días.
- **Bootstrap IC95 por bloques-día** (remuestrea días con reemplazo; respeta la
  autocorrelación intradía). 2000 réplicas.
- **Dos escenarios de coste** (contabilidad de turnover `coste = c_lado · turnover`):
  - **TAKER** (retail, a mercado): `c_lado = +3,0 bps`.
  - **MAKER** (fondo, límite pasivo en la banda + rebate): `c_lado = −0,25 bps`
    (relleno al precio de la señal; **optimista**, ver caveats).
- **Métrica clave — test de rebote bid-ask (`entry_lag`):** se repite TODO metiendo un
  retardo de 1 barra (5 min) entre la señal y el relleno. Si el edge desaparece con el
  retardo, no era alfa predecible y tradeable: era el rebote del último trade entre bid
  y ask. **Este test es el que decide.**

> Nota sobre maxDD: es sobre la suma acumulada de retornos por periodo (no compuesta).
> En TAKER los valores >100% solo significan sangrado monótono sostenido sobre los
> periodos muestreados, no una quiebra de una cuenta concreta.

---

## ESTRATEGIA A — Cross-sectional intraday reversion

Cartera **dólar- y beta-neutral** sobre 100 nombres líquidos: largo el decil de residuo
más bajo (barato), corto el más alto (caro), equal-weight, rebalanceo intradía, **flat al
cierre**. Residuo = retorno de L barras − β·retorno de SPY, neutralizado en sección
cruzada. (`backtest_xsec_intradia.py`)

### Dependencia del horizonte (pooled, 100 nombres, 60 días)

El horizonte lo es todo. La "reversión" solo aparece a 5 minutos; a 30 min (lo que
sugería el encargo) no hay nada.

| Config (señal/hold) | BRUTO exp | BRUTO Sharpe | ¿señal bruta? |
|---|---|---|---|
| **5-min / 5-min** (look1/rebal1) | **+1,14 bps** | **+20,2** | sí, pero → ver test rebote |
| 5-min / 10-min (look1/rebal2) | +0,83 bps | +7,7 | débil |
| 30-min / 30-min (look6/rebal6) | **−0,03 bps** | −0,1 | **no, muerta** |

### Config principal 5-min/5-min — TAKER vs MAKER (3 semillas + pooled)

Turnover 1,67/periodo · ~76 holdings/día · 100 nombres · 60 días.

| Escenario | Semilla | Sharpe (anual.) | Expectancy | IC95 expectancy | maxDD |
|---|---|---|---|---|---|
| **BRUTO** | s1 | +18,95 | +1,021 bps | [+0,751, +1,287] | −1,2% |
| | s2 | +20,67 | +1,169 bps | [+0,928, +1,417] | −1,3% |
| | s3 | +20,61 | +1,167 bps | [+0,917, +1,415] | −1,1% |
| | **pooled** | **+20,23** | **+1,144 bps** | **[+0,947, +1,350]** | −1,3% |
| **TAKER** | s1 | −74,15 | −3,989 bps | [−4,241, −3,740] | −122,7% |
| | s2 | −68,09 | −3,856 bps | [−4,083, −3,621] | −118,5% |
| | s3 | −68,23 | −3,862 bps | [−4,092, −3,632] | −118,7% |
| | **pooled** | **−68,58** | **−3,879 bps** | **[−4,064, −3,687]** | −179,1% |
| **MAKER** | s1 | +26,69 | +1,438 bps | [+1,167, +1,705] | −1,1% |
| | s2 | +28,07 | +1,588 bps | [+1,345, +1,837] | −1,1% |
| | s3 | +28,00 | +1,586 bps | [+1,335, +1,836] | −0,9% |
| | **pooled** | **+27,62** | **+1,563 bps** | **[+1,365, +1,770]** | −1,1% |

### El test que lo decide — rebote bid-ask (entrada 1 barra = 5 min después)

| | BRUTO exp | BRUTO IC95 | MAKER exp | MAKER IC95 |
|---|---|---|---|---|
| **lag 0** (entra en la barra de la señal) | **+1,144 bps** | [+0,95, +1,35] | +1,563 bps | [+1,37, +1,77] |
| **lag 1** (entra 5 min después) | **−0,308 bps** | [−0,56, −0,06] | +0,110 bps | [−0,14, +0,36] |

**Con un solo retardo de 5 minutos el edge bruto desaparece y se vuelve negativo; el
MAKER cae a +0,11 bps con IC que cruza cero (no significativo).**

### Veredicto A (honesto)

- Hay una "reversión" cross-sectional a 5 min con Sharpe bruto absurdamente alto
  (+20) — y ese Sharpe disparatado es justo **la firma de un artefacto de
  microestructura**, no de una estrategia (el stat-arb real de HF vive en Sharpe 2–4).
- El test de retardo lo confirma: **el 100% del edge es rebote bid-ask.** Al entrar
  5 min después, el componente predictivo es cero/negativo.
- Por eso "sobrevive y mejora" en MAKER: el escenario MAKER modela rellenar al precio de
  la señal cobrando el spread → estás **cobrando exactamente el rebote** que el test
  demuestra que es toda la señal. No es alfa; es el spread.
- **TAKER: desastre** (−3,9 bps/periodo; pagas el spread que el MAKER cobra).
- A 30 min (el horizonte que pedía el encargo) **no hay nada** (−0,03 bps).

> **A no es un edge accesible.** Lo que hay es el spread bid-ask, que solo captura quien
> *provee* liquidez (prioridad de cola, colocation, rebates) — infra de fondo, no de
> retail. Y aun para el fondo, el componente NO-rebote (predecible) es ≈0 a estos
> horizontes en estos nombres líquidos.

---

## ESTRATEGIA B — Pairs / stat-arb en cesta (mejora del pairs existente)

Extiende el pairs intradía ya en el repo (`backtest_pairs_massive.py`, pares de ETFs,
que dio "señal real pero sub-coste a mercado") con las dos mejoras que pedía el encargo:
**(1) cesta neteada** de pares (cartera, no par suelto) y **(2) ejecución MAKER** además
de la taker. Para no pisar aquel fichero, esta versión vive en `backtest_pairs_cesta.py`
y opera **acciones** (no ETFs) elegidas del universo PIT.

Cesta de pares (no par suelto) seleccionados POINT-IN-TIME por correlación de retornos
diarios de formación + hedge ratio OLS. Señal: z-score del spread (rolling causal),
entra |z|>2,0, sale |z|<0,5, stop |z|>3,5, **flat al cierre**. Las posiciones de los 9
pares se netean en un vector de pesos por nombre → Sharpe de CARTERA con diversificación.
TAKER y MAKER + test de rebote. (`backtest_pairs_cesta.py`)

**9 pares** (correlación formación, 1 nombre por par): GOOGL/GOOG, XOM/CVX, META/AMZN,
JPM/BAC, COIN/MARA, NVDA/AVGO, MSFT/CRM, AMD/AMAT, QCOM/MU. 985 trades, turnover 0,046.

### TAKER vs MAKER (3 semillas + pooled)

| Escenario | Semilla | Sharpe | Expectancy | IC95 expectancy | maxDD |
|---|---|---|---|---|---|
| **BRUTO** | s1 | −0,41 | −0,009 bps | [−0,120, +0,097] | −1,8% |
| | s2 | +1,48 | +0,032 bps | [−0,075, +0,137] | −1,1% |
| | s3 | +2,76 | +0,061 bps | [−0,051, +0,172] | −1,2% |
| | **pooled** | **+0,77** | **+0,017 bps** | **[−0,078, +0,105]** | −1,7% |
| **TAKER** | s1 | −6,44 | −0,149 bps | [−0,260, −0,041] | −4,9% |
| | s2 | −5,11 | −0,111 bps | [−0,218, −0,004] | −3,7% |
| | s3 | −3,62 | −0,080 bps | [−0,193, +0,032] | −2,7% |
| | **pooled** | **−5,45** | **−0,122 bps** | **[−0,215, −0,033]** | −6,0% |
| **MAKER** | s1 | +0,09 | +0,002 bps | [−0,108, +0,109] | −1,8% |
| | s2 | +2,03 | +0,044 bps | [−0,063, +0,149] | −1,1% |
| | s3 | +3,29 | +0,073 bps | [−0,039, +0,184] | −1,2% |
| | **pooled** | **+1,28** | **+0,029 bps** | **[−0,066, +0,117]** | −1,6% |

### El espejismo del "+1,5 bp"

Mediana por trade = **+1,87 bps**, pero **media por trade ≈ −0,01 bps** (981 trades,
win 56%). Es decir: el trade típico gana ~1,9 bps (de ahí el "+1,5 bp señal real" que se
vio antes), pero la **cola izquierda** (stop-outs cuando el spread sigue divergiendo /
el par se rompe) se come toda la media. **La expectancy harvestable es cero**, aun en
bruto.

- Barrido de parámetros (W∈{39,78}, entra∈{1,5 / 2,0 / 2,5}): en **todos**, el IC95 de
  la expectancy bruta **cruza cero** → no hay señal bruta significativa.
- Diversificar 9 pares **bajó la varianza** (Sharpe cerca de 0 en vez de muy negativo),
  pero no crea una media que no existe.
- Test de rebote (lag 1): BRUTO cae a −0,002 bps, MAKER +0,011 bps (IC cruza cero). Igual
  que A: lo poco que había era microestructura.

### Veredicto B (honesto)

> **B no mejora a positivo.** La diversificación en cesta y la ejecución maker no
> convierten el pairs en rentable porque **no hay señal bruta** que rescatar: la media
> por trade es ~0 y el IC95 cruza cero en todas las configuraciones. El "+1,5 bp" era la
> **mediana** de un trade típico, no la expectancy. TAKER pierde claramente.

---

## Conclusión global

| | ¿Señal bruta real? | TAKER (retail) | MAKER (fondo) | ¿Edge accesible? |
|---|---|---|---|---|
| **A** xsec reversion 5-min | **No** (es rebote bid-ask; muere con lag 1) | −3,9 bps, Sh −69 | +1,6 bps, Sh +28 *(= cobrar el spread)* | **No** |
| **A** xsec 30-min (lo pedido) | No (−0,03 bps) | pierde | ≈0 | No |
| **B** pairs en cesta | **No** (media ≈0, IC cruza 0) | −0,12 bps, Sh −5,5 | +0,03 bps, Sh +1,3 (no signif.) | **No** |

**Respuesta a la pregunta central.** No existe —en estos datos, universo y horizontes— un
stat-arb intradía con Sharpe robusto que sobreviva al coste de ejecución retail (taker).
El único escenario en el que "algo funciona" es el MAKER de la Estrategia A, y está
**confirmado que ese algo es el spread bid-ask**, no alfa: desaparece al introducir 5 min
de retardo. Cobrar el spread de forma sistemática **no es accesible para nosotros**:
exige ser *proveedor de liquidez* con prioridad de cola, colocation y rebates (infra de
market-maker/fondo HF). Como cuenta retail somos tomadores de liquidez → pagamos ese
spread, y por eso todo da negativo.

**Implicación operativa (coherente con el resto del repo): NO cablear.** Confirma, con
rigor PIT y doble escenario de coste, que el intradía —también el stat-arb -"como los
fondos"— no es una fuente de edge para una cuenta retail que cruza a mercado. El dinero
de este juego está en el lado maker/rebate, que no es nuestro lado.

### Caveats (honestidad > resultado)

1. El escenario MAKER es **optimista**: supone relleno pasivo al precio de la señal al
   100%. En realidad el límite pasivo sufre *adverse selection* (te llenan justo cuando
   el precio se te va en contra) y no siempre entra. El MAKER realista es aún peor que lo
   reportado, lo que refuerza el veredicto.
2. Muestra deliberadamente pequeña (3 meses, 100 nombres top-liquidez, 40 días/semilla)
   para terminar y cachear. Los fondos usan cientos/miles de nombres; más escala baja la
   varianza pero **no cambia la media** (y la media bruta es ≈0 salvo el rebote).
3. Sin neutralidad sectorial explícita (solo mercado/beta). Añadirla reduciría varianza,
   no crearía señal donde el test de rebote muestra que no la hay.
4. Datos 5-min OHLC (no tick/quotes). El test de `entry_lag` es precisamente el control
   que aísla el efecto microestructura sin necesitar el libro.

---

*Infra: `universo_pit.py`, `fuentes/massive.py` (`barras_intradia`, `grouped_diario`),
`backtest_xsec_intradia.py`, `backtest_pairs_cesta.py` (extiende el existente
`backtest_pairs_massive.py`). Solo research/paper.*
