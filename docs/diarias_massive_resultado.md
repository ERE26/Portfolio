# Estrategias DIARIAS/swing sin sesgo de supervivencia (Massive EOD) — veredicto honesto

**Fecha:** 2026-10-05 · **Datos:** Massive EOD (plan Starter, ~5 años + deslistados) ·
**Código:** `backtest_diarias_massive.py`, `universo_pit.py` · **Solo research/paper.**

## Qué se midió y por qué

El ORB intradía acaba de caer al quitar la supervivencia. Toca hacer lo mismo con las
estrategias DIARIAS, que nunca se habían medido limpias. Objetivo: ¿tienen edge REAL
sobre SPS (comprar-y-mantener SPY) una vez que dejamos de hacer trampa con el sesgo de
supervivencia?

**Encuadre honesto desde el principio (importante):** de las tres estrategias, solo UNA
tiene un universo de acciones donde el sesgo de supervivencia muerde. Las otras dos
operan **cestas FIJAS de ETFs** que siguen cotizando:

| Estrategia (libro) | Universo | ¿Sesgo de supervivencia? |
|---|---|---|
| TSMOM (Libro C) | 7 ETFs cross-asset FIJOS (SPY,QQQ,IWM,DIA,TLT,GLD,HYG) | **No** — ninguno se deslista |
| Rotación Europa (Libro D) | 10 ETFs de país FIJOS (EWG,EWQ,…) | **No** — ninguno se deslista |
| Residual momentum | acciones US cribadas por liquidez | **Sí** — aquí es donde importa |

Para TSMOM y rotación no hay "fantasma de supervivencia" que exorcizar: la pregunta real
es simplemente **¿baten a SPY con datos limpios?**. Solo para residual momentum tiene
sentido comparar universo point-in-time (con deslistados) vs universo de hoy (sesgado) y
medir el *haircut*. Y ahí el haircut es brutal.

### Universo point-in-time (para residual momentum)

`universo_pit.py` descarga el endpoint *grouped* de Massive (todo el mercado US de cada
sesión, deslistados incluidos) y guarda, por sesión, el top-1500 de acciones comunes por
volumen en dólares. Resultado: **1.246 sesiones (2021-10 → 2026-10), 6.243 tickers
únicos** — muy por encima de los ~1.500/día, lo que confirma que captura el *churn*
(entran y salen nombres; los deslistados están, hasta su baja). Un valor que sale del
universo se liquida a su último cierre; no desaparece mágicamente.

### Método (igual de estricto para todas)

Point-in-time sin look-ahead (señal con datos hasta t, retorno de t a t+1), rebalanceo
**mensual**, coste por turnover (10 bps ETFs / 20 bps acciones), warmup descontado (no se
cuentan los meses sin histórico suficiente). Métricas: CAGR, Sharpe anualizado, maxDD,
Calmar, % meses positivos y **IC95 del Sharpe por bootstrap**. Benchmark: SPY buy&hold en
el MISMO periodo de cada estrategia.

> **Aviso de muestra:** Massive Starter da ~5 años, y estas estrategias gastan ~1 año de
> *lookback* (TSMOM necesita 252 sesiones; residual momentum, una ventana de regresión).
> El periodo efectivo medible es ~3 años (n≈33-35 meses). Los IC95 son ANCHOS a propósito:
> con 3 años no se puede afirmar gran cosa con rotundidad, y el documento lo refleja.

---

## Resultados

### 1) TSMOM — Libro C (time-series momentum, cesta fija de ETFs)

Periodo activo **2022-11 → 2026-09 (n=33 meses)**. Sin sesgo de supervivencia (cesta fija).

| | CAGR | Sharpe | maxDD | Calmar | +meses | IC95 Sharpe |
|---|---|---|---|---|---|---|
| **TSMOM** | +10.5% | 1.13 | **−6.7%** | 1.56 | 61% | **[−0.03, 2.49]** |
| SPY B&H | +18.1% | 1.32 | −8.8% | 2.06 | 61% | — |

**Veredicto: NO bate a SPY.** Menos CAGR (10.5% vs 18.1%), menos Sharpe (1.13 vs 1.32) y
menos Calmar. Su único punto a favor es el **drawdown más bajo** (−6.7% vs −8.8%): el
perfil defensivo esperado. Pero el IC95 del Sharpe **incluye el 0** → con n=33 ni siquiera
se distingue de cero. El TSMOM es "crisis alpha" (gana en caídas prolongadas), y en el
régimen 2022-2026 (alcista, sin crisis larga) esa póliza de seguro no se cobró. **Se
mantiene solo como diversificador de drawdown, NO por retorno esperado sobre SPY.**

### 2) Rotación Europa — Libro D (momentum transversal, cesta fija de ETFs de país)

Periodo activo **2022-09 → 2026-09 (n=35 meses)**. Sin sesgo de supervivencia (cesta fija).

| | CAGR | Sharpe | maxDD | Calmar | +meses | IC95 Sharpe |
|---|---|---|---|---|---|---|
| **Rotación** | +18.0% | 1.15 | −14.4% | 1.25 | 57% | **[0.02, 2.51]** |
| SPY B&H | +16.1% | 1.07 | −8.8% | 1.83 | 60% | — |

**Veredicto: empate técnico con SPY, con matiz importante.** Gana a SPY por poco en CAGR
(18.0 vs 16.1) y Sharpe (1.15 vs 1.07), pero pierde en drawdown (−14.4% vs −8.8%) y en
Calmar (1.25 vs 1.83): más retorno a costa de más riesgo. El IC95 del Sharpe apenas
excluye el 0. **Caveat clave: batir a SPY no es alfa.** Buena parte del "edge" es **beta
europea + divisa** en un periodo (2022-2026) en que Europa y el EUR tiraron fuerte. El
test que importa es contra su PROPIO benchmark (VGK, renta variable europea), donde el
propio `validar_libroD.py` ya daba DSR 0.89 (<0.95, marginal). **Es un diversificador
modesto, no alfa heroica; no subir tamaño por este resultado.**

### 3) Residual momentum — aquí SÍ muerde la supervivencia

Periodo **≈2023-12 → 2026-09 (n=34 meses)**. Ventana de regresión FF3 acortada a 24m
(dentro de la rejilla validada original) por los 5 años de datos. Universo sin-supervivencia
= unión de los top-150 líquidos de cada mes (**745 acciones**, deslistados incluidos);
universo sesgado = los 150 supervivientes líquidos de HOY aplicados hacia atrás.

| versión | CAGR | Sharpe | maxDD | Calmar | +meses | IC95 Sharpe |
|---|---|---|---|---|---|---|
| **Long-only, SIN supervivencia (745)** | **+27.8%** | **1.45** | −9.6% | 2.91 | 68% | [0.30, 2.64] |
| Long-only, SESGADO (150 supervivientes) | +64.2% | 1.83 | −18.8% | 3.41 | 68% | [0.72, 3.15] |
| L/S, SIN supervivencia | +5.4% | 0.44 | −19.2% | 0.28 | 50% | [−0.77, 1.64] |
| L/S, SESGADO | +17.9% | 0.74 | −28.6% | 0.63 | 53% | [−0.45, 1.75] |
| **SPY B&H (mismo periodo)** | +23.9% | **1.77** | −8.0% | 3.00 | 71% | — |

**El haircut de supervivencia (long-only):**

| métrica | sesgado | sin superv. | haircut |
|---|---|---|---|
| CAGR | +64.2% | +27.8% | **−36.4 puntos** |
| Sharpe | 1.83 | 1.45 | **−0.38** |
| Calmar | 3.41 | 2.91 | −0.50 |

**Veredicto:**
- **L/S: sin edge.** El IC95 del Sharpe incluye el 0 en ambos universos. Confirma el
  veredicto negativo que ya tenía el repo (`validar_residual_momentum.py`).
- **Long-only: el número bonito es un ESPEJISMO de supervivencia.** El Sharpe 1.83 /
  CAGR 64% del universo sesgado se desploma a **Sharpe 1.45 / CAGR 28%** con datos
  limpios — y eso lo deja **por DEBAJO del Sharpe de SPY (1.77)** y con CAGR casi igual.
  El long-only no es alfa: es momentum-beta que **no supera a comprar el índice** una vez
  que dejas de contar solo los ganadores. **No se cablea** (coincide con la decisión
  previa de no desplegarlo).

---

## Conclusión global (honesta)

1. **Ninguna de las tres bate a SPY de forma robusta (risk-adjusted) en 2022-2026.** TSMOM
   pierde; rotación empata con peor drawdown; residual momentum long-only parecía ganar
   pero era supervivencia.
2. **El sesgo de supervivencia, donde aplica (residual momentum), vale ~36 puntos de CAGR
   y ~0.4 de Sharpe.** Es gigantesco: justo lo que el ORB nos acaba de enseñar en intradía,
   repetido en diario. Cualquier backtest de acciones sobre el universo de hoy está inflado.
3. **TSMOM y rotación no tienen sesgo que quitar** (cestas fijas de ETFs), así que su
   número ya era "limpio" — y aun así no superan a SPY en este régimen. Su justificación es
   **diversificación / crisis-alpha**, no retorno esperado; mantienen sentido dentro de una
   cartera multi-libro con presupuesto de riesgo, no en solitario contra el índice.
4. **Limitación honesta:** ~3 años de muestra (los 5 de Massive menos el warmup), régimen
   alcista específico, IC95 anchos. Esto NO valida ni invalida de forma definitiva; mide lo
   que se puede medir con los datos disponibles y dice la verdad sobre su incertidumbre. Con
   más historia (vendor con 15-20 años + deslistados) el veredicto podría afinarse,
   especialmente para TSMOM (que necesita incluir una crisis larga para lucir).

**En una frase:** las estrategias diarias no son una máquina de batir al índice; son
diversificadores cuya única defensa real (menor drawdown en TSMOM) no se pagó en un
mercado sin crisis, y el único que parecía alfa (residual momentum) era sesgo de
supervivencia disfrazado.
