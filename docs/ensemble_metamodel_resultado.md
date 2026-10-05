# Meta-modelo ensemble intradía: ¿mezclar señales débiles cruza el coste?

**Fecha:** 2026-10-05 · **Datos:** Massive Starter (barras de 5-min, el máximo que sirve el
plan en intradía: 2022-01→2026-09) · **Solo research/paper.**

## Pregunta central

El intradía suelto ya murió en este repo: cada señal (reversión del z de pares, distancia a
VWAP, reversión de 1-5 barras, volumen/rango/RVOL, hora del día) es **individualmente débil o
sub-coste**. La hipótesis del encargo:

> ¿Un **META-MODELO** (meta-labeling de López de Prado: señal primaria = reversión; un modelo
> secundario decide **SI** operar) que opere **solo cuando la convicción es alta** —pocos
> trades, más edge por trade— puede **cruzar el coste** donde cada señal suelta no llega?

**Respuesta corta (honesta): NO para una cuenta retail (taker).** El meta-modelo *sí* aprende
a concentrarse en un ~0,1 % de los eventos con expectancy positiva, pero (1) bajo coste
**TAKER** realista nunca es robusto (DSR < 0,95, IC95 que cruza/roza cero, muestra diminuta), y
(2) lo que parece robusto lo es solo bajo coste **MAKER** —cobrar el spread—, y el **test de
retardo demuestra que ESE es todo el edge**: el componente predictivo bruto por evento se
evapora de **+0,36 bps a −0,02 bps al entrar 5 minutos después**. Es el mismo rebote bid-ask de
siempre, ahora visto a través del meta-labeling. La mezcla no crea alfa; concentra el rebote.

---

## Metodología (honestidad > resultado)

- **Datos survivorship-free:** universo **POINT-IN-TIME** (`universo_pit.formar`, top-40 por
  dólar-volumen *as-of* 2021-Q4, con deslistados incluidos — p.ej. LCID, MRNA, PYPL) + barras
  de minuto de Massive. **276.373 eventos**, 40 tickers, **1.190 sesiones (4,72 años)**,
  2022-01-03→2026-09-30.
- **Señal primaria (reversión idiosincrática):** `resid = ret_L(stock) − β·ret_L(SPY)`
  (L = 5 barras = 25 min), z-score causal sobre ventana de 78 barras. `|z| ≥ 1,5` dispara un
  candidato a **revertir** (z > 0 = subió de más → corto; z < 0 → largo). Dirección balanceada
  (49 % largo / 51 % corto).
- **Etiqueta TRIPLE-BARRERA (LdP):** barreras ±1·σ (σ = vol de 1 barra causal) y barrera
  vertical 12 barras o EOD. `y = 1` si el trade primario gana **bruto**. Hold real: **mediana
  1 barra (~5 min)**, media 1,65 — territorio de microestructura.
- **Features point-in-time (15):** `|z|`, `resid`, reversión de 1/2/3/5 barras, distancia a
  VWAP de sesión, RVOL, rango de barra, ATR %, minutos desde apertura, retorno y tendencia de
  SPY, β, día de la semana. Todas conocidas en `t`; el trade va de `t` (o `t+lag`) hacia
  delante.
- **Meta-modelo:** HistGradientBoosting y Logística, con **PURGED WALK-FORWARD + EMBARGO** (6
  folds expanding, purga de 5 días entre train y test).
- **Umbral de convicción:** se elige **IN-SAMPLE** (en el train de cada fold, el umbral que
  maximiza la expectancy neta) y se aplica **OOS** (test del fold). Nunca se mira el OOS para
  elegirlo.
- **Coste realista (ida y vuelta = 2 lados):** **TAKER** (retail, a mercado) 3 bps/lado →
  −6 bps/round-trip; **MAKER** (fondo, límite pasivo + rebate) −0,25 bps/lado → +0,5 bps de
  crédito.
- **Anti-overfit:** **Deflated Sharpe Ratio** (Bailey-LdP) + **PBO/CSCV** (Bailey et al.) +
  **IC95 bootstrap por bloques-día**.
- **Test de retardo (`entry_lag`):** se repite TODO entrando 1 barra (5 min) después. **Este
  test es el que decide.**

### Cómo se evita el leakage (explícito)

1. **β** de cada sesión usa solo cierres diarios **anteriores** (rolling 60 sesiones, `shift +1`).
2. **z-score causal:** media/std del resid sobre `[t−W, t−1]` (no incluye `t`).
3. **Barreras hacia delante:** la etiqueta solo usa barras `j > t`; la barra que decide no es la
   que paga.
4. **Sin solapamiento:** tras un evento se salta hasta su barra de salida (etiquetas no
   solapadas por ticker).
5. **Purga + embargo** de 5 días entre train y test en el walk-forward.
6. **Umbral elegido en train, aplicado en test.**
7. **`entry_lag`** desenmascara el rebote bid-ask (microestructura vs alfa).

---

## Resultados — LAG 0 (entrada en la barra de la señal)

**Baseline (tomar TODOS los eventos):** BRUTO **+0,36 bps**, TAKER **−5,64 bps**, MAKER
**+0,86 bps** (win 49,9 %). La señal cruda es ridícula; a mercado pierde el spread entero.

Umbral elegido in-sample y aplicado OOS (236.892 eventos OOS):

| Modelo | Coste | exp OOS | n (sel) | % eventos | DSR | PBO | IC95 (bps) | ¿robusto? |
|---|---|---|---|---|---|---|---|---|
| Logística | TAKER | +22,4 bps | 136 | 0,06 % | 0,58 | 0,03 | **[−4,9, +44,6]** | ❌ IC cruza 0 |
| Logística | MAKER | +28,9 bps | 136 | 0,06 % | 0,75 | 0,03 | [+1,6, +51,1] | ❌ DSR<0,95 |
| **HGB** | **TAKER** | **+14,3 bps** | 283 | 0,12 % | **0,84** | 0,00 | [+5,5, +22,8] | ❌ DSR<0,95 |
| **HGB** | **MAKER** | **+20,8 bps** | 283 | 0,12 % | **0,98** | 0,00 | [+12,0, +29,3] | ⚠️ "pasa" el gate |

Barrido fijo de umbrales (HGB, OOS) — la expectancy sube con la convicción pero la muestra se
desploma:

| umbral | n | TAKER exp | MAKER exp |
|---|---|---|---|
| 0,45 | 224.104 | −5,37 bps | +1,13 bps |
| 0,50 | 143.817 | −4,82 bps | +1,68 bps |
| 0,55 | 9.068 | −1,72 bps | +4,78 bps |
| 0,60 | 971 | +6,95 bps | +13,45 bps |
| 0,65 | 283 | +14,33 bps | +20,83 bps |

El meta-modelo **sí** separa: al exigir convicción, pasa de −5 bps (todos) a positivo. Pero el
positivo vive en **0,12 % de los eventos (283 de 236.892, ~60 trades/año)**, y solo el escenario
**MAKER** supera el triple filtro DSR≥0,95 / PBO≤0,30 / IC95>0. El **TAKER** (lo que paga una
cuenta retail) se queda en DSR 0,84 con muestra diminuta: **no robusto**.

---

## El test que lo decide — rebote bid-ask (`entry_lag` = 1 barra = 5 min)

Se reconstruye TODO entrando **5 minutos después** de la señal. Si el edge era predicción, debe
sobrevivir; si era el rebote entre bid y ask, desaparece.

| | eventos | **gross medio/evento** | HGB TAKER (mejor OOS) | HGB MAKER (mejor OOS) |
|---|---|---|---|---|
| **lag 0** (entra en la señal) | 276.373 | **+0,36 bps** | +14,3 bps · DSR 0,84 · IC[+5,5,+22,8] | +20,8 bps · DSR 0,98 · IC[+12,0,+29,3] |
| **lag 1** (entra 5 min después) | 227.720 | **−0,02 bps** | +4,2 bps · DSR **0,37** · IC**[−2,2,+10,2]** | +9,9 bps · DSR 1,00 · IC[+6,1,+13,7] |

**El componente predictivo bruto por evento (+0,36 bps) se evapora a −0,02 bps con un solo
retardo de 5 minutos.** Es decir: **todo** el edge por evento era rebote bid-ask, no predicción.
Consecuencias en el meta-modelo:

- **HGB TAKER** cae a +4,2 bps con **DSR 0,37** e **IC95 que cruza cero** → muerto. Los buckets
  de alta convicción casi se vacían (n = 83, 5) porque la "convicción" era convicción sobre el
  rebote.
- **HGB MAKER** aún sale +9,9 bps, pero (a) se ha **partido por la mitad** (de +20,8), y (b)
  sigue siendo MAKER = **cobrar el spread**. Lo que queda tras quitar el rebote es literalmente
  el crédito de +0,5 bps/trade del rebate, cosechado en los nombres donde proveer liquidez es
  más seguro. Es el negocio del market-maker, no alfa predictiva.
- **Logística MAKER** a lag 1: +0,94 bps pero **PBO 0,70** (sobreajuste alto) y TAKER −5,56 bps.

---

## Importancia de features (permutation AUC, OOS)

| feature | lag 0 | feature | lag 0 |
|---|---|---|---|
| `resid` | +0,033 | `bar_range` | +0,001 |
| `min_apertura` | +0,025 | `dist_vwap` | +0,001 |
| `ret1` | +0,006 | `rvol` | +0,001 |
| `absz` | +0,002 | resto (SPY, β, ATR, dow) | ≈0 |

La "mezcla" de muchas señales **no existe de verdad**: el meta-modelo se apoya casi solo en la
**magnitud del residuo** (tamaño del desajuste idiosincrático) y la **hora del día**. VWAP,
RVOL, rango, estado de SPY, β, día de la semana no aportan AUC. No es un ensemble de edges
independientes; es "coge los desajustes grandes a ciertas horas" —y ese desajuste, a 5 minutos,
es el spread.

---

## Veredicto honesto

> **El meta-labeling NO convierte las señales débiles en un edge accesible para un taker.** Bajo
> coste retail (TAKER) la expectancy OOS nunca es robusta: a lag 0 es positiva pero DSR < 0,95
> sobre 0,12 % de los eventos (≈60 trades/año), y a lag 1 se cae a DSR 0,37 con IC95 que cruza
> cero. El único escenario que "pasa" el gate automático es **HGB + MAKER a lag 0**, y el test
> de retardo lo desmonta: el edge bruto por evento desaparece con 5 min de retraso → era
> **rebote bid-ask**. "Sobrevive" en MAKER porque MAKER *es* cobrar ese rebote.

Es el patrón de libro del repo, reproducido ahora con el aparato completo de LdP (triple-barrera
+ purged walk-forward + DSR + PBO): **hay señal bruta minúscula, pero es microestructura; el
dinero está en el lado maker/rebate, que no es el nuestro.** La selectividad del meta-modelo
baja la varianza y concentra el rebote, pero **no crea una media que no existe** una vez que se
cobra (taker) el spread en lugar de cobrarlo (maker).

**Implicación operativa (coherente con el resto del repo): NO cablear.** La mezcla de señales
intradía débiles con meta-labeling, por riguroso que sea, no cruza el coste de ejecución retail.

### Caveats (honestidad > resultado)

1. **El gate automático marcó "EDGE ROBUSTO NETO" el HGB+MAKER lag 0.** Lo reportamos tal cual y
   lo desmontamos: es MAKER (spread), muestra de 0,12 %, y cae a la mitad con `entry_lag`. Un
   veredicto automático no sustituye la lectura crítica.
2. **DSR optimista:** `n_trials` usado = 10, pero el nº real de configuraciones probadas
   (2 modelos × 2 costes × 5 umbrales × hiperparámetros × L/W/barreras) es mucho mayor. Con más
   trials el DSR baja: el "0,98" del MAKER lag 0 es un techo, no un suelo.
3. **MAKER es optimista** (relleno pasivo al precio de la señal al 100 %, sin adverse
   selection). El MAKER realista es peor → refuerza el veredicto.
4. **Umbral in-sample vía fit sobre el propio train** (no CV anidada completa): optimista en la
   selección de umbral, pero DSR/PBO/IC95 no dependen de ella y son los que mandan.
5. **Datos 5-min OHLC** (no tick/quotes). El `entry_lag` es justo el control que aísla el efecto
   microestructura sin necesitar el libro de órdenes.
6. **Survivorship entre megacaps es pequeño** (rara vez deslistan), pero el universo se forma
   PIT igualmente (sin selección look-ahead); el sesgo que importaba aquí era el de ejecución,
   no el de supervivencia.

---

*Infra: `backtest_ensemble_metamodel.py` (builder + meta-modelo + validación), `universo_pit.py`,
`fuentes/massive.py`. Reutiliza los validadores DSR/PBO del `orb_metalabel.py` y el patrón de
coste taker/maker + `entry_lag` del `backtest_xsec_intradia.py`. Solo research/paper.*
