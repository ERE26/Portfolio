# ¿El edge del runner está en la SELECCIÓN? (meta-labeling de low-float runners)

Fecha: 2026-10-05 · Alcance: **SOLO research / paper**, cero gasto en órdenes.
Datos: Massive (plan Starter) — grouped diario (incluye deslistados) + barras de
1 min. Código: `runners_seleccion.py`, que **reutiliza** `universo_runners_pit.py`
(universo PIT sin supervivencia) y `backtest_lowfloat_massive.py` (gate in-play +
la R real de cada setup), y aplica la maquinaria anti-overfit de `orb_metalabel.py`
(purged walk-forward, DSR, PBO/CSCV, IC95 bootstrap).

## La pregunta

`docs/lowfloat_runners_resultado.md` cerró en negativo la hipótesis del runner
*en media*: gap-and-go, ORB runner y short parabólico **no tienen edge robusto**
sobre el universo point-in-time sin supervivencia. Pero su **sección 7** dejó una
puerta abierta: *"una selección más fina que la de 'top gappers' … que suba el
acierto o la cola de forma estable entre ventanas"*. El day-trader real **no opera
todos los gappers que pasan el gate: selecciona**. Esta es la prueba de esa puerta:

> ¿existe un **subconjunto** de runners —identificable **point-in-time** con
> features conocidas a la apertura— donde el gap-and-go o el ORB runner **sí** sea
> rentable neto de coste, de forma **robusta OOS** (IC95>0, DSR alto, PBO bajo)?

## Montaje (mismo R que el baseline, para que sea comparable)

- **Universo y R idénticos al doc de baseline.** Se parte de los mismos candidatos
  PIT sin supervivencia y del mismo gate in-play (RVOL de apertura ≥ 2, $-volumen
  del rango de apertura ≥ $50k). La etiqueta es la **R real** de `gap_and_go` y
  `orb_runner` de `backtest_lowfloat_massive.py` (entrada a la ruptura del rango
  —≥ 09:35, sin look-ahead—, stop en el mínimo del rango con suelo 2%, salida por
  fade de VWAP / EOD, slippage 25 bps/lado base). **No invento un R nuevo.**
- **Dataset:** **2.802 runner-days** que pasan gate, 378 sesiones, 2023-07-05 →
  2024-12-31. El gap-and-go dispara en 980; el ORB runner en 876.
- **Features, todas PIT (conocidas en/antes de la entrada ≥ 09:35):** `gap`,
  `precio` (apertura), `opening_rvol`, `dvol_or` ($-vol del rango de apertura),
  `or_range_pct`, `dow`, `ret_prev1`, `ret_prev3`, `n_green3`, `gap_x_rvol`.
- **Validación:** meta-modelo (logística / HGB) con **purged walk-forward**
  (embargo 3 d), selección por probabilidad; **DSR**, **PBO/CSCV** e **IC95
  bootstrap** por umbral. Más **cortes simples** con PBO sobre el *menú* de cortes.
  Veredicto "EDGE ROBUSTO" solo si **DSR ≥ 0.95 y PBO ≤ 0.30 e IC95(OOS) > 0** a la
  vez.

## Baseline "tomar TODOS" (reproduce el "sin edge en media")

| Setup | n | Expectancy @25bps | Win % | IC95 | ¿Edge en media? |
|---|---|---|---|---|---|
| Gap-and-go | 980 | **−0.069R** | 25 % | [−0.126, −0.007] | **NO** |
| ORB runner | 876 | **−0.144R** | 29 % | [−0.258, −0.018] | **NO** |

Coherente con el doc de baseline (gap-and-go −0.07R; ORB −0.19/+0.05R entre
ventanas). Punto de partida confirmado: **en media, rojo**.

## ¿Rescata la selección el edge, neto de coste y OOS?

### ORB runner (`R_orb`)

Meta-modelo (OOS, purged walk-forward):

| Modelo | mejor umbral | % tomado | Expectancy | n | DSR | PBO | IC95 |
|---|---|---|---|---|---|---|---|
| Logística | 0.60 | 13 % | **+0.252R** | 98 | 0.16 | 0.17 | [−0.202, +0.896] |
| HGB | 0.40 | 20 % | +0.079R | 153 | 0.14 | 0.95 | [−0.167, +0.362] |

Cortes simples: el mejor, `gap 30-50% & precio>$3`, da **+0.042R** (n=85, DSR 0.07,
IC95 [−0.326, +0.473]); **PBO del menú 0.66**. Nota: los gaps **grandes son
peores** (`gap 10-30%` −0.044R vs `gap 50-100%` −0.361R vs `gap >100%` −0.405R) —
confirma, en dirección, que 10-30 % > 100 %, pero **ninguno es positivo robusto**.

### Gap-and-go (`R_gapgo`)

| Modelo | mejor umbral | % tomado | Expectancy | n | DSR | PBO | IC95 |
|---|---|---|---|---|---|---|---|
| Logística | 0.60 | 14 % | +0.072R | 120 | 0.14 | 0.57 | [−0.102, +0.304] |
| HGB | 0.45 | 10 % | **+0.179R** | 86 | 0.33 | 0.71 | [−0.066, +0.483] |

Cortes simples: el mejor, `precio>$5 & rvol alto`, da **+0.077R** (n=60, DSR 0.08,
IC95 [−0.262, +0.511]); **PBO del menú 0.51**.

## Lectura honesta

La selección **mueve la expectancy hacia 0 e incluso ligeramente positiva en los
cortes más estrechos** (ORB +0.25R al umbral 0.60; HGB gap-and-go +0.18R). En
dirección, parece que "seleccionar ayuda". Pero **no sobrevive a la vara
anti-overfitting en ningún caso**:

- **DSR** del mejor subconjunto: **0.07–0.33** en todos los casos (hace falta
  ≥ 0.95). Muy lejos; no es cuestión de décimas.
- **IC95 OOS** del mejor subconjunto: **cruza 0 siempre**. No hay un solo
  subconjunto cuyo intervalo despegue de cero.
- **PBO**: 0.17–0.95 (meta) y **0.51–0.66** (menú de cortes). El "ganador"
  in-sample tiende a quedar en la mitad baja OOS: es selección-de-azar.
- Las ganancias aparentes viven en **n pequeño** (54–153 ops): es el **espejismo
  de cola** que el propio baseline advirtió, ahora cuantificado. Al estrechar el
  corte subes la media de la muestra pero **destruyes la significancia**.

**Coste.** Todo esto es a **25 bps/lado** (extremo optimista; el spread real de
estos nombres de $2–3 es 30–150 bps). El baseline ya muere a 50 bps, y la selección
no lo saca de rojo robusto ni al coste favorable, así que a 50–100 bps **empeora**
(el baseline y todos los subconjuntos caen más en rojo; el veredicto no cambia).

## VEREDICTO

**NO. Ningún subconjunto —ni meta-modelo (logística/HGB) ni corte simple— rescata
una expectancy neta > 0 robusta OOS.** Con 2.802 runner-days PIT sin supervivencia
y validación exigente, el mejor subconjunto en cada caso falla por un **margen
amplio** (DSR ≤ 0.33 frente a 0.95, IC95 siempre cruzando 0, PBO alto). La
selección empuja la media hacia cero pero no produce una **regla estable y
significativa**: lo que brilla lo hace en muestras pequeñas y colapsa bajo DSR/PBO.

Esto **cierra también la puerta de la sección 7**: el edge del runner no está en
una selección sencilla ni aprendible *con estas features de apertura*. Si existe,
vive en información que no está aquí (float / short-interest reales, naturaleza del
catalizador, order-flow / L2) — y la carga de la prueba está en encontrarla y
demostrar que sobrevive OOS, no en asumirla.

## Reproducir

```
python runners_seleccion.py --desde 2023-07-01 --hasta 2024-12-31 --topk 12 --construir --label R_orb
python runners_seleccion.py --label R_gapgo        # reanaliza desde el CSV ya construido
```

Solo lectura / paper. No toca `gestionar.py` ni `operativa_*.py`. No cablea nada.
CSV y caché en `datos/` (gitignored): no se suben datos.
