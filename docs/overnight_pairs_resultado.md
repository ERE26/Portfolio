# Pares OVERNIGHT: ¿revierte el spread de un par de la noche a la apertura?

Fecha: 2026-10-05 · Alcance: **solo investigación / paper, NUNCA dinero real.**
Datos: **Massive EOD + minuto, survivorship-free** (incluye deslistados con su
histórico hasta la baja), 2021-10-06 → 2026-10-02 (~1 253 sesiones).
Código de la prueba: `scratchpad/ovp_*.py` (no se commitea; usa `datos/cache`
y la API de Massive por cabecera `Authorization`, nunca claves en disco versionado).

## La hipótesis (mezcla de los dos únicos hallazgos reales del estudio)

1. **La prima de la renta variable se gana DE NOCHE** (close→open); el intradía
   aporta poco o nada.
2. **El spread de pares emparejados REVIERTE** intradía —señal bruta real
   (≈+1,5 bp, ~62 % de acierto)— pero **muere al coste** de pagar el spread
   bid-ask repetidas veces a lo largo del día (*churn*).

**Hipótesis nueva:** un par con z extremo **al cierre** podría revertir **a la
apertura siguiente**, capturado con **UN SOLO round-trip nocturno** (MOC al
cierre: largo la pata barata / corto la cara; MOO en la apertura), evitando el
*churn* intradía y sobre un movimiento potencialmente mayor de noche.

## Metodología (honestidad > resultado bonito)

- **Causal.** Para cada día D: beta OLS (hedge ratio, log-precios) y z del spread
  con una ventana que **termina en el cierre de D** (L=60 sesiones; L=40 como
  robustez). Nada del futuro entra en la señal.
- **Operación.** Si |z|≥umbral (1,5 / 2,0 / 2,5) se abre al **cierre de D** (MOC)
  y se cierra en la **apertura de D+1** (MOO). Dollar-neutral, bruto=1
  (w_a=1/(1+β), w_b=β/(1+β)). PnL de reversión = −sign(z)·(w_a·ret_a − w_b·ret_b),
  con ret = open(D+1)/close(D) − 1.
- **Coste MOC/MOO realista.** c bp **por pata y por lado**; un round-trip toca
  las dos patas en entrada (MOC) + dos en salida (MOO), y como el notional bruto
  por lado suma 1, el **coste total por operación = 2·c bp**. Se barre c ∈ {1, 2, 3}.
- **Selección de pares sin sesgo.** Lista propia de **large-caps del mismo
  sector/subsector** (lógica económica, no scan ciego) + **cribado de
  cointegración Engle-Granger (ADF p<0,10, β>0, media-vida 1–60 d) calculado SOLO
  en la ventana de entrenamiento**. Los pares que se operan OOS se fijan en train.
- **Survivorship-free de verdad.** El universo incluye pares con una pata
  deslistada (FRC baja 2023-04-28; ATVI 2023-10-12; WBA 2025-08; EA 2026-08), con
  su histórico completo hasta la baja. Sin sesgo de supervivencia.
- **Robustez anti-autoengaño:** walk-forward 60/40 con purga/embargo (horizonte
  1 noche), bootstrap IC95 (10 000 remuestreos), **DSR** (Bailey & López de Prado,
  6 trials = 2 lookbacks × 3 umbrales), consistencia train≈test, y **test
  entry_lag** (salir en open+15 min en vez de en la subasta de apertura).

## Paso 0 — ¿se ve la prima nocturna en estos datos? (sí)

Retorno acumulado 2021-10 → 2026-10, survivorship-free:

| ETF | Noche (close→open) | Intradía (open→close) | Buy&Hold | bp/día noche | bp/día intradía |
|-----|-------------------:|----------------------:|---------:|-------------:|----------------:|
| SPY | **+36,3 %**        | +29,8 %               | +77,0 %  | +2,69        | +2,49           |
| QQQ | **+61,2 %**        | +29,3 %               | +108,4 % | +4,18        | +2,77           |
| IWM | **+57,6 %**        | **−18,7 %**           | +28,0 %  | +4,00        | −0,96           |

Hallazgo #1 **replicado**: la noche carga la prima; en small-caps (IWM) el
intradía es directamente **negativo**. Pero —clave para lo que sigue— esta prima
es **direccional** (estar largo overnight), no algo que capture una pata-contra-pata
market-neutral.

## Paso 1 — cribado de cointegración (solo TRAIN < 2024-04)

**12 de 50 pares** pasan (ADF p<0,10, β>0, media-vida operable 1–60 d):

```
KEY/CFG (p=.037, 22,5d)  RF/HBAN (.003, 16,6d)  V/MA (.000, 12,7d)
HD/LOW (.086, 33,4d)     MCD/YUM (.066, 31,3d)  COP/EOG (.082, 23,1d)
AMAT/LRCX (.054, 27,5d)  HON/MMM (.053, 29,3d)  EMR/ETN (.048, 29,1d)
F/GM (.031, 31,8d)       UNH/HUM (.005, 20,7d)  ALL/TRV (.055, 30,3d)
```

## Paso 2 — resultado del par overnight (muestra completa, L=60)

bp = puntos básicos por operación. "neto cX" = neto a c = X bp/pata (round-trip 2X).

| Umbral | n | Bruto | Acierto | Neto c1 | Neto c2 | Neto c3 | Sharpe/op | Sharpe día (anual) | t | IC95 neto c2 | DSR | Break-even (bp/pata) |
|--------|--:|------:|--------:|--------:|--------:|--------:|----------:|-------------------:|----:|:------------:|----:|---------------------:|
| \|z\|≥1,5 | 3 910 | +1,21 | 50,2 % | −0,79 | −2,79 | −4,79 | −0,051 | −1,06 | −3,20 | [−4,50; −1,04] | 0,000 | **0,61** |
| \|z\|≥2,0 | 1 841 | −0,05 | 49,7 % | −2,05 | −4,05 | −6,05 | −0,072 | −1,63 | −3,07 | [−6,60; −1,43] | 0,000 | −0,02 |
| \|z\|≥2,5 |   739 | +1,32 | 51,8 % | −0,68 | −2,68 | −4,68 | −0,047 | −1,04 | −1,27 | [−6,89; +1,35] | 0,005 | **0,66** |

La señal bruta es **finísima** (≤+1,3 bp), con acierto apenas ~50–52 % y
**Sharpe por operación NEGATIVO** incluso en bruto (cola izquierda gorda: los
gaps nocturnos perdedores son mayores que los ganadores). El **break-even del
coste es ≤0,66 bp/pata**, así que incluso 1 bp/pata (round-trip 2 bp) ya lo deja
en negativo.

## Paso 3 — walk-forward 60/40 (pares fijados en train; umbral elegido en train)

Umbral elegido en TRAIN por Sharpe diario = **|z|≥2,5** (el menos malo en train).

| Fold | Umbral | n | Bruto | Neto c2 | t | IC95 neto c2 | P(media>0) | DSR | Break-even |
|------|--------|--:|------:|--------:|----:|:------------:|-----------:|----:|-----------:|
| TRAIN | \|z\|≥2,5 | 329 | +2,76 | −1,24 | −0,42 | [−6,70; +4,99] | 32,6 % | 0,052 | 1,38 |
| **TEST (OOS)** | \|z\|≥2,5 | 410 | **+0,16** | **−3,84** | −1,29 | [−10,07; +1,61] | 9,0 % | 0,002 | **0,08** |
| TEST (OOS) | \|z\|≥1,5 | 2 181 | +1,68 | −2,32 | −1,83 | [−4,77; +0,25] | 3,7 % | 0,002 | 0,84 |
| TEST (OOS) | \|z\|≥2,0 | 1 050 | +0,21 | −3,79 | −2,04 | [−7,45; −0,18] | 1,9 % | 0,000 | 0,10 |

**Train≈test FALLA.** El único ajuste que parecía algo en train (|z|≥2,5, bruto
+2,76 bp, break-even 1,38 bp/pata) **se desploma a +0,16 bp OOS** (break-even
0,08 bp/pata). Firma clásica de sobreajuste al elegir umbral.

Robustez **L=40**: FULL |z|≥2,5 bruto +0,33 bp (BE 0,16); OOS |z|≥2,5 bruto +0,14
bp (BE 0,07). Mismo veredicto en los 6 trials: DSR≈0 en todos.

## Paso 4 — test entry_lag (¿era microestructura?)

OOS, |z|≥2,5, 410 ops, coste c=2 bp/pata (round-trip 4 bp), con **barras de minuto**:

| Salida | Bruto | Acierto | Neto c2 | t (bruto) |
|--------|------:|--------:|--------:|----------:|
| Apertura 09:30 (minuto) | +0,16 bp | 52,7 % | −3,84 | +0,06 |
| **Open + 15 min (09:45)** | **−0,57 bp** | 48,5 % | −4,57 | −0,17 |

Dos lecturas: (a) la salida 09:30 por minuto **coincide** con la salida por la
apertura diaria (+0,16 bp) → los datos EOD y de minuto son consistentes; (b) el
mísero sliver bruto de la apertura **se vuelve negativo** a los 15 minutos, y el
t-stat es ≈0 en ambos casos. Es decir: lo poco que se ve en el print de apertura
es **ruido de microestructura** (bid-ask bounce / subasta), **no alfa**.

## ¿Cruza el coste OOS? NO.

El edge bruto OOS es ≤0,16 bp/op (break-even 0,08 bp/pata). Un coste MOC/MOO
realista de **1–3 bp/pata** (round-trip 2–6 bp) está **1–2 órdenes de magnitud
por encima** del edge bruto. No hay ningún umbral, lookback ni ventana donde el
neto OOS sea positivo.

## VEREDICTO: HIPÓTESIS RECHAZADA (no sobrevive neto OOS)

La idea era elegante pero **no se sostiene**. Los dos hallazgos **no se componen**:

- La **prima nocturna es direccional** (todos quieren el riesgo overnight). Un par
  **market-neutral** no tiene exposición neta nocturna, así que **no cobra nada**
  de esa prima; solo se come el riesgo de gap idiosincrático.
- La **reversión de pares es un fenómeno intradía de microestructura** (muchos
  incrementos pequeños a lo largo del día). **No se concentra** en un único
  round-trip nocturno: la ventana overnight la domina el **gap idiosincrático**
  (resultados, noticias, saltos de un solo nombre) que **no** revierte a nivel de
  par. Por eso el acierto cae del ~62 % intradía al ~50–52 % overnight y el
  Sharpe/op es negativo.

Combinarlos da lo peor de cada mundo: ni prima (neutral) ni reversión (pocos
round-trips, dominado por gaps). El coste solo remata a un bruto que ya es ruido.

Lección metodológica confirmada: el edge intradía de pares era frágil **al
coste**; trasladarlo a overnight no lo rescata —lo convierte directamente en
**ruido con cola izquierda**. Honestidad > resultado bonito: **se archiva**.
