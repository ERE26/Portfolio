# Síntesis maestra: re-validación de TODO el sistema con datos limpios (Massive)

**Fecha:** 2026-10-05 · **Datos:** Massive Starter (minuto + EOD, 5 años, deslistados,
endpoint grouped) · **Solo research/paper.** Re-validación completa tras pagar el plan $29,
sobre universo **point-in-time sin sesgo de supervivencia**. Honestidad > resultado bonito.

## Tabla maestra — SESGADO vs LIMPIO

| Estrategia | Sesgado (antes) | **Limpio (Massive)** | ¿Edge robusto? | Causa de muerte |
|---|---|---|---|---|
| **ORB stock 5-min** (confirmado, EOD) | +0,26R | **−0,069R** | ❌ | supervivencia (haircut +0,33R) |
| ORB stock 1-min | +0,26R | −0,110R | ❌ | supervivencia + ruido |
| **ORB de índice** (SPY/QQQ/IWM) | — | SPY −0,083R (IC95<0) | ❌ **pierde** | el ORB era selección de acciones, no ruptura de mercado |
| Momentum genérico 15m | −0,3R | (ya muerto) | ❌ | sin edge |
| Pullback a VWAP | muerto | (ya muerto) | ❌ | sin edge |
| Mean-reversion a VWAP (in-play) | pierde | (ya muerto) | ❌ | navaja cayendo |
| **Lado corto** (breakdown) | — | sin edge; trailing pierde | ❌ | sin edge |
| Gap-fade en ETF | — | no robusto | ❌ | espejismo de muestra |
| Momentum última hora | — | Sharpe negativo | ❌ | ruido |
| Meta-ML (López de Prado) | sin lift | (base muerta → moot) | ❌ | nada que etiquetar |
| Filtro de régimen (HMM/Hurst/SPY) | sin lift | (base muerta → moot) | ❌ | sin lift |
| **TSMOM** (ETF cross-asset) | Sharpe ~1,1 | 1,13 (sin superv.) | ⚠️ no bate SPY (1,32) | diversificador, no alfa |
| **Rotación Europa** (ETF país) | — | 1,15 vs SPY 1,07 | ⚠️ empate = beta €+divisa | no alfa |
| **Residual momentum** long-only | Sharpe 1,83 / CAGR 64% | **1,45 / 28%** < SPY 1,77 | ❌ | supervivencia (haircut −36 pts CAGR) |
| Overnight **broad-stock** (equal-w, churn diario) | — | bruto +6,6%/año PERO neto<0, < SPY | ❌ | comisiones del cruce diario |
| **Overnight QQQ + tendencia (SMA200)** | — | **Sharpe 1,30, IC95 [0,25,2,43], train≈test** | ✅ **candidato** | — (a validar ejecución en paper) |
| **Pares OVERNIGHT** (\|z\| al cierre → MOO, market-neutral) | — | bruto ≤+1,3 bp/op (~50–52%, Sharpe/op<0); **OOS +0,16 bp**, neto<0 a 1–3 bp/pata | ❌ | ruido + gap idiosincrático (los 2 hallazgos **no se componen**) |

## El patrón honesto
Casi todo nuestro "edge" era **sesgo de supervivencia** (ORB stock, residual momentum) o **no
sobrevive a los costes** (overnight broad-stock) o simplemente **no existía** (momentum,
pullback, mean-rev, última hora, gap-fade, lado corto, ORB índice). Es el resultado típico de
la validación honesta: la mayoría de ideas no sobreviven, y el superviviente es simple.

## El ÚNICO superviviente: overnight de QQQ + filtro de tendencia
- Mantener QQQ solo de noche (MOC→MOO) y solo si cierre>SMA200. Sharpe 1,30 vs 0,70 B&H,
  maxDD −9,6% (vs −35% de QQQ en 2022), IC95 del Sharpe excluye 0 (el único de todo el
  estudio), consistente OOS (train 1,29 / test 1,31), tolerante a coste hasta ~1,5 bp/lado.
- **Matiz honesto (del estudio overnight sobre acciones):** la prima nocturna del MERCADO la
  capturas con solo mantener SPY (overnight de SPY ≈ +6,4%/año con 1 trade). La ventaja del
  QQQ+tendencia no es alfa pura: es **menos exposición y menos drawdown** (fuera en noches
  bajistas) sobre un instrumento **líquido y barato de ejecutar** (por eso sobrevive al coste
  donde la cesta de 3.000 acciones no). Su viabilidad real depende del **slippage de las
  subastas MOC/MOO**, que es justo lo que mide el paper.

## Qué se ha hecho y qué queda
- ✅ **Overnight QQQ → PAPER** (`operativa_overnight.py`, triggers diarios MOC/MOO, tamaño de
  medición, sin tocar TSMOM) para medir el slippage real. Es el muro que decide.
- ✅ **Campaña intradía CERRADA**: pairs/stat-arb, vol-compresión, MR large-cap y el resto, todas
  medidas en limpio (ver tabla abajo). Ninguna da edge robusto neto de costes.
- ❌ Descartado para cablear: todo lo intradía. Único vivo: overnight QQQ (en paper).

## Actualización — más familias intradía (rondas nuevas, todas en limpio)

| Familia intradía | Resultado limpio | ¿Edge robusto? | Muere por |
|---|---|---|---|
| Lado corto (breakdown) | EOD −0,09R (IC95 cruza 0); trailing pierde | ❌ | no hay señal |
| Gap-fade acciones (>2-3%) | acierto alto pero gaps raros → sin muestra | ⚠️ sugerente, infra-potenciado | falta de muestra |
| Mean-reversion VWAP (ETF + large-cap) | **−0,37R, IC95<0, 2 lados × 3 semillas** | ❌ pierde | la extensión continúa, no revierte |
| Time-of-day (patrón horario) | solo continuación débil última hora (+7,5 bps bruto, 2/3) | ❌ | no robusto / neto≈0 |
| **Pairs / stat-arb ETFs** (market-neutral) | **bruto +1,5 bp/op, 62% acierto, ROBUSTO (IC95>0 ×3)** → neto a 4bp/pata −6,5 bp | ❌ **sub-coste** | arbitrado por debajo del coste de ejecución |
| Vol-compresión → expansión | baseline −0,03R (no robusto); NR7/apertura-tight **empeoran** (−0,10/−0,12R, IC95<0) | ❌ pierde | la compresión precede chop/head-fake, no expansión |
| **Stat-arb cross-sectional** (estilo fondo, 100 nombres, dólar/beta-neutral) | bruto Sharpe +20 a 5-min (¡señal de artefacto!); **test de retardo 5min → el edge DESAPARECE** (de +1,14 a −0,31 bps); TAKER −3,9 bps (desastre) | ❌ **no es alfa, es el spread bid-ask** | el "edge" era rebote bid-ask; solo lo cobra quien PROVEE liquidez (market-maker) |
| Pairs en CESTA (acciones, maker + diversificación) | bruto ni significativo (+0,017 bps, IC95 cruza 0); TAKER pierde; MAKER +0,03 bps no signif. | ❌ | sin señal real tras netear |
| **Low-float runners — gap-and-go** (universo del day-trader retail, PIT) | −0,07/−0,12R @25bps (IC95 cruza 0); muere a 50bps | ❌ | ejecución (spread) |
| **Low-float runners — ORB runner** | −0,19/+0,05R (IC95 cruza 0, inconsistente entre ventanas); muere a 50-100bps | ❌ | ruido + ejecución |
| **Low-float runners — short parabólico** | −0,77/−0,32R, **PIERDE siempre** (~80% stops); borrow caro/HTB | ❌ | el runner te arrolla |
| **Low-float runners — SELECCIÓN** (meta-labeling + cortes, 2.802 runner-days PIT) | mejor subconjunto ORB +0,25R / gap-and-go +0,18R PERO **DSR ≤ 0,33, IC95 cruza 0 siempre, PBO 0,5–0,95** | ❌ | espejismo de cola (no sobrevive OOS/DSR/PBO) |
| **Meta-modelo ENSEMBLE** (meta-labeling LdP: reversión + 15 features, 276k eventos, triple-barrera, purged WF) | TAKER nunca robusto (lag0 +14bps DSR0,84 sobre 0,12% eventos; lag1 +4bps DSR0,37 IC cruza 0). "Robusto" solo en MAKER, y **gross/evento +0,36→−0,02bps con entry_lag=1** | ❌ **rebote bid-ask** | mezclar señales débiles concentra el spread, no crea alfa (ver `ensemble_metamodel_resultado.md`) |

**Matiz clave del pairs/stat-arb:** es el **único intradía con señal bruta real y robusta**
de todo el estudio (reversión genuina del spread de ETFs emparejados). Pero su break-even está
en ~1 bp/pata/lado y ejecutar 4 fills en ETFs líquidos cuesta más → **neto negativo**. Es el
resultado de libro: la anomalía de pares en instrumentos muy líquidos **ya está arbitrada hasta
por debajo del coste**. Hay señal; no hay margen tras costes. (Distinto del ORB, que moría por
supervivencia, y del resto, que no tenía señal.)

**Matiz de la SELECCIÓN de runners** (`docs/runners_seleccion_resultado.md`): el baseline del
runner no tenía edge *en media*, pero el day-trader **selecciona**. Se probó si un subconjunto
—meta-modelo logística/HGB con purged walk-forward, o cortes simples— rescata expectancy neta > 0
robusta OOS, sobre 2.802 runner-days PIT sin supervivencia y con el **mismo R** que el baseline.
La selección empuja la media hacia 0 e incluso positiva en los cortes más estrechos, pero
**ningún subconjunto sobrevive** (DSR ≤ 0,33 vs 0,95 exigido, IC95 OOS cruza 0 siempre, PBO alto):
es el espejismo de cola, ahora cuantificado. Cierra la puerta que dejó abierta la sección 7 del
doc de runners: **el edge no está en una selección aprendible con features de apertura.**

**La respuesta definitiva a "cómo ganan los fondos intradía" (test estilo-fondo).** Se montó el
stat-arb como lo hacen ellos (cross-sectional de 100 nombres, dólar/beta-neutral, + escenario de
ejecución MAKER con rebate). El Sharpe bruto salió absurdo (+20 a 5-min) — y el **test de retardo
de 5 min lo desenmascara: el edge entero es REBOTE BID-ASK**, no predicción (al entrar 5 min
después, el edge es ~0/negativo). "Funciona" en MAKER solo porque MAKER = cobrar el spread =
cobrar ese rebote. **Conclusión: el dinero intradía de los fondos en nombres líquidos es
PROVISIÓN DE LIQUIDEZ (market-making: cobrar el spread con prioridad de cola, colocation y
rebates), no una estrategia predictiva.** Es infraestructura de fondo, inaccesible para retail —
y aun para ellos, el componente predecible (no-rebote) a estos horizontes es ≈0. Cierra la
pregunta: no es que nos falte la idea; es que ese dinero se cobra siendo el creador de mercado.

## Recomendación honesta
1. **Dejar correr el paper del overnight QQQ** ~2-4 semanas y mirar el slippage MOC/MOO real.
   Si ≤~1 bp/lado → edge confirmado operable; si >1,5 bp → se cae como la cesta de acciones.
2. **No cablear nada intradía** hoy: no hay edge intradía robusto en limpio.
3. **TSMOM/rotación**: mantener solo como diversificadores de drawdown dentro de la cartera
   multi-libro, NO por retorno esperado sobre SPY.
4. **Lección de cartera:** gran parte del valor de este ejercicio ha sido NEGATIVO —saber qué
   NO funciona— y eso vale tanto como encontrar un ganador: evita arriesgar capital en humo.
## Actualización — overlays sobre el superviviente (overnight QQQ, survivorship-free)

Se intentó **mejorar** el único superviviente con overlays (saltar noches malas o
escalar las buenas), survivorship-free y validados OOS: régimen de VIX (proxy
VIXY/VIXM + vol realizada, **sin FRED en el entorno**), estacionalidad (día de la
semana, turn-of-month, pre/post-FOMC), tendencia/breadth de corto plazo, blackout de
FOMC y vol-targeting. Walk-forward, IC95 bootstrap, **DSR con corrección por nº de
pruebas** (27 por activo / 81 en cesta QQQ-SPY-IWM) y PBO/CSCV.

| Estudio | Hallazgo | Sharpe (coste) | IC95 | DSR (N pruebas) | ¿Sobrevive? | Doc |
|---|---|---|---|---|---|---|
| **Overlays overnight QQQ** (sobre cierre>SMA200) | Ningún overlay bate el baseline tras DSR. Mejor candidato = **evitar turn-of-month** (contradice el efecto TOM clásico → probable artefacto 2022). Tendencia/breadth y blackout FOMC **empeoran**. Hallazgo real: fragilidad a coste y edge casi exclusivo de QQQ (SPY 0,66 / IWM 0,53). | baseline 1,15 @2bps (1,55 @0bps; **0,55 @5bps**); mejor overlay 1,73 @2bps | baseline [0,34, 1,97] | mejor overlay **0,885-0,934** (N=27/81); baseline 0,515 | **No** — nada llega a 0,95 | [overnight_overlays_resultado.md](overnight_overlays_resultado.md) |

**Implicación operativa:** no cablear ningún overlay en `operativa_overnight.py`;
mantener el baseline puro. La prioridad sigue siendo **medir el slippage MOC/MOO real
en paper** — el baseline pasa de Sharpe 1,55 a 0,55 entre 0 y 5 bps.

## Actualización — par OVERNIGHT: ¿se componen los dos hallazgos reales? (NO)

Idea: mezclar los **dos únicos hallazgos reales** del estudio —(1) la prima de la
renta variable se gana **de noche** (close→open); (2) el **spread de pares revierte**
(bruto +1,5 bp/op, 62%, pero muere al *churn* intradía)— en **un único round-trip
nocturno**: |z| extremo al cierre → MOC (largo barata / corto cara), salir en la
apertura siguiente (MOO). Objetivo: evitar el *churn* y capturar un movimiento mayor
de noche. Survivorship-free (Massive EOD+minuto, 50 pares large-cap por sector,
cribado de cointegración Engle-Granger **solo en train**, 12/50 pasan). Causal
(beta/z con datos hasta el cierre de D). Walk-forward 60/40 con purga/embargo,
bootstrap IC95, **DSR** (6 trials) y **test entry_lag** con barras de minuto.

| Estudio | Hallazgo | Bruto / break-even | Neto OOS (c=1–3 bp/pata) | DSR | ¿Sobrevive? | Doc |
|---|---|---|---|---|---|---|
| **Par overnight (\|z\| cierre → MOO, market-neutral)** | Señal bruta = ruido: ≤+1,3 bp/op, acierto ~50–52%, **Sharpe/op<0** (cola izquierda por gaps); OOS +0,16 bp. Train≈test **falla** (el \|z\|≥2,5 de train, +2,76 bp, se desploma a +0,16 OOS). **entry_lag**: el sliver del print de apertura se vuelve **negativo** a los 15 min (t≈0) → microestructura, no alfa. | bruto ≤+1,3 bp; **break-even ≤0,66 bp/pata (0,08 OOS)** | **−2,3 a −3,8 bp/op** (neg. a cualquier coste) | ~0,00 (6 trials) | **No** — rechazado | [overnight_pairs_resultado.md](overnight_pairs_resultado.md) |

**Por qué no se componen:** la **prima nocturna es direccional** (un par
market-neutral no tiene exposición neta de noche → no cobra nada de ella) y la
**reversión de pares es microestructura intradía** (muchos incrementos pequeños; no se
concentra en un único salto nocturno, que lo domina el **gap idiosincrático** —
resultados, noticias— que no revierte a nivel de par). Combinarlos da lo peor de cada
mundo: ni prima ni reversión. El coste solo remata a un bruto que ya es ruido. **Se
archiva.**
