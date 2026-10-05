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

**Matiz clave del pairs/stat-arb:** es el **único intradía con señal bruta real y robusta**
de todo el estudio (reversión genuina del spread de ETFs emparejados). Pero su break-even está
en ~1 bp/pata/lado y ejecutar 4 fills en ETFs líquidos cuesta más → **neto negativo**. Es el
resultado de libro: la anomalía de pares en instrumentos muy líquidos **ya está arbitrada hasta
por debajo del coste**. Hay señal; no hay margen tras costes. (Distinto del ORB, que moría por
supervivencia, y del resto, que no tenía señal.)

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
