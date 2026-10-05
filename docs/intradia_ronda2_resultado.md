# Intradía — RONDA 2: familias GENUINAMENTE no testeadas (Massive minuto, PIT)

**Fecha:** 2026-10-05 · **Datos:** Massive (Polygon-compatible), barras de 5 min de sesión
regular (09:30–16:00 ET), ~2023-11 → 2026-09. **Solo research/paper.**
**Código:** `backtest_pairs_massive.py`, `backtest_compresion_massive.py`.

## Por qué esta ronda y qué NO se repite

El mapa previo del repo ya dejó muertas, con rigor OOS y (donde aplica) sin sesgo de
supervivencia, estas familias intradía:

- ORB largo de acción (confirmado): el edge se lo comía la **supervivencia** (`orb_massive_resultado.md`).
- ORB de índice SPY/QQQ/IWM: **PIERDE** (`intradia_massive_resultado.md`).
- Momentum genérico, pullback-VWAP, mean-reversion a VWAP in-play, última hora
  (momentum y reversión), gap-and-go, gap-fade en ETF: **muertas** (`intradia_busqueda_edge.md`,
  `intradia_massive_resultado.md`).
- **Lado corto (breakdown, simétrico al ORB): ya probado y MUERTO** — sin edge, el trailing
  PIERDE (`intradia_nuevos_massive_resultado.md`, Estrategia 1). **No se repite aquí.**
- Único superviviente de todo el estudio: **overnight de QQQ filtrado por tendencia** — y ese
  **NO es intradía** (es close→open, mantiene de noche).

Eduardo quiere **intradía DE VERDAD** (operar durante la sesión, plano overnight). Esta ronda
ataca **dos familias que el repo nunca había medido**:

1. **Pairs / stat-arb intradía** (market-neutral) — familia totalmente nueva.
2. **Volatilidad: compresión → expansión** (breakout sólo tras compresión de rango).

Rigor idéntico al resto: point-in-time sin look-ahead, universo sin supervivencia por
construcción, muestras pequeñas, 3 semillas, bootstrap IC95, costes modelados. Honestidad >
resultado.

---

## Familia 1 — PAIRS / STAT-ARB INTRADÍA · `backtest_pairs_massive.py`

**Idea.** Operar el **spread** entre dos ETFs económicamente emparejados (mismo sector/tema,
holdings solapados → cointegración REAL, no minada). Market-neutral: no se apuesta a la
dirección del mercado, sino a que una divergencia intradía del spread revierte.

**Universo FIJO de 10 pares** (pre-especificados por lógica económica; ningún ETF se deslista
⇒ sin sesgo de supervivencia, y sin elegir el par mirando el resultado):
XLE·XOP, XLE·VDE, XLF·VFH, XLF·KRE, XLK·VGT, SMH·SOXX, GDX·GDXJ, XHB·ITB, XBI·IBB, QQQ·XLK.

**Point-in-time, sin look-ahead:**
- El **hedge ratio β** de cada día se estima por OLS `logA~logB` sobre las **20 sesiones
  anteriores** (nunca el día en curso).
- La **escala σ** del z-score es la dispersión INTRADÍA del spread (residuo del spread tras
  quitarle la media de cada sesión previa) — la escala correcta para una divergencia dentro
  del día, estimada también sólo con el pasado.
- El spread del día se ancla a su **nivel de la mañana** (media de las primeras 12 barras):
  `z_t = (spread_t − μ_mañana) / σ`. Así z mide divergencia intradía genuina y la reversión a
  z=0 refleja movimiento de precio real, no el decaimiento mecánico de una media móvil que
  absorbe el shock (ese fue un artefacto que se detectó y corrigió).
- **Entrada**: primera barra con |z|≥2 (antes de las 15:30). z>0 → spread caro → corto A /
  largo B. z<0 → al revés. **Salida**: |z|≤0.5 (reversión), |z|≥4 (stop), o cierre forzado
  15:55 (**plano overnight**). A lo sumo **una operación por par-día**.
- **Retorno** por operación sobre capital bruto, neto de coste:
  `r = [signo·(retA − β·retB) − 2·SLIP·(1+β)] / (1+β)`.
- **3 semillas** (42/7/123) muestrean 60 días por par ⇒ ~1.150 operaciones pooled.

### Resultado

**Señal bruta (coste 0) — ¿revierte el spread intradía?**

| seed | ops | acierto | exp/op | IC95 (bps) | PF | veredicto |
|---|---:|---:|---:|---|---:|---|
| 7 | 384 | 64% | +2.0 bp | [+1.1, +3.0] | 1.65 | edge+ |
| 42 | 379 | 64% | +1.4 bp | [+0.3, +2.4] | 1.41 | edge+ |
| 123 | 382 | 59% | +1.2 bp | [+0.3, +2.3] | 1.35 | edge+ |
| **POOL** | **1145** | **62%** | **+1.5 bp** | **[+1.0, +2.1]** | 1.46 | **ROBUSTO+ (IC95>0 en las 3)** |

**Sí hay reversión intradía real** en el spread de ETFs emparejados: +1.5 bp/op bruto, 62% de
acierto, **robusto en las 3 semillas**. Es un hallazgo limpio (el único "edge+ robusto" de la
señal intradía en todo el estudio, junto al overnight) — pero mírese la magnitud.

**Neto de coste — ¿paga la ejecución?** El edge bruto es de ~1,5 bp; una operación de par son
**4 fills** (2 patas × entrada+salida). Barrido de coste (`SLIP` por pata y por lado):

| coste/pata/lado | round-trip aprox. | exp/op neto | IC95 (bps) | veredicto |
|---|---|---:|---|---|
| 0 bp (bruto) | 0 bp | +1.5 bp | [+1.0, +2.1] | edge+ robusto |
| 1 bp | ~4 bp | −0.5 bp | [−1.0, +0.1] | break-even (IC incluye 0) |
| 2 bp | ~8 bp | −2.5 bp | [−3.0, −1.9] | **PIERDE** |
| **4 bp** (estándar del repo) | ~16 bp | **−6.5 bp** | [−7.0, −5.9] | **PIERDE** |

Tabla neta al coste estándar (4 bp/pata/lado), por semilla: −6.0 / −6.6 / −6.8 bp, las tres
con IC95 que **excluye el 0 por abajo** → pierde de forma consistente.

### Veredicto Familia 1: edge REAL pero SUB-COSTE → no operable

El stat-arb intradía de ETFs **sí tiene un edge de reversión real y robusto** (+1,5 bp/op,
3/3 semillas), pero su **break-even está en ~1 bp/pata/lado**. El coste realista de ejecutar 4
fills en ETFs líquidos (spread + impacto + comisión) es difícilmente inferior a 1–2 bp/pata, así
que el **neto es de break-even-a-negativo**. Es el resultado de libro: la anomalía de pares en
instrumentos muy líquidos **ya está arbitrada hasta por debajo del coste**. No es cablear.
Muere por economía de ejecución, no por falta de señal — distinto del ORB (que moría por
supervivencia) y del lado corto (que no tenía ni señal).

---

<!-- SIGUIENTE: Familia 2 (compresión -> expansión) -->
