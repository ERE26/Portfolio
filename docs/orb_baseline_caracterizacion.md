# Caracterización del edge del ORB — BASELINE (sesgado por supervivencia)

Análisis sobre `datos/orb_dataset.csv` (10.074 breakouts, 150 tickers, 2024-09 a 2026-10),
con `analisis_orb_baseline.py`. **Todo con SESGO DE SUPERVIVENCIA** (universo líquido actual
vía Alpaca) → es el **baseline optimista** contra el que compararemos el número limpio del
universo point-in-time de Massive. Métricas con IC95 bootstrap; "ROBUSTO+" = IC95>0.

## 1. Dónde vive el edge

| Corte | n | WR | R medio | IC95 | ¿Robusto? |
|---|---:|---:|---:|---|---|
| Todos (entrada al tocar máx) | 10.074 | 37% | +0,27R | [+0,20,+0,35] | ✅ |
| **Confirmada** (cierra sobre máx) | 6.603 | 43% | **+0,61R** | [+0,50,+0,73] | ✅ |
| Falsa/mecha (ext≤0) | 3.471 | 26% | −0,37R | [−0,42,−0,32] | ❌ pierde |
| Confirmada + RVOL>2 + ≤30 min | 2.459 | 47% | +0,62R | [+0,49,+0,77] | ✅ |

Hallazgos:
- **Confirmación = el filtro clave.** Cerrar sobre el máximo +0,61R; solo mechear −0,37R.
- **Timing importa:** breakout ≤15 min +0,35R (robusto); **>60 min −0,15R (negativo)**.
  → candidato de mejora: estrechar la ventana de entrada de 2h a ~30 min (validar en limpio).
- **OOS temporal 70/30:** la confirmada aguanta fuera de muestra (+0,56 dentro / +0,71 fuera).
  No es artefacto del periodo.
- **Régimen (SPY MA50):** ambos lados positivos (trend +0,17 / no-trend +0,42) → filtrar por
  régimen no añade edge robusto (coherente con el análisis previo).
- **Hurst:** no separa (~+0,27 en todos los tramos).

### ⚠️ Artefacto a NO perseguir
`or_range_atr ≤0,5` marca +3,6R, pero es un artefacto de medición: rango de apertura
minúsculo → stop minúsculo → el R-multiple se dispara (WR 22%, mediana −1,63). No es edge
tradeable.

## 2. ¿Es real o lo sostienen 4 outliers?

**Es ANCHO (no cuatro nombres):**
- 136/150 tickers (91%) rentables; top-10 tickers solo 34% del PnL; sin ellos → +0,43R.
- 62% de días positivos; top-10 días solo 28% del PnL.
- Quitar los 100 mayores ganadores (de 6.603) → sigue +0,25R.

**PERO es un edge de COLA:**
- Media recortada 5% = +0,10R; **recorte 10% = −0,05R (negativo)**; mediana negativa.
- La mayoría de trades pierden poco; un puñado de ganadores grandes paga todo. Es real pero
  exige **dejar correr los ganadores** (mantener EOD / trailing); capar los ganadores lo mata.

## Implicación para la validación con Massive
El edge vive en **la cola** (ganadores grandes) y el sesgo de supervivencia **infla justo la
cola** (los supervivientes son los que tuvieron subidones). Hipótesis: el número limpio de
Massive recortará más la cola que el cuerpo. Por eso el dato con deslistados es decisivo para
saber cuánto edge queda de verdad.

## Estado de la validación de Massive (ver docs/massive_validacion.md)
DATOS BUENOS confirmado en vivo: EOD calca a Yahoo, splits cuadran, **deslistados traen
barras** (p. ej. DBRG, 61 sesiones hasta su baja). Limitación del tier gratis: **~2 años** de
histórico EOD. Suficiente para construir el universo point-in-time y empezar; más histórico
exigiría subir de plan.
