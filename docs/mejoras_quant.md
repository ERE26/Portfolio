# Hoja de ruta: depurar el análisis y construir un modelo quant de confianza

Notas de trabajo para la siguiente fase. Reúne (1) qué tickers/sectores funcionan
mejor según los resultados actuales, y (2) los métodos cuantitativos concretos
—con su referencia y su forma de implementarse— para subir el rendimiento y
añadir confianza a las señales. Ordenado por retorno sobre esfuerzo.

## 1. Dónde funciona hoy el sistema (datos propios)

Del walk-forward diario (res final, 745 operaciones seleccionadas, Sharpe 1.21,
PF 1.67), desglose **por sector** (robusto, 35–68 operaciones por grupo):

| Sector | Ret. medio (ATR) | % ganadoras | Profit Factor |
|---|---|---|---|
| ETF de índice | +1.21 | 68% | 3.60 |
| Financiero | +1.21 | 70% | 3.86 |
| Tecnología | +0.93 | 68% | 2.88 |
| Industrial | +0.82 | 56% | 2.34 |
| Materiales | +0.65 | 55% | 1.93 |
| Energía | +0.57 | 56% | 1.83 |
| … | | | |
| Salud | +0.19 | 45% | 1.22 |
| Consumo discrecional | −0.05 | 36% | 0.95 |
| Consumo básico | −0.14 | 34% | 0.86 |
| Utilities | −0.44 | 29% | 0.60 |

**Lectura económica:** el edge de patrones/momento vive en sectores **cíclicos y
tendenciales** (financiero, tecnología, industrial, índices) y **desaparece en
defensivos** (utilities, consumo básico), que son más de reversión y menos
direccionales. Es una señal defendible, no un artefacto.

Por ticker el resultado es demasiado ruidoso (8–11 operaciones por valor) para
fiarse: los "mejores" (XLK, EEM, MS, DIS, BAC, DE, GE, MSFT) son indicativos, no
un ranking fiable. La conclusión operable es **de sector**, no de ticker.

> Nota metodológica importante: antes de creerse cualquiera de estas mejoras hay
> que pasar el filtro de sobreajuste (sección 2.1). Con búsqueda de parámetros,
> selección de features y 100+ tickers, el Sharpe reportado está casi seguro
> inflado por *multiple testing*.

## 2. Métodos para depurar y añadir confianza (priorizados)

### 2.1 Métricas anti-sobreajuste — PRIMERO (lo que más mueve la aguja)
- **Deflated Sharpe Ratio (DSR)** y **Probability of Backtest Overfitting (PBO)**:
  corrigen el Sharpe por nº de experimentos, asimetría y curtosis, y estiman la
  probabilidad de que el "mejor" backtest sea suerte. Bailey & López de Prado
  (2014); Bailey et al. (2017, CSCV).
- Implementación: `pypbo`, o ~100 líneas propias. Contar N (nº de pruebas)
  honestamente, incluyendo cada config probada.
- Caveat: N efectivo es difícil (pruebas correladas cuentan como menos).

### 2.2 Calibración del meta-modelo (encaje perfecto con meta-labeling)
- Que cuando el meta diga 0.70 acierte ~70%: la P calibrada alimenta el sizing.
- **Isotónica** (`CalibratedClassifierCV(method="isotonic")`) con ≥1-2k muestras, o
  **Venn-Abers** (`venn-abers`, da un intervalo [p0,p1] cuyo ancho es incertidumbre).
- Clave: calibrar en un *fold separado purgado* dentro de cada ventana.
- Hoy la correlación confianza↔acierto es solo +0.035: hay margen real aquí.

### 2.3 Pesos por unicidad + embargo (bug silencioso probable)
- Las etiquetas triple-barrier se solapan en el tiempo → no son independientes;
  el modelo sobre-pondera periodos densos. López de Prado, AFML cap. 4.
- *Average uniqueness*, *time-decay*, *sequential bootstrap*; verificar que el
  **embargo ≥ horizonte** de la barrera vertical. (Ya hay `pesos_unicidad` en el
  repo: revisar que se use en TODAS las capas, incluido el meta y la calibración.)

### 2.4 Position sizing y riesgo
- **Volatility targeting** (w = σ_objetivo/σ̂): de los ajustes más robustos; mejora
  el Sharpe realizado. Harvey et al. (2018).
- **Fractional Kelly** (0.2–0.3×) sobre la P *calibrada*: f* = 2p−1 (odds simétricas)
  o f*=p−(1−p)/b; full Kelly es ruinoso con P mal estimada.
- **Ledoit-Wolf** (`sklearn.covariance.LedoitWolf`) o **HRP** (`PyPortfolioOpt`) para
  la covarianza de 100+ tickers: evita concentrar riesgo. Suaviza el equity, no
  sube el acierto.

### 2.5 Conformal prediction como "puerta" de abstención
- Regla de cuándo-operar con cobertura (bajo intercambiabilidad): abstenerse si el
  conjunto es ambiguo. `MAPIE` o `crepes` (Mondrian = calibrar por régimen/sector).
- Para k-NN, el *nonconformity score* natural es el ratio de distancias intra/inter
  clase → casi gratis sobre la estructura actual.
- Caveat: las series no son intercambiables; usarlo como heurística, no garantía.

### 2.6 Mejoras de señal (quick wins)
- **SHAP** (`shap.TreeExplainer`) sobre el meta para podar features y, sobre todo,
  **pesar la métrica del k-NN** por importancia (mueve más que añadir modelos).
- **Ensemble de horizontes**: combinar varios triple-barrier (distintos TP/SL y
  barrera temporal). Barato y sube robustez.
- **Normalización rolling por ventana/régimen** de las features del k-NN.

### 2.7 Régimen (con escepticismo)
- **HMM de 2 estados** (`hmmlearn`) sobre vol realizada: útil como capa *defensiva*
  (reducir exposición en alta-vol), que solapa con vol targeting. Inestable OOS;
  medir siempre el aporte incremental con PBO.

### Lo que ayuda poco (no invertir semanas)
- BMA frente a stacking regularizado; HMM de muchos estados; full Kelly;
  modelos factoriales de covarianza muy elaborados (Ledoit-Wolf/HRP dan el 90%);
  tunear sin fin los hiperparámetros del GBM (cada barrido infla N en el DSR).

## 3. Orden sugerido de implementación
1. Instrumentar **DSR + PBO** y recalibrar expectativas.
2. **Pesos por unicidad** + verificar **embargo ≥ horizonte** en todas las capas.
3. **Calibrar el meta** (isotónica/Venn-Abers) en fold purgado → umbral por utilidad.
4. **Vol targeting + Ledoit-Wolf/HRP** y **fractional Kelly 0.2–0.3×** sobre P calibrada.
5. **SHAP** para podar/pesar la métrica k-NN; **ensemble de horizontes**.
6. (Opcional) **conformal por régimen** y **HMM 2 estados** como capa defensiva.

## Referencias principales
- López de Prado (2018), *Advances in Financial Machine Learning* (caps. 3-4, 7, 8, 11-14).
- Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*; Bailey et al. (2017), *PBO/CSCV*.
- Angelopoulos & Bates (2021), *A Gentle Introduction to Conformal Prediction* (arXiv:2107.07511);
  Vovk & Petej (2014), *Venn-Abers*.
- Ledoit & Wolf (2004, 2020), *shrinkage*; López de Prado (2016), *HRP*.
- Kelly (1956); Thorp (2006); Harvey et al. (2018), *vol targeting*.
- Platt (1999); Zadrozny & Elkan (2002) (calibración). Lundberg & Lee (2017) (SHAP).
- Librerías: `mlfinlab`/`mlfinpy`, `pypbo`, `MAPIE`/`crepes`, `venn-abers`,
  `PyPortfolioOpt`, `sklearn.covariance`, `hmmlearn`, `arch` (MCS), `shap`.
