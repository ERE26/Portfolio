# ORB confirmado sin sesgo de supervivencia — el número honesto (Massive Starter)

**Fecha:** 2026-10-05 · **Alcance:** SOLO research/paper, solo lectura de datos.
**Scripts:** `backtest_orb_massive.py`, `universo_pit.py`, `fuentes/massive.py` (`barras_intradia`, `grouped_diario`).

## Pregunta

El edge del ORB-confirmado medido hasta ahora (`docs/orb_baseline_caracterizacion.md`,
`docs/intradia_busqueda_edge.md`) vale **+0,26R/trade** (entrada al CIERRE de la vela que
confirma, 3 semillas +0,30/+0,23/+0,24), pero **con SESGO DE SUPERVIVENCIA**: se muestreaba el
universo líquido de HOY (Alpaca), que excluye los que quebraron, se deslistaron o fueron
absorbidos. La caracterización del baseline ya avisó: *"el edge vive en la COLA (ganadores
grandes) y el sesgo de supervivencia infla justo la cola"*. Aquí se mide el número **limpio**
sobre un universo **point-in-time** (deslistados incluidos) y se compara **1-min vs 5-min**.

## Metodología (idéntica al ORB ya validado; no se inventó otra)

- **Lógica ORB:** se **importa** de `backtest_orb5` (`_opening_rvol`, `_sim`) y el IC95 de
  `backtest_intradia` (`_boot_ci`). Rango de apertura = 1ª vela; `opening_rvol` = vol 1ª vela /
  media 1ª vela de los 14 días previos, gate >1; 1ª vela alcista (long); **ruptura CONFIRMADA**
  (`confirmado=True`: entra al CIERRE de la vela que cierra sobre el máximo del rango); stop =
  mínimo del rango; salida **EOD** y variante **EOD+trailing 2ATR**. `SLIP=0,0004`,
  `SLIP_STOP=0,0008`. Point-in-time, sin look-ahead. Una entrada por ticker-día.
- **Datos:** barras de minuto reales de **Massive Starter** (`barras_intradia`, paginando
  `next_url`, UTC→NY, filtradas a 09:30–16:00). Se baja **1-min** una vez por ticker-día y el
  **5-min se deriva por resample** — verificado que reproduce EXACTO el 5-min nativo de Massive
  (dif OHLC 0,0), así ambos timeframes comparten las mismas velas subyacentes.
- **Universo POINT-IN-TIME (survivorship-free):** para cada día muestreado,
  `universo_pit.universo_asof(día)` vía el **endpoint GROUPED** de Massive
  (`/v2/aggs/grouped/locale/us/market/stocks/{date}`, verificado en Starter: ~12.600
  tickers/día en 1 llamada). Candidatos = los que cotizaban **antes** de ese día (si se
  deslistaron después, ese día siguen siendo candidatos: ese es el punto). Filtros: precio en
  [5, 5000], `dollar_vol_20d > 10M`. **Mega-caps excluidas por PROXY** (percentil 95 de
  dollar-vol as-of — aproximación documentada, no market cap real), para operar la banda
  mid/large líquida comparable al baseline. Top-150 por dollar-vol; el backtest samplea ~70/día.
  - *Nota:* no hizo falta el fallback de `tickers_referencia` + `barras_diarias`; GROUPED está
    incluido en Starter.
- **Validación del survivorship-free:** del universo as-of de hace ~12–24 meses, **3–4%** son
  tickers **hoy deslistados/renombrados/absorbidos** (p. ej. PXD→Exxon, SWAV→J&J, HES→Chevron,
  DFS→Capital One, SQ→renombrado). Es >0 → los deslistados SÍ entran. (Con top-1% en vez de
  top-5% el universo es casi todo large-cap superviviente y el % cae a ~0,7%: insuficiente;
  bajando en capitalización el % sube a 5–9%. La supervivencia crece al bajar de capitalización.)
- **Muestra:** ~70 tickers × 30 días al azar × 24 meses × **3 semillas independientes**
  (42, 7, 123); ~700–750 operaciones por configuración (pooled). Veredicto **ROBUSTO+** solo si
  IC95>0 en las 3 semillas.

## Resultado

Baseline SESGADO (universo de hoy): ORB confirmado entrada-al-cierre ≈ **+0,26R/trade**.

### 5-min | EOD (mantener hasta el cierre)

| semilla | ops | acierto | expectancy | IC95 | PF | veredicto |
|---|---:|---:|---:|---|---:|---|
| 42  | 251 | 45% | −0,082R | [−0,184, +0,024] | 0,79 | sin edge |
| 7   | 196 | 40% | +0,006R | [−0,177, +0,197] | 1,01 | sin edge |
| 123 | 269 | 43% | −0,111R | [−0,225, −0,002] | 0,75 | pierde |
| **POOL** | **716** | **43%** | **−0,069R** | **[−0,143, +0,010]** | **0,84** | **sin edge / pierde** |

### 5-min | EOD + trailing 2ATR

| semilla | ops | acierto | expectancy | IC95 | PF | veredicto |
|---|---:|---:|---:|---|---:|---|
| 42  | 251 | 48% | −0,065R | [−0,160, +0,036] | 0,82 | sin edge |
| 7   | 196 | 42% | −0,061R | [−0,206, +0,094] | 0,86 | sin edge |
| 123 | 269 | 45% | −0,060R | [−0,177, +0,052] | 0,86 | sin edge |
| **POOL** | **716** | **46%** | **−0,062R** | **[−0,130, +0,005]** | **0,85** | **sin edge / pierde** |

### 1-min | EOD

| semilla | ops | acierto | expectancy | IC95 | PF | veredicto |
|---|---:|---:|---:|---|---:|---|
| 42  | 237 | 38% | −0,006R | [−0,187, +0,191] | 0,99 | sin edge |
| 7   | 244 | 30% | −0,228R | [−0,422, −0,016] | 0,68 | pierde |
| 123 | 269 | 32% | −0,095R | [−0,283, +0,096] | 0,85 | sin edge |
| **POOL** | **750** | **33%** | **−0,110R** | **[−0,216, +0,008]** | **0,83** | **sin edge / pierde** |

### 1-min | EOD + trailing 2ATR

| semilla | ops | acierto | expectancy | IC95 | PF | veredicto |
|---|---:|---:|---:|---|---:|---|
| 42  | 237 | 46% | −0,070R | [−0,213, +0,083] | 0,86 | sin edge |
| 7   | 244 | 40% | −0,270R | [−0,396, −0,130] | 0,56 | pierde |
| 123 | 269 | 45% | −0,078R | [−0,212, +0,057] | 0,85 | sin edge |
| **POOL** | **750** | **44%** | **−0,138R** | **[−0,216, −0,054]** | **0,75** | **PIERDE (IC95<0 pooled)** |

## El número honesto vs el baseline sesgado — haircut de supervivencia

| timeframe × modo | sesgado | **sin supervivencia** | haircut | ¿robusto (IC95>0 ×3)? |
|---|---:|---:|---:|---|
| 5-min · EOD            | +0,26R | **−0,069R** | **+0,329R** | ❌ no |
| 5-min · EOD+trail2ATR  | +0,26R | **−0,062R** | +0,322R | ❌ no |
| 1-min · EOD            | +0,26R | **−0,110R** | +0,370R | ❌ no |
| 1-min · EOD+trail2ATR  | +0,26R | **−0,138R** | +0,398R | ❌ no |

**El haircut de supervivencia (~+0,32 a +0,40R) es MAYOR que todo el edge sesgado (+0,26R).**
Al quitar el sesgo, el punto estimado **cruza a negativo** en las 4 configuraciones. Ninguna es
robusta: no hay ninguna con IC95>0 en las 3 semillas; los IC95 pooled rozan o están por debajo
de 0. Es exactamente lo que predijo la caracterización del baseline: la supervivencia inflaba la
cola de ganadores, y al incluir los que desaparecieron (muchos por caídas/quiebras, pero también
absorciones) la cola se recorta y el edge se evapora.

## 1-min vs 5-min

**1-min NO mejora a 5-min: mete ruido.** 5-min EOD −0,069R vs 1-min EOD −0,110R; el acierto cae
de 43% a 33% (rango de apertura de 1 minuto = stop minúsculo = más stop-outs y colas más salvajes),
y la única configuración con IC95 pooled **enteramente negativo** es 1-min+trailing (−0,138R,
[−0,216, −0,054]). Más granularidad = más coste de ejecución relativo y más falsas confirmaciones,
sin compensación. Tras slippage, el 1-min es estrictamente peor.

## Conclusión honesta

- **El edge del ORB-confirmado NO sobrevive al quitar el sesgo de supervivencia.** El +0,26R
  del baseline era, en lo esencial, un artefacto de muestrear solo supervivientes. Con universo
  point-in-time el número honesto es **ligeramente negativo** (≈ −0,06 a −0,14R/trade) y, sobre
  todo, **no es robusto**: indistinguible de cero-a-perder en las 3 semillas y los 2 timeframes.
- **1-min no rescata nada:** es igual o peor que 5-min, puro ruido tras costes.
- **Veredicto:** con los datos limpios que ahora permite Massive Starter, el ORB intradía
  largo-only **no es un edge operable**. Coincide con la literatura honesta (réplicas de
  "stocks in play" que rompen break-even con poco slippage) y cierra la hipótesis abierta en
  `docs/intradia_busqueda_edge.md`: *"universo point-in-time (quitar supervivencia)"* → **el edge
  se lo come la supervivencia.**

### Salvedades (para no sobre-vender el negativo)

- Muestra ~700–750 ops/config (menor que los 6.603 breakouts del dataset sesgado); los IC95 son
  anchos y en EOD cruzan 0, así que el mensaje preciso es **"no hay edge robusto"**, no "pierde
  mucho y seguro". La dirección (de +0,26R a negativo) y su consistencia entre 4 configs × 3
  semillas sí son claras.
- El universo y periodo no son idénticos al baseline (proxy de dollar-vol vs `Value.Traded`+cap;
  24 meses de Massive). Parte del gap podría venir de composición de universo, no solo de
  supervivencia — pero la magnitud y el signo coinciden con la hipótesis de cola inflada, que es
  el mecanismo esperado.
- Exclusión de mega-caps por **proxy de dollar-vol** (percentil 95), no market cap real.
- Starter da ~5 años de minuto; aquí se usaron 24 meses para que termine en tiempo razonable.
  Ampliar el periodo/muestra es el siguiente paso barato (hay cache por ticker-día), pero no se
  espera que cambie el signo: las 3 semillas ya apuntan igual.

**Entregable:** el ORB-confirmado, limpio de supervivencia, no es operable. Honestidad > resultado
bonito: el edge bonito era del sesgo.
