# Overlays sobre el overnight de QQQ (cierre > SMA200): qué sobrevive al DSR

**Fecha:** 2026-10-05
**Script reproducible:** `backtest_overnight_overlays.py`
**Datos:** Massive/Polygon EOD, split-adjusted, plan gratis (survivorship-free).
Ventana efectiva **2021-10-06 → 2026-10-01** (~5 años naturales; **1052 noches**
con SMA200 ya formada). Sin claves FRED ni Alpaca en el entorno (ver *Caveats*).
**Veredicto en una línea:** **ningún overlay bate al baseline OOS tras el haircut
por multiple-testing (DSR).** El hallazgo con más valor no es un overlay, sino la
**fragilidad a coste** del propio baseline.

---

## 1. Baseline replicado (control)

Overnight de QQQ: comprar en el cierre (MOC) y vender en la apertura siguiente
(MOO) **solo si el cierre del día estaba por encima de su SMA de 200 sesiones**.
Retorno overnight bruto = `open[t+1] / close[t] − 1`; la señal (`close[t] > SMA200[t]`)
se conoce en el cierre de `t`, así que **no hay look-ahead**. Coste round-trip
cobrado solo las noches operadas.

| Coste round-trip | Sharpe anual | CAGR | maxDD |
|---|---|---|---|
| 0 bps | **1.55** | 16.7% | −9.0% |
| **2 bps** | **1.15** | 12.0% | −9.6% |
| 5 bps | 0.55 | 5.2% | −10.5% |
| 10 bps | −0.46 | −5.1% | −21.3% |

- A 2 bps: train 0.94 / test 1.34; **IC95 bootstrap del Sharpe = [0.34, 1.97]**
  (bloques de 10, 2000 resamples). Positivo pero ancho: 4 años de overnight dan
  poca potencia estadística.
- **Replicación vs. la cifra de referencia (1,30):** mi baseline da **1,15 @2bps**
  y **1,55 @0bps**, que la *encajonan*. La diferencia se explica por el supuesto de
  coste y por la ventana de 5 años del plan gratis (no la serie larga original).
- **El edge es casi exclusivo de QQQ:** mismo baseline da **SPY 0,66** e
  **IWM 0,53** @2bps. No es un fenómeno genérico de "el overnight", sino del
  Nasdaq-100 en este régimen.

> **Fragilidad a coste (el titular honesto):** el drift overnight es tan pequeño
> que el Sharpe cae de 1,55 a 0,55 entre 0 y 5 bps y se vuelve negativo a 10 bps.
> **Antes que cualquier overlay, hay que medir el coste real de los cruces
> MOC/MOO de QQQ.** Si el round-trip efectivo (spread de subasta + imbalance +
> comisión) supera ~3-4 bps, el baseline deja de ser interesante.

---

## 2. Overlays probados (27 configuraciones por activo)

Todos actúan **sobre** el baseline (solo pueden *saltar* noches o *escalar*
exposición). Selección honesta: en las familias con umbral se elige el mejor en
**train** y se reporta **test** (OOS verdadero). DSR = Deflated Sharpe Ratio
(Bailey-López de Prado) usando los Sharpe de **las 27 pruebas** como corrección
por multiple-testing.

**QQQ, coste 2 bps.** `n_trials = 27`; Sharpe-umbral esperado por puro azar entre
27 pruebas ≈ **0,98** anual. Para "creérselo" exijo **DSR > 0,95**.

| Familia | Mejor variante | full | train | test | maxDD | **DSR(27)** |
|---|---|---|---|---|---|---|
| **B_turn_of_month** | evita_TOM | **1.73** | 1.75 | 1.73 | −6.0% | **0.934** |
| B_dow_drop | drop_lunes | 1.48 | 1.25 | 1.70 | −8.6% | 0.837 |
| A_rvol_regime | skip rvol>p70 | 1.33 | 1.63 | 1.10 | −9.6% | 0.764 |
| E_vol_target | target 1.0%/día | 1.33 | 1.10 | 1.53 | −8.9% | 0.758 |
| A_vix_backwardation | skip ratio>p70 | 1.28 | 1.21 | 1.34 | −9.6% | 0.722 |
| A_vix_spike | skip Δvix>p95 | 1.22 | 1.07 | 1.36 | −8.9% | 0.689 |
| B_fomc | skip pre-FOMC | 1.04 | 0.84 | 1.22 | −9.8% | 0.550 |
| C_breadth | QQQ+SPY+IWM>SMA200 | 1.00 | 0.66 | 1.29 | −9.6% | 0.515 |
| D_macro_blackout | blackout FOMC | 0.75 | 0.55 | 0.94 | −9.6% | 0.329 |
| C_trend | sobre SMA20 | 0.36 | 0.23 | 0.49 | −10.5% | 0.107 |
| C_trend | momentum 5d > 0 | 0.14 | −0.01 | 0.28 | −9.6% | 0.045 |

### Lectura por familia

1. **Régimen de VIX / vol (A).** Proxy (sin FRED): VIXY/VIXM (futuros de VIX corto
   vs. medio) para la estructura temporal, y vol realizada a 20 días. Saltar noches
   de *backwardation* o de vol alta **sube el Sharpe modestamente** (1,2-1,3) pero
   **no supera el umbral de azar** (DSR 0,69-0,76). El efecto es real-ish pero
   pequeño y no distinguible del ruido con 4 años.
2. **Estacionalidad (B).** La señal más fuerte del estudio: **evitar las noches de
   turn-of-month** (últimos 1 + primeros 3 días hábiles) sube a Sharpe 1,73 con
   train≈test≈1,73 y maxDD −6%. **Pero DSR 0,934 < 0,95** y, sobre todo, el signo
   **contradice** el efecto turn-of-month clásico (que es *positivo* en el
   overnight): aquí el TOM es *peor*, lo que apunta a un artefacto del drawdown de
   2022 en una muestra corta. *Drop-lunes* (quitar el overnight de fin de semana,
   económicamente más plausible) da 1,48 pero DSR 0,837.
3. **Tendencia/breadth de corto plazo (C).** **Empeora.** Condicionar a momentum 5d
   o SMA20 destruye el edge (Sharpe 0,14-0,36). El drift overnight **no** es
   condicional a la tendencia de corto plazo: es una prima difusa que se captura
   mejor estando dentro siempre que `close > SMA200`.
4. **Blackout de earnings / macro (D).** Earnings **N/A** en un ETF. El blackout de
   FOMC (saltar el overnight del día del anuncio) **reduce** el Sharpe a 0,75:
   quitar noches sin un motivo de riesgo claro solo recorta exposición a la prima.
5. **Vol targeting (E).** Escalar por `vol_objetivo / vol_realizada(20)` (cap 2×)
   da 1,33 — mejora cosmética, DSR 0,758. No aporta sobre el filtro SMA200.

---

## 3. Multiple-testing: PBO y DSR cesta-ancha (el haircut que manda)

- **PBO (CSCV, 27 variantes, retornos mensuales):** QQQ **0,23**, SPY 0,14,
  IWM 0,16. Bajo (<0,5): el procedimiento de selección **no** está groseramente
  sobreajustado — el mejor in-sample tiende a seguir siendo decente OOS. Es un
  punto *a favor* de que el ranking es estable.
- **Pero el DSR es la restricción que ata.** Contando honestamente la búsqueda
  completa de la **cesta** (27 variantes × 3 activos = **N = 81**; Sharpe-umbral de
  azar ≈ **1,13** anual):

  | Candidato (QQQ) | Sharpe full | **DSR(N=81)** |
  |---|---|---|
  | evita_TOM | 1.73 | **0.885** |
  | drop_lunes | 1.48 | 0.753 |
  | baseline | 1.15 | 0.515 |

**Ni el mejor overlay llega a 0,95**, ni contando solo su activo (DSR 0,934) ni la
cesta (0,885). Y ese DSR es todavía **optimista**: los umbrales de las familias A se
fijan con cuantiles de *toda* la muestra (contaminación leve del nivel, que empuja
los overlays *al alza*), y la búsqueda real (elección de SMA=200, de coste, del
diseño de cada overlay) es mayor que 81. El sesgo de todas esas simplificaciones
va en contra de los overlays, así que la conclusión negativa es **conservadora**.

---

## 4. Veredicto y qué cablear

**No sobrevive nada al DSR.** Por tanto **no se recomienda cablear ningún overlay**
en `operativa_overnight.py`. En concreto:

- `operativa_overnight.py` debe quedarse como está: el **baseline puro** (mantener
  cada ETF de la cesta overnight sii su `close > SMA200`). El gate actual vive en
  `entrar()` (la línea `if not en_tend: ... continue`, ~L112) y es correcto.
- El único candidato con train≈test y consistencia en cesta (**evita_TOM**) se queda
  en DSR 0,885-0,934 y tiene una historia económica débil/contraintuitiva. **No** se
  cablea. Si alguien quisiera seguirlo **en paper como experimento de investigación**
  (nunca dinero real), el cambio mínimo y EXACTO sería añadir un gate extra justo
  después de obtener la señal en `entrar()`, sin tocar `_senal()`:

  ```python
  # operativa_overnight.py, dentro de entrar(), tras:  en_tend, ult, sma = s
  # EXPERIMENTAL — NO recomendado (falla DSR 0,95; aquí solo documentado):
  # saltar las noches de turn-of-month (últimos 1 + primeros 3 días hábiles del mes).
  et_mes = et.to_period("M")
  cal = pd.bdate_range(et_mes.start_time, et_mes.end_time)   # días hábiles del mes
  es_tom = et.normalize() in set(cal[:3]) | {cal[-1]}
  if es_tom:
      print(f"  {t}: noche turn-of-month -> EFECTIVO (overlay research-only)"); continue
  ```

  Marcarlo como *research-only* y revalidarlo con más historia y con el coste MOC/MOO
  real medido en el propio paper antes de darle ningún crédito.

**La acción de mayor EV no es un overlay: es medir el coste real de ejecución
MOC/MOO de QQQ.** El baseline pasa de Sharpe 1,55 a 0,55 entre 0 y 5 bps; todo el
valor del sistema vive o muere ahí.

---

## 5. Metodología y caveats (honestidad > resultado)

- **Survivorship-free:** Massive/Polygon sirve también deslistados; QQQ/SPY/IWM no
  están deslistados, pero la fuente no tiene sesgo de supervivencia. PIT: cada señal
  usa solo información conocida en el cierre de `t`.
- **Walk-forward:** corte temporal a la mitad de la muestra válida; umbral elegido en
  train, Sharpe reportado en test.
- **DSR** (Bailey-López de Prado) con corrección por nº de pruebas (27 por activo,
  81 en cesta); **PBO/CSCV** como contraste. **IC95** por bootstrap de bloques.
- **Costes** MOC/MOO realistas barridos a 0/2/5/10 bps (round-trip por noche operada).
- **Muestra corta:** ~4 años de overnight tras el warm-up de SMA200. Poca potencia;
  los IC95 son anchos. Más historia podría mover algún overlay, pero hoy no.
- **Sin FRED ni Alpaca en el entorno:** pese a lo indicado, `FRED_API_KEY` **no**
  está presente, así que el VIX **spot** y su término (VIX3M) **no** se usan; se
  aproximan con **ETFs de futuros de VIX (VIXY corto, VIXM medio)**, que arrastran
  *roll/decay* y no son el índice. Los índices `I:VIX` dan 403 en el plan gratis.
  Esto debilita específicamente la familia A (régimen de VIX): su lectura aquí es
  orientativa, no concluyente.
- **Dividendos:** las barras son *split-adjusted* pero **no** *dividend-adjusted*;
  en los ex-div de QQQ (~3ª semana de mar/jun/sep/dic, **fuera** de la ventana TOM)
  la apertura cae ~el dividendo, lo que resta ~0,5-0,6 %/año repartido en 4 noches.
  Sesga el overnight **a la baja**, así que los resultados son, por este lado,
  conservadores (y no explican la señal evita_TOM).
- **Reproducir:** `PYTHONPATH=. python3 backtest_overnight_overlays.py`
  (vuelca `datos/overnight_overlays_dump.json`, gitignored).
