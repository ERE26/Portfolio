# Estrategias NUEVAS de menor frecuencia (días–semanas) con datos limpios Massive

**Fecha:** 2026-10-06 · **SOLO RESEARCH / PAPER — NUNCA dinero real, nunca órdenes en vivo.**
Datos **Massive** (grouped EOD survivorship-free, deslistados incluidos, universo
point-in-time). Panel diario construido una vez y cacheado en scratchpad; reutilizado por
todos los backtests (≈0 llamadas nuevas a la API).

Esta ronda **NO prueba más microestructura intradía** (ya está cerrada: muere por *entry_lag*
= rebote bid-ask, ver `revalidacion_massive_sintesis.md`). Prueba **anomalías de menor
frecuencia** (horizonte días–semanas), cross-seccionales, **robustas a *entry_lag*** porque NO
dependen del bid-ask bounce. Son familias que el repo aún no había medido con rigor:
momentum cross-seccional, reversión (mensual/multidía/semanal-residual), baja-vol/baja-riesgo,
overnight cross-seccional y trend/timing.

**Gate de la casa:** net>0 robusto OOS, tras **coste + turnover**, **survivorship-free**,
**entry_lag T+1**, **DSR ≥ 0,95**, IC95 que no cruce 0, PBO < 0,5, y batir al benchmark
(SPY / cartera equiponderada del propio universo) en riesgo-ajustado. Ante la duda, regla 7
→ `net_positivo=false`.

---

## (a) Tabla maestra

| Estrategia | Familia | Neto anual | Sharpe neto | Turnover | entry_lag | DSR / PBO / OOS | ¿net? |
|---|---|---|---|---|---|---|---|
| **52w-high (George-Hwang)** `high52w_gh` | Momentum XS (anchoring) | D10 LO +10,8% · L/S +1,8% | 1,04 LO · 0,08 L/S | 0,68/mes · L/S 0,52 | **SOBREVIVE** (nivel 252d, lag T+2 L/S muere) | 0,64 / 0,69 / test>train pero es beta; L/S cruza 0 | ❌ |
| **Momentum 12-1 amplio** `mom12_1_amplio` | Momentum XS | D10 +29,2% · L/S +27,0% | 1,01 LO · 1,03 L/S | ~33%/mes · L/S 36% | **SOBREVIVE** (lag mejora) | 0,81 / 0,69 / + ambas mitades, degrada | ❌ |
| **Residual momentum CAPM** `resmom_capm_amplio` | Momentum XS (idiosincrático) | D10 +2,7% · L/S −1,2% | 0,18 LO · −0,07 L/S | 0,91/mes | SOBREVIVE (trivial: no hay edge) | 0,24 / 0,24 / L/S cambia de signo | ❌ |
| **Reversión mensual L/S** `rev_mes_ls` | Reversión | −12,3% | −0,45 | 340% 2-sided/mes | pasa (irrelevante: sin edge) | 0,004 / — / colapsa (train 0,02→test −1,02) | ❌ |
| **Reversión extremo multidía** `rev_extremo_md` | Reversión | −11,5% @20bps | −0,31 | 90,8x libro/año | pasa (T+1 bate cierre-T) | 0,016 / 0,02 / colapsa (test −44,7%) | ❌ |
| **Reversión semanal residual** `rev_residual_sem` | Reversión | −11,2% @10bps | −0,47 | 335% 2-sided/sem | **SOBREVIVE** (señal real) | 0,003 / 0,12 / neto<0 ambas mitades | ❌ |
| **Baja vol total (q20 252d)** `lowvol_decil` | Baja-vol | +5,2% | 0,54 | 0,38/mes | SOBREVIVE (mejora al retardar) | 0,335 / 0,82 / degrada (test 0,30) | ❌ |
| **Baja-beta LO (BAB retail)** `lowbeta_lo` | Baja-riesgo | +6,5% | 0,64 | 0,43/mes | SOBREVIVE (mejora al retardar) | 0,40 / 0,05 / + consistente pero <SPY | ❌ |
| **Baja vol idiosincrática (IVOL)** `ivol_bajo` | Baja-vol | +9,1% | 0,77 | 1,20/mes | SOBREVIVE | 0,577 / 0,55 / estable pero <bench | ❌ |
| **Dual momentum global (GEM)** `dualmom_gem` | Trend/Timing | +17,5% | 1,10 | 1,5 cambios/año | SOBREVIVE | 0,46 abs / 0,27 / estable, ambas <SPY | ❌ |
| **Rotación sectores US + gate** `sectrot_us` | Trend/Timing (rotación) | +14,9% | 1,18 | 0,465/rebal | SOBREVIVE | **0,968** / 0,29 / train 0,99 test 1,33 | ❌ |
| **Donchian multi-lookback CTA** `trendbreak_donchian` | Trend-following | +10,9% | 1,22 | 7,67x/año | SOBREVIVE | 0,585 abs / 0,14 / degrada pero + | ❌ |
| **Overnight turn-of-month cesta** `tom_onbasket` | Overnight XS / estacional | **−7,7%** @3bps | −1,45 | 92,6 lados/año | sin look-ahead (calendario) | ≈2e-7 / ≈1 / neg. IS y OOS | ❌ |

Leyenda: **D10** = decil top / **LO** = long-only / **L/S** = largo-corto dólar-neutral /
**abs** = DSR del retorno absoluto (no del exceso vs SPY). Benchmarks en la misma ventana:
SPY B&H Sharpe ≈ 1,2–1,45 según tramo; cartera equiponderada del universo Sharpe ≈ 0,99–1,01.

---

## (b) VEREDICTO honesto

**NO existe ninguna estrategia nueva neto-positiva robusta OOS tras coste + turnover +
survivorship + entry_lag.** Cero supervivientes (tras verificación adversarial: `[]`).
**Ninguna de las 13** se cablea. El único superviviente del repo sigue siendo el **overnight
QQQ + SMA200** (en paper, bloqueado por slippage MOC/MOO); los diversificadores TSMOM/rotación
son beta/trend, no alfa.

El **hallazgo estructural más importante de la ronda** (y lo que la distingue de todo lo
intradía ya muerto): **estas familias lentas SÍ sobreviven a *entry_lag*** — retrasar la
entrada a la apertura de T+1 (incluso T+2) no destruye la señal, a menudo la mejora. Es decir,
cuando hay algo, **no es rebote bid-ask / microestructura**; es una señal lenta real y
ejecutable por un taker retail. Eso resuelve la pregunta que dejó abierta el intradía. **Pero
sobrevivir a entry_lag no crea alfa que no existe.** Cada familia muere por otra vía:

### Por qué falla cada familia

- **Momentum cross-seccional** — Es la anomalía más real del lote (12-1 da deciles
  **monótonos** D1→D10 y bate a SPY en retorno absoluto +29% vs +18%), **sobrevive entry_lag y
  coste** (turnover trivial ~33%/mes, 0,8%/año). Muere en el **gate estadístico**: DSR 0,81 <
  0,95, IC95 del L/S cruza 0, PBO 0,69, y sobre todo **pierde contra SPY en Sharpe** (1,01 vs
  1,46) con mucho peor drawdown (−25/−31% vs −7,6%) — gana CAGR, pierde riesgo-ajustado, mismo
  patrón que mató al residual momentum. El 52w-high no reproduce ni el anchoring (D6-D8 > D10,
  no monótono). El residual CAPM, con el universo ancho que pedía el doc, sigue plano
  (deciles no-monótonos, L/S neto-negativo, cambia de signo OOS) → **cierra la puerta al
  momentum residual para retail.** Causa raíz común: muestra corta (~3 años) y **100% alcista**,
  sin un crash de momentum que permita al DSR cruzar 0,95.

- **Reversión** — **Familia cerrada: no hay reversión explotable en el universo líquido
  moderno.** Mensual L/S: la señal es de signo **contrario** (si acaso continuación débil,
  rank-IC +0,025 t=1,0), neto −12%, DSR 0,004, colapsa OOS. Extremo multidía: edge bruto
  marginal (+6,6%/año) **por debajo del coste** (90,8x turnover → 18%/año), colapsa OOS
  (test −44,7%). Semanal **residual**: aquí sí hay **señal bruta real** (Sharpe 0,43,
  sobrevive entry_lag, confirma el lift de Blitz et al. ~7x sobre total-return) pero
  **break-even ≈4,8 bps/lado** < coste taker retail (10-20 bps) por el turnover semanal de
  deciles → neto robustamente <0. **Se archiva junto al pairs: señal genuina, arbitrada por
  debajo del coste de ejecución.** La reversión vive en micro-caps ilíquidas / microestructura,
  inaccesible para taker retail.

- **Baja-vol / baja-riesgo** — Las tres variantes (vol total 252d, beta, IVOL residual FF3)
  son **nominalmente positivas en absoluto** (+5 a +9%/año), **sobreviven entry_lag** (señal
  252d, no bid-ask) y tienen **turnover bajo/realista** — pero **no existe prima** en esta
  muestra: quedan por debajo de SPY **y por debajo de su propia cartera equiponderada** del
  mismo universo (low-vol 0,54 vs EW 0,99; IVOL 0,77 vs EW 1,01). El contraste
  naif-vs-sector-neutral (low-vol) demuestra que **no es un tilt sectorial encubierto**: es
  low-vol puro que simplemente se rezagó en un mercado liderado por mega-caps/alta-beta. La
  versión fondo BAB L/S apalancada **pierde aún más** (la pata corta de alto-beta se disparó en
  el bull; neto −7,8%, DSR 0,03). DSR 0,34-0,58, IC95 cruza 0, PBO 0,55-0,82. **Caveat
  decisivo:** la baja-vol/baja-beta gana en ciclo **completo** (su defensa está en los bear
  2008/2020/2022) y la muestra (post-bear, 3-3,8 años) **excluye justo ese régimen**. El
  juicio es "sin edge neto en esta ventana alcista", NO "anomalía muerta para siempre".

- **Overnight cross-seccional (turn-of-month)** — **Muere en BRUTO, antes de costes**
  (break-even −5,2 bps/lado: pierde incluso a coste 0), con **signo invertido** respecto al
  clásico Ariel/Lakonishok-Smidt en las 5 variantes (broad, inverse-dvol, small/mid, SPY, QQQ).
  Hallazgo honesto y accionable: en 2023-2026 survivorship-free **las noches TOM son las PEORES
  del mes** (−11 bps) y la prima overnight vive en el **resto** del mes (+9 bps). Confirma y
  generaliza el "evitar TOM" que el overlay de QQQ ya había encontrado (no era artefacto de
  2022 ni de un solo nombre). La churn-reduction es mecánicamente correcta (−82% turnover) pero
  irrelevante porque la señal es negativa. DSR ≈ 2e-7, IC95 excluye 0 por el negativo.

- **Trend / timing (dual momentum, rotación sectores, Donchian CTA)** — Las tres dan **retorno
  neto positivo, robusto OOS, con turnover bajísimo y sobreviven entry_lag y coste** — son
  trend/managed-vol **ejecutables**, justo como admite la hipótesis ("no alfa puro"). Pero las
  tres **son BETA, no alfa**: no baten a comprar y mantener SPY (GEM 1,10 / Donchian 1,22 /
  sectores 1,18 vs SPY 1,21-1,45), el **exceso vs SPY tiene IC95 que cruza 0 y DSR ≈ 0**
  (GEM exceso 0,005; Donchian 0,0002; sectores 0,968 absoluto pero exceso cruza 0). Mismo
  veredicto que TSMOM y rotación Europa ya archivados (⚠️ empate = beta). **Matiz crítico de
  datos:** su única razón de ser —la **convexidad / protección de cola en crisis**— es
  **INVERIFICABLE** porque Massive Starter sólo sirve ~5 años y el **warmup de momentum 12m se
  come el bear de 2022 entero**. La defensa nunca tuvo que funcionar (GEM: BND 1/48 meses;
  sectores: IEF 1/48 meses). `sectrot_us` es el único que cruza DSR≥0,95 (0,968) pero es un
  DSR alto para **exposición beta en un bull de régimen único = lo esperado, no evidencia de
  alfa**; falla su propia vara (no bate SPY risk-adjusted).

### Resumen de causas de muerte

| Causa | Familias afectadas |
|---|---|
| **Es beta, no alfa** (no bate SPY risk-adjusted, exceso IC95 cruza 0) | Momentum XS (LO), baja-vol/beta/IVOL, GEM, sectores US, Donchian CTA |
| **Gate estadístico** (DSR<0,95, PBO>0,5, L/S cruza 0) | Momentum 12-1, 52w-high, residual CAPM |
| **Sub-coste / turnover** (break-even < coste taker retail) | Reversión semanal residual, reversión extremo multidía |
| **Señal inexistente o de signo contrario** | Reversión mensual, residual CAPM, overnight TOM |
| **Tesis inverificable por falta de muestra** (bear en warmup) | GEM, sectores US, Donchian CTA, baja-vol/beta (defensa de cola) |

---

## (c) Qué se aprendió por familia

**Momentum cross-seccional.** Medido **limpio por primera vez en el repo**. El momentum 12-1
es una anomalía **genuina** (deciles monótonos, sobrevive entry_lag y coste) — el fracaso más
limpio del lote. Pero sobre muestra corta 100% alcista **no alcanza DSR 0,95 ni bate a SPY en
riesgo-ajustado**. El 52w-high no reproduce el anchoring. El residual CAPM, con el universo
ancho PIT que el doc previo reclamaba, **sigue sin funcionar** → `mom12_1_amplio` queda como el
**baseline honesto** contra el que medir; el momentum residual para retail queda **cerrado**.
Reevaluar sólo con muestra multi-ciclo que incluya un momentum crash.

**Reversión.** Familia **cerrada** en líquidas. No hay reversión mensual (señal contraria/nula);
la de movimiento extremo es sub-coste y colapsa OOS; la **semanal residual tiene señal real
(confirma Blitz et al.) pero break-even ~4,8 bps < coste taker** → se archiva junto al pairs
("señal genuina, arbitrada por debajo del coste de ejecución"). El edge de reversión vive en
microestructura / micro-caps ilíquidas, inaccesible. Palanca futura (sólo si se reabre):
atacar el **turnover** (holding mensual, bandas de no-rebalanceo, hedge con futuros), no el alfa.

**Baja-vol / baja-riesgo.** No hay prima de baja-vol/baja-beta en esta ventana: pierde contra
su propio universo equiponderado. Research de valor: el contraste **sector-neutral ≈ naif**
prueba que NO es apuesta sectorial encubierta; y el contraste **retail-LO vs fondo-BAB-L/S**
prueba que el premio BAB tampoco se rescata apalancando (la pata corta cara lo mata en bull).
Conclusión honesta tipo repo: es **beta defensiva, no edge accesible**. El resultado **acota**
(sin edge en bull), NO refuta la anomalía de ciclo completo — requeriría muestra con bear.

**Overnight cross-seccional.** El turn-of-month overnight en cesta amplia **no existe y tiene
signo contrario** al clásico: las noches TOM son las peores del mes. Confirma survivorship-free
y en cesta amplia el "evitar TOM" del overlay de QQQ. El único overnight que sobrevive sigue
siendo QQQ+SMA200, y **no se mejora** metiéndole timing TOM.

**Trend / timing.** Tres variantes nuevas (dual momentum US/intl/bono, rotación sectorial con
gate a bono, Donchian CTA cross-asset) — todas **ejecutables, baratas, robustas OOS y a
entry_lag**, pero todas **beta sin alfa** sobre SPY, coherente con TSMOM y rotación Europa ya
archivados. Lección de datos **transversal y limitante**: con Massive Starter (~5 años) el
**bear de 2022 cae en el warmup de 12m**, así que la **tesis de protección de cola —la única
razón de ser de estas familias— es inverificable**. Su valor real está en el drawdown que
evitan en un bear de verdad (como el overnight-QQQ, que sobrevive por **menos exposición**, no
por más retorno), y eso aquí no se puede medir. Reconsiderar sólo con histórico largo
(2008/2020/2022 completos) y total-return real (dividendos).

---

## Recomendación

1. **No cablear nada de esta ronda.** Cero supervivientes; `nuevo_libro.procede = false`.
2. **Overnight QQQ sigue siendo el único vivo** (en paper): la decisión depende del slippage
   MOC/MOO real, no de más backtests.
3. Reabrir estas familias **sólo** con muestra multi-década que incluya varios bears (momentum
   crash + crisis de cola) y total-return. Hasta entonces, el valor de esta ronda es
   **NEGATIVO**: saber qué no funciona evita arriesgar capital en humo.
4. Confirmado el **principio estructural**: las anomalías lentas (días–semanas) sobreviven a
   entry_lag (no son bid-ask bounce) — pero eso sólo dice que **si** hubiera edge sería
   ejecutable; en esta muestra el edge o es beta, o no cruza el gate estadístico, o está
   sub-coste, o no existe.
