# Refinamiento del intradía — 2026-10-06

**Solo research / paper. NUNCA dinero real.** Ninguna conclusión de este documento
autoriza una orden en vivo. El intradía sigue siendo un arnés de MEDICIÓN (slippage,
ejecución), no una fuente de capital.

Ronda de aprendizaje disparada por la petición de "seguir pensando en estrategias
mirando los ganadores de ayer, refinando, buscando nuevos filtros y haciendo nuevos
tests con Massive". Se caracterizaron los ganadores realizados del 2026-10-05 y se
corrieron **10 backtests** nuevos (ORB, fade, régimen, continuación, salidas), la mayoría
revalidados survivorship-free con Massive PIT. **Supervivientes tras verificación
adversarial: 0.** El valor de la ronda es, otra vez, conocimiento NEGATIVO: qué NO cablear.

---

## (a) Caracterización honesta de los ganadores de ayer (de VERDAD vs anécdota)

El día verde (+$1.276) **lo hizo UNA posición: QDEL +$2.429** (ORB corredora, corrió a su
TP 2R). Neto, todo lo demás del libro perdió ~−$1.150. Son ~10 trades: cualquier "patrón"
leído de ahí es anécdota hasta que el dataset completo lo confirme. Qué se sostiene y qué no:

**Lo que es REAL (confirmado en el dataset completo, 6.603 rupturas confirmadas):**
- **La única firma honesta del ganador ORB es la CONFIRMACIÓN de la ruptura**
  (`ext_breakout>0`, y cuanto más decidido el cierre sobre el nivel, mejor). Es el único
  rasgo con separación **monótona** de la expectativa: quintil mecha `meanR −0,05R` →
  quintil fuerte `+2,12R` (verificado hoy sobre `datos/orb_dataset.csv`). No gana más a
  menudo: **desplaza toda la distribución a la derecha** y deja correr el 2-3% de rupturas
  que explotan. La expectativa la paga la **cola derecha**, no el cuerpo (la mediana del
  trade pierde ~1R en todos los cortes).
- **La estructura de gestión importa más que la señal.** Lo que capturó QDEL no fue un
  filtro de entrada mágico: fue **TP 2R alcanzable + OCO que nadie pisó + dejar correr**.
  Esa es la lección del post-mortem y sigue siendo la correcta.

**Lo que es ANÉCDOTA (y el dato en vivo lo CONTRADICE):**
- **"Alto rvol = ganador": FALSO.** QDEL, el gran ganador, tenía `opening_rvol 1,89`
  (percentil **54**, bajo-medio). Los grandes perdedores STNE/PAGS tenían rvol **6,4/7,5**
  (percentil **94/95**). En el dataset completo el rvol **no separa**: AUC 0,526, quintiles
  planos (+0,62 vs +0,57), terciles no monótonos (bajo 0,545 / medio 0,752 / **alto 0,518**).
  El rvol alto de PICS/PAX es coincidencia, no causa.
- **"Entrar pronto / anti-chase": ya REFUTADO** en el post-mortem con los propios trades
  (PAGS perdió siendo la ruptura MENOS extendida). No hay relación extensión↔resultado.
- **Gap, régimen SPY, día de semana, ATR, Hurst: no separan.** Los fade-shorts que
  ganaron (RXO/INTR/PCVX) y las patas de pairs (CVX/PEP/GOOG) son micro-magnitudes dentro
  del ruido de un día alcista; no son evidencia de edge.

**Síntesis:** el denominador común REAL de los ganadores es *confirmación de ruptura +
gestión que deja correr la cola*, no ningún filtro de volumen/gap/régimen. Y, crucialmente,
esa firma es **real in-sample pero se evapora al corregir el sesgo de supervivencia** (ver
veredicto): no es cosechable.

---

## (b) Tabla maestra

Neto taker/maker en R (el R ya incluye slippage base 4bps entrada / 8bps stop).
"Robusta?" = supervive IC95>0 + OOS + DSR≥0,95 + entry_lag, survivorship-free.

| Hipótesis | Libro | Exp. base → filtrada | Neto taker / maker | entry_lag | DSR / OOS | ¿Robusta? |
|---|---|---|---|---|---|---|
| **orb_conf_decisiva** (ext>0.5, o combo or≥1.0 & bmin≤30) | orb | sesgado +0,61R → **limpio −0,076R → decisivo −0,008R / sano +0,019R** | ~−0,01/+0,02R (EOD); IC95 cruza 0 | estable (no es bid-ask) pero MOOT | sesgado DSR 1,0 / OOS 5/5 — **limpio: 1/3 semillas, pooled cruza 0** | ❌ se evapora |
| **orb_guardia_or** (or_range_atr≥1.0) | orb | +0,61R → +0,37R (baja por diseño) | mejora neto en spreads anchos; sin crear edge | N/A (guarda) | sube WR/mediana robusto OOS, pero **baja meanR** | ❌ higiene, no alfa |
| **orb_timing_gate** (bmin≤30; vetar >60) | orb | +0,61R → +0,64R | +0,52R @5bps *(sobre base sesgada)* | N/A (horizonte EOD) | **DSR 0,00**; delta IC95 cruza 0 train y test | ❌ mejora dentro del ruido |
| **orb_vol_regime** (atr_pct≥mediana) | régimen | +0,28R → +0,006R (realista) | neg. a 2-5bps extra | **DECISIVO: edge cae +0,60→+0,006R al corregir look-ahead** | DSR 0,117; OOS cruza 0 | ❌ look-ahead + recorta la cola |
| **orb_spy_no_gate** (vacuna: NO apagar en bajista) | régimen | no-gate +0,61R vs pro-bull +0,50R | — | N/A | bear-edge DSR 1,0 = **artefacto de supervivencia** | ❌ (vacuna: confirma NO cablear gate) |
| **fade_skip_rvol_alto** (excluir rvol≥1.5/3) | fade | base −0,018R → incl. +0,000R | no cruza coste | **falla** (se degrada con lag) | DSR ~0,04; diff IC95 cruza 0 en 3/3 folds | ❌ no replica (era muestra chica) |
| **fade_gap_band** (fadear 1-3%; vetar ≥15-20%) | fade | base +1,9bp → banda −2,3bp | −13,3bp / −3,7bp | **DECISIVO: +58,5bp apertura → +13bp a 5min** | DSR 0,152; OOS cambia de signo | ❌ microestructura de apertura |
| **fade_prefiltro_shortable** (px≥$5 & dvol≥$20M) | fade | base +0,150R *(cola)* → gate −0,016R | ~−0,02R / +0,019R (cruza 0) | estable en ~0 (nunca fue +) | DSR 0,04-0,07; OOS inconsistente | ❌ gate de EJECUCIÓN, no alfa |
| **drive_rs** (fuerza relativa top-K largo) | nuevo | sesgado +0,72R → **limpio spearman(rs,fwd)≈0** | top-K taker **−3,8bp** | **falla** (+26→+15bp; rs solo +4,2→−1,1bp) | DSR 0,40; OOS train<0 test>0 | ❌ artefacto superv.+solape |
| **vwap_pull** (pullback a VWAP en líder RS) | nuevo | base −0,020R → MAIN +0,130R (n=27) | −0,60R (taker) / −0,30R (maker) | sin señal (bruto≈0) | DSR 0,008-0,17; OOS cambia de signo | ❌ nulo limpio |
| **tail_exit** — quitar TP, dejar correr a EOD | salidas | TP2R −0,18R → **no-cap +0,27R** *(sobre 10k SESGADO)* | +0,27R (slip incl.) | estable (estructura, no bid-ask) | IC95 [+0,17,+0,38]; OOS estable — **pero todo sobre dataset con supervivencia** | ⚠️ **ver veredicto** |
| ↳ **tail_exit** — trailing Chandelier tras +3R | salidas | +0,0056R vs EOD puro | ruido | ruido | DSR 0,56; n_cola=8 | ❌ no bate EOD puro |
| **salidas_timestop_duds** (cortar si no +1R en N min) | salidas | EOD −0,137R → ts30 −0,078R | −0,078R (taker) | sobrevive, pero nivel negativo | DSR 0,849; delta IC95 cruza 0 | ❌ capa la cola derecha |

---

## (c) VEREDICTO

### ¿Qué mejora el intradía de forma ROBUSTA OOS? — **NINGUNO.**

Cero supervivientes tras verificación adversarial. No hay ningún filtro ni estrategia
intradía con regla+umbral que cruce el listón de la casa (IC95>0 survivorship-free + OOS +
DSR≥0,95 + entry_lag + neto de coste). Por la regla DEFAULT de la casa — *ante la duda,
`net_positivo=false`* — no se cablea nada como fuente de edge.

### El candidato que más tienta, y por qué TAMPOCO sobrevive: `tail_exit` (quitar el TP)

Es el único con `net_positivo=true` **medido sobre el dataset de 10.074 trades** (no-cap
+0,27R vs TP2R −0,18R, IC95 que no cruza 0, estable OOS y a outliers, supera entry_lag).
Parece el hallazgo del repo. **Pero no sobrevive a la verificación adversarial por una razón
única y decisiva:** su expectativa positiva vive **íntegra en la cola derecha gorda**
(top-1% de trades = 89% del EV), y esa cola es **exactamente lo que infla el sesgo de
supervivencia**. El ORB-confirmado medido limpio en Massive PIT (universo con deslistados)
es **−0,069R**: la cola de monstruos del dataset sesgado no existe en la misma magnitud en
datos survivorship-free. Quitar el TP sobre datos limpios **no** reproduce el +0,27R — solo
deja de capar una cola que, en limpio, está muy adelgazada. Es el mismo modo de fallo que
mató `orb_conf_decisiva`: la monotonía de la señal es real in-sample, pero el NIVEL positivo
es artefacto de supervivencia. **Conclusión: no cablear como estrategia de beneficio.**

> Matiz operativo (no es cableado nuevo): la operativa viva YA usa TP 2R realista + OCO
> protegido + tope 12% + breakeven, y eso capturó QDEL. Es una decisión de gestión ya
> tomada y validada en paper; **no** se deriva de este backtest ni se cambia por él. No se
> introduce "quitar el TP" ni "trailing tras +3R" en `operativa_orb.py`.

### Qué se rechaza y por qué (resumen de modos de fallo)

- **Supervivencia** (el "edge" era el universo líquido de hoy): `orb_conf_decisiva`,
  `drive_rs`, `tail_exit`, y la base de casi todo el libro ORB. El haircut limpio borra el
  nivel positivo aunque la forma de la señal sea real.
- **Look-ahead / entry_lag — rebote bid-ask o print de apertura** (desaparece al entrar 1
  barra después): `orb_vol_regime` (+0,60→+0,006R), `fade_gap_band` (+58,5→+13bp),
  `fade_skip_rvol_alto`, `drive_rs`. No es alfa predecible; es microestructura.
- **Confundir win-rate con expectativa** (sube el acierto recortando la cola donde vive el
  EV): `orb_vol_regime`, `orb_guardia_or`. Para una estrategia de cola, subir WR BAJA la
  expectativa.
- **Mejora dentro del ruido / DSR<0,95** (no supera multiple-testing): `orb_timing_gate`
  (DSR 0,00), `salidas_timestop_duds` (DSR 0,849), `vwap_pull` (DSR 0,01), el trailing de
  `tail_exit` (DSR 0,56).
- **Señal real pero sub-coste / inejecutable**: patas de pairs y fades de ayer; el gate de
  shortabilidad (`fade_prefiltro_shortable`) es higiene de ejecución válida (ETB 29%→90%)
  pero deja el fade en edge ~0.
- **Vacuna confirmada** (`orb_spy_no_gate`): NO existe ni debe añadirse un gate pro-tendencia
  SPY al ORB; apagarlo en bajista habría tirado los mejores setups (coherente con QDEL
  rompiendo con SPY plano). Status quo correcto = sin gate de régimen.

---

## (d) Refinamientos concretos a cablear — para supervivientes

**NINGUNO.** Supervivientes = 0 → no hay cableado de alfa que proponer al orquestador.

Para constancia, los únicos cambios *defendibles* que surgieron NO son alfa y por tanto
**no** se proponen como cableado de beneficio (quedan como notas de higiene por si el
experimento sigue en paper, marcados explícitamente como ejecución/riesgo, no expectativa):

1. *(higiene, opcional, NO alfa)* `operativa_fade.py` — gate de shortabilidad px≥$5 &
   dvol20d≥$20M para el corto: sube fills ETB 29%→90% y reduce varianza ~5x, SIN cambiar la
   expectativa (sigue ~0). Solo si el libro fade continúa en paper. **No habilita capital real.**
2. *(higiene, opcional, NO alfa)* `orb_guardia_or` / `orb_timing_gate`: gates de
   riesgo-cero que no perjudican, pero tampoco crean edge. Mientras el ORB base limpio sea
   ≤0, no justifican operar en real.

La prioridad REAL que dejan estos tests (y la única palanca que queda) es de EJECUCIÓN,
no de señal: **medir el slippage real del confirmado en el paper vivo** (`operativa_orb.py`)
y el slippage MOC/MOO del overnight QQQ — el único superviviente del repo. Eso es medición
en paper, no cableado de filtro.

---

## (e) Recordatorio de disciplina anti-overfitting

1. **Un día no es una muestra.** ~10 trades no validan ni refutan un filtro. QDEL fue cola +
   gestión, no una señal replicable. Todo filtro se valida en el dataset completo / Massive
   PIT, **nunca** con el P&L de un día (el "anti-chase" y el "rvol alto" ya murieron así).
2. **Default = `net_positivo=false`.** Ante IC95 que cruza 0, DSR<0,95, OOS inconsistente o
   muestra fina, se rechaza. No se busca la narrativa que salve el número.
3. **entry_lag es el juez.** Si el edge desaparece al entrar 1 barra después, era rebote
   bid-ask / print de apertura, no alfa.
4. **Win-rate ≠ expectativa.** En estrategias de cola, subir el acierto recortando la cola
   derecha DESTRUYE el EV. No celebrar un WR más alto.
5. **Survivorship primero.** El dataset rápido (Alpaca actual) infla ~+0,33R por
   supervivencia. Cualquier número bonito sobre él se confirma en Massive con deslistados o
   no cuenta.
6. **Multiple-testing se paga.** Más cortes probados = umbral DSR más alto. Probar 12 cosas y
   quedarse con la mejor NO es un hallazgo.
7. **Conocimiento negativo vale.** Saber qué NO cablear evita quemar capital en humo; es el
   producto principal de esta ronda, igual que de la revalidación maestra.
