# Búsqueda de NUEVAS familias intradía (Massive 5m + EOD) — veredicto honesto

**Fecha:** 2026-10-05 · **Datos:** Massive (5 min y EOD, plan Starter, ~5 años) ·
**Código:** `backtest_intradia_massive.py` · **Solo research/paper.**

## Contexto

La búsqueda previa del repo (`docs/intradia_busqueda_edge.md`) ya exprimió el intradía con
rigor OOS y dejó un mapa claro: momentum genérico MUERTO (−0,3R), pullback-VWAP muerto,
mean-reversion a VWAP pierde, meta-ML sin lift, filtro de régimen sin lift. El único
superviviente era el **ORB de stocks-in-play confirmado** (+0,26R) — y ese acaba de caer al
quitar la supervivencia (`docs/orb_massive_resultado.md`). Conclusión del muro anterior:
quedaba probar **familias que aún no se habían medido**.

Aquí se prueban cuatro NUEVAS, sobre un **universo fijo de ETFs líquidos** (SPY/QQQ/IWM +
sectores), que por construcción **no tienen sesgo de supervivencia** (ninguno se deslista),
con datos 5-min limpios de Massive (mejores que el IEX 15m de antes). Rigor idéntico al
resto del repo: point-in-time sin look-ahead, costes realistas, IC95 por bootstrap y split
train/test. Honestidad > resultado bonito.

## Fase 1 — las cuatro familias nuevas (casi todas mueren)

| familia | qué es | resultado | veredicto |
|---|---|---|---|
| **A. Overnight vs intradía** | comprar MOC, vender MOO (noche) vs open→close (día) | noche Sharpe SPY −0,13 / QQQ 0,18 / IWM 0,14; día negativo; B&H 0,69-0,70 | noche sola NO bate a B&H (el coste del churn diario se la come) |
| **B. ORB de índice** (SPY/QQQ/IWM, 5m) | ruptura confirmada del rango de apertura, stop OR-opuesto, salida EOD | SPY −0,083R (IC95 [−0,158, −0,010], **PIERDE**); QQQ −0,008R; IWM −0,001R (IC incluyen 0) | **sin edge**; el ORB de stock NO transfiere al índice |
| **C. Momentum última hora** | seguir la dirección del día desde las 15:00 | Sharpe SPY −0,76 / QQQ −0,28 / IWM −0,91 | pierde (la última hora no continúa) |
| **D. Gap-fade en ETF** | contrariar el hueco de apertura hacia el cierre | exp ~0; XLE +0,111%/op pero IC95 [−0,002, +0,229] | nada robusto (XLE huele a multiple-testing) |

**Lectura honesta de la Fase 1:** confirma el patrón del repo — la mayoría de ideas
intradía no sobreviven la validación. Hallazgo limpio y útil: **el ORB de índice PIERDE**
(SPY con IC95 que excluye el 0 por abajo). Es decir, el edge del ORB de stocks-in-play era
**selección idiosincrática de acciones en juego**, no un efecto de ruptura de mercado; a
nivel de índice el breakout revierte. Eso cierra la puerta a "cablear el ORB sobre SPY/QQQ".

## Fase 2 — refinar los hilos no del todo muertos (ojo: exploratorio, multiple-testing)

- **Última hora en REVERSIÓN** (fade en vez de continuación): Sharpe −1,9. Peor aún. La
  última hora es **ruido** neto de coste en ambas direcciones. Cerrada.
- **Gap-fade subiendo el umbral** (0,5/1,0/1,5%): la expectativa sube pero la muestra se
  desploma y **todos los IC95 incluyen el 0** (SPY>1,5%: n=39). Espejismo de muestra pequeña.
  Cerrada.
- **Overnight con filtro de TENDENCIA** (mantener la noche solo si el cierre > SMA200): **esto
  sí mejora** y de forma consistente. Es el único superviviente.

## El superviviente: OVERNIGHT + filtro de tendencia

Idea: la anomalía del **overnight drift** (la prima de la renta variable se gana sobre todo
*de noche*, close→open; el intradía aporta poco — documentado en Cooper-Cliff, Lou-Polk-
Skouras y réplicas). Aquí se opera así: cada día, si el ETF cerró ayer **por encima de su
SMA200**, se mantiene la posición de **cierre a apertura** (MOC→MOO); si no, efectivo. El
filtro evita las noches en tendencia bajista (donde el drift se da la vuelta).

Validación a fondo (coste central **1 bp/lado**; IC95 del Sharpe por bootstrap; split
train/test 60/40):

| ETF | Sharpe | IC95 Sharpe | CAGR | maxDD | % noches en mercado | train/test | B&H Sharpe |
|---|---|---|---|---|---|---|---|
| **QQQ** | **1,30** | **[0,25, 2,43]** | +15,2% | −9,6% | 70% | **1,29 / 1,31** | 0,70 |
| SPY | 0,76 | [−0,27, 1,81] | +5,8% | −9,6% | 70% | 0,72 / 0,81 | 0,69 |
| IWM | 0,73 | [−0,41, 1,88] | +8,6% | −14,6% | 60% | 0,67 / 0,81 | 0,28 |

Sensibilidad al coste (Sharpe): QQQ 0,5bp→**1,52** · 1,0bp→**1,30** · 1,5bp→**1,08**.

**Por qué QQQ es creíble (no un espejismo):**
1. **IC95 del Sharpe excluye el 0** [0,25, 2,43] — el único de todo el estudio intradía que
   lo hace.
2. **Consistencia OOS notable:** train 1,29 / test 1,31 (casi idénticos en las dos mitades).
3. **Aguanta el coste:** incluso a 1,5 bp/lado (pesimista para MOC/MOO en QQQ) sigue en
   Sharpe 1,08, por encima del B&H de QQQ (0,70).
4. **Mejor perfil que comprar-y-mantener:** Sharpe 1,30 vs 0,70 y menos exposición (solo el
   70% de las noches, nada de día), con maxDD −9,6% (QQQ B&H llegó a ~−35% en 2022).
5. **Base económica real:** no es data-mining puro, reproduce una anomalía documentada.

**SPY e IWM** apuntan en la misma dirección (y el de IWM mejora muchísimo sobre su B&H de
0,28) pero sus **IC95 incluyen el 0**: con ~5 años no se distinguen de cero. Sugerentes, no
probados; sirven como diversificación dentro de una cesta overnight, no en solitario.

## Veredicto y qué hacer

| estrategia | ¿edge? | acción |
|---|---|---|
| **Overnight + tendencia (QQQ)** | **SÍ, robusto** (Sharpe 1,3; IC95 excluye 0; train≈test) | **Candidata a PAPER ya.** Medir el slippage REAL de MOC/MOO antes de nada |
| Overnight + tendencia (SPY/IWM) | sugerente (IC95 incluye 0) | añadir como cesta overnight junto a QQQ, no sueltas |
| ORB de índice | NO (SPY pierde) | descartado; el ORB es de stock, no de índice |
| Última hora (momentum o reversión) | NO | descartado |
| Gap-fade en ETF | NO robusto | descartado |

### Cómo operar la superviviente (para cablear en paper)

- **Universo:** QQQ (núcleo) + SPY/IWM (diversificación overnight).
- **Señal diaria (al cierre):** ¿cierre de hoy > SMA200? → mantener la noche. Si no, efectivo.
- **Ejecución:** comprar en la subasta de cierre (MOC), vender en la de apertura (MOO).
- **Sin apalancar.** La exposición es solo overnight y solo en tendencia alcista.
- **Riesgo propio:** se come los **gaps de apertura** (es justo la exposición que cobra la
  prima). El maxDD −9,6% ya lo refleja.

## Avisos honestos

1. **Un superviviente de muchos tests.** Se probaron 4 familias + refinamientos; QQQ es el
   que pasa. La consistencia train/test y la base documentada mitigan el riesgo de
   overfitting, pero trátalo como **candidato a validar en paper**, no como certeza.
2. **Un solo régimen (~5 años).** 2021-2026 fue favorable al overnight drift tecnológico. La
   literatura dice que la anomalía **decae** con el tiempo; vigila que no se apague.
3. **La ejecución es el muro.** El Sharpe depende del coste real de MOC/MOO. 1 bp/lado es
   plausible en QQQ pero hay que MEDIRLO en vivo (paper) antes de arriesgar nada.
4. **Riesgo overnight genuino:** mantienes durante cierres de mercado; un gap adverso grande
   (sorpresa macro/earnings del índice) te pilla dentro. Es el precio de la prima nocturna.

**En una frase:** de cuatro familias intradía nuevas, tres mueren limpio (incluido el ORB de
índice, que PIERDE) y una sobrevive: **el overnight de QQQ filtrado por tendencia**, un edge
robusto y documentado (Sharpe ~1,3, consistente OOS, tolerante al coste) que **sí merece
pasar a paper** — con los ojos puestos en el slippage de las subastas y en que la anomalía no
se apague.
