# Scalping a fondo: ¿existe un scalping NETO positivo (muchas ops, beneficio pequeño)?

**Fecha:** 2026-10-05 · **Solo research/paper, NUNCA dinero real.**

Pregunta de Eduardo: operar MUCHAS transacciones sacando beneficios PEQUEÑOS pero
POSITIVOS. Este doc junta (a) la medición EN VIVO del scalping maker en paper —evidencia de
ejecución real, no simulada—, (b) el espacio de variantes enumerado de forma exhaustiva, y
(c) la teoría y la investigación previa, que convergen en la misma respuesta.

## Veredicto en una línea
**No.** Ningún scalping de muchas-operaciones/beneficio-pequeño es neto positivo para
nosotros (retail, taker o maker sin infraestructura). El rebate+spread que puede cobrar un
maker lo **devora la selección adversa**: tu orden límite se llena precisamente cuando el
precio ya va en tu contra. El dinero de scalping existe, pero lo cobra el **creador de mercado
con prioridad de cola, colocation y rebates reales** — inaccesible para nosotros.

## (a) Evidencia EN VIVO — scalp maker en paper (lo más fuerte: ejecución real)
`operativa_scalp.py`: entrada MAKER (límite 0,1% por debajo del último, para cobrar ~medio
spread en vez de pagarlo), TP +0,4% / SL −0,5%, muchos nombres in-play. Resultado real:

| Métrica | Valor |
|---|---|
| Fill-rate maker | **~90%** (las límite SÍ cruzan) |
| Operaciones cerradas | 14 |
| **Neto medio por operación** | **−0,39%** |
| Acierto | **~21%** |

El dato clave es la combinación: **fill-rate altísimo + neto negativo**. Eso ES la selección
adversa en estado puro — te llenan casi siempre, pero te llenan **cuando el precio ya se gira
en tu contra**, y luego se va al stop. Un 21% de acierto con TP +0,4% / SL −0,5% significa que
las pocas que ganan no cubren las que pierden. El rebate (~0,25 bps) ni se acerca a compensarlo.

## (b) Espacio de variantes (enumerado exhaustivamente, 8 diseños concretos)
La investigación enumeró y fusionó 8 variantes rigurosas antes de que el backtest pesado
tumbara el contenedor. Todas apuntan a la única vía con opción (maker), más un test taker:

1. **mm_spread_ancho** — market-making pasivo bilateral en spread ancho con filtro de
   toxicidad (OFI); rebate −0,25 bps.
2. **vwap_banda_maker** — quoting pasivo anclado a VWAP de sesión (la versión rigurosa de
   nuestro scalp en vivo).
3. **sweep_fade_pasivo** — fade pasivo del overshoot en barras de sweep/ignición (proveer
   liquidez al pánico).
4. **rebate_cola** — captura de rebate+spread en libros profundos; cuantifica el muro de
   selección adversa y la cola.
5. **microrev_1m** — micro-reversión a 1 min a ancla corta, maker: ¿queda residuo no-rebote?
6. **pares_gemelos_1m** — reversión de pares gemelos a 1-5 min, maker, dos patas.
7. **xsec_negauto_1m** — cartera contrarian lag-1 de retornos de minuto (autocorrelación neg.).
8. **ofi_taker_tight** — continuación de OFI, microscalp TAKER agresivo en spread estrecho
   (el test taker honesto; +3 bps/lado).

El común denominador de las 7 maker: su viabilidad depende EXACTAMENTE de si el residuo
predecible **no-rebote** supera la selección adversa. La evidencia en vivo y la previa dicen
que no.

## (c) Teoría + investigación previa (convergen)
- **Nuestra investigación previa ya lo demostró**: el "edge" del stat-arb/pares a 5 min
  **desaparece con el test entry_lag** (de +1,14 a −0,31 bps al entrar 1 barra después) →
  era **rebote bid-ask**, no alfa. El ensemble meta-modelo **concentra** ese rebote, no crea
  alfa (de +0,36 a −0,02 bps con entry_lag). Ver `docs/statarb_fondo_resultado.md` y
  `docs/ensemble_metamodel_resultado.md`.
- **Descomposición honesta del scalping maker:** PnL/op ≈ (medio-spread cobrado + rebate) −
  (selección adversa) − (coste de los que cruzan como taker en el stop). Para retail la
  selección adversa ≥ medio-spread+rebate, porque sin prioridad de cola **solo te llenan las
  órdenes que el flujo informado quiere cruzar** (las que van a perder). Resultado ≤ 0.
- **Por qué los HFT sí ganan:** colocation (latencia μs), prioridad de cola (su límite se
  llena ANTES, también en las que van a su favor), y rebates reales de exchange. Es
  infraestructura, no una señal que podamos copiar.

## Tabla de síntesis (para docs/revalidacion_massive_sintesis.md)
| Familia | Resultado | ¿Edge neto? | Muere por |
|---|---|---|---|
| **Scalping maker (muchas ops, beneficio pequeño)** | vivo −0,39%/op, 21% acierto, fill-rate ~90%; 8 variantes maker enumeradas; teoría + previa convergen | ❌ **pierde** | **selección adversa ≥ rebate+spread; el residuo no-rebote ≈0 (entry_lag)** |

## Backtests completados (survivorship-free, DSR/PBO) — los 3 NET-NEGATIVOS
Antes de que el límite de uso de la sesión cortara la barredura se completaron 3 backtests
rigurosos que cubren el espectro de ejecución (mid-cap ilíquida, VWAP-banda = nuestro scalp,
megacap profunda); los TRES rechazan net_positivo con números duros:

| Variante | Bruto/op | **Neto maker/op** | DSR | entry_lag | Veredicto |
|---|---|---|---|---|---|
| **vwap_banda_maker** (≈ scalp en vivo) | −0,98 bp | **−2,34 bp** (IC95<0) | 0,000 | lag1 ≈ +0,10 bp = CERO; residuo no-rebote ≈0 | RECHAZADA |
| **mm_spread_ancho** (spread ancho + OFI) | −7,3 bp | **−8,8 bp** | ~0 | empeora (−10,4 bp) | RECHAZADA |
| **rebate_cola** (megacaps profundos) | +11,4 bp (espejismo) | **−19,8 bp** (IC95 [−23,3,−16,5]) | 0,000 | empeora (−32,4 bp) | NET-NEGATIVO |

Patrón idéntico en los tres: el "medio-spread + rebate" que el maker cree cobrar NO se
materializa, porque **solo te llenan cuando el precio cruza tu quote y SIGUE en tu contra**
(forward condicional al fill: −4,6 a −5,8 bp a +5 min según variante). La selección adversa
≥ rebate+medio-spread en TODO el espectro, de mid-caps ilíquidas a megacaps profundas. DSR≈0 y
PBO sin rescate: ninguna configuración de rejilla generaliza OOS. El `rebate_cola` cuantifica
el muro de infraestructura de market-maker: +11,4 bp "por construcción" → −19,8 bp real = ~31
bps destruidos por la selección adversa.

## Nota de método (honestidad)
Los 5 backtests restantes (microrev_1m, pares_gemelos_1m, xsec_negauto_1m, ofi_taker_tight,
sweep_fade_pasivo) y la síntesis automática **no llegaron a correr**: la sesión alcanzó su
límite de uso (reseteo 18:00 UTC). Ya reseteado, el workflow se ha **relanzado** para completar
la barredura y este doc se actualizará con el cuadro completo. Los 3 backtests ya completados
+ la medición EN VIVO (−0,39%/op) + la investigación previa convergen todos: no hay scalping
neto positivo accesible para retail.

## Recomendación
No cablear scalping. El único superviviente de todo el estudio sigue siendo el **overnight de
QQQ** (en paper, midiendo su slippage MOC/MOO). El valor de este ejercicio es negativo y vale:
saber que "muchas operaciones, beneficio pequeño" **pierde a coste** evita quemar capital real
en un patrón que solo es rentable con infraestructura de market-maker.
