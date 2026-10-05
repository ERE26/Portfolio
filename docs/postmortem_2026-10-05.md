# Post-mortem 2026-10-05 — qué funcionó, qué no, y qué cambiar mañana

**Día verde: +$1.276 (+1,27%).** Pero la atribución es lo que importa, y es honesta e incómoda.

## La verdad del día en una frase
**El día entero lo hizo QDEL.** QDEL (corredora ORB) realizó **+$2.429** (comprada 2270 @ 10,97,
vendida 2270 @ 12,04 al tocar su take-profit). Daily change total **+$1.276** ⇒ **todo lo demás
del libro, neto, PERDIÓ ~−$1.150 hoy.** Sin esa única posición, hoy habría sido un día ROJO.

## Números por libro (hoy)
| Libro | Resultado hoy | Lectura |
|---|---|---|
| **QDEL** (ORB, 1 corredora) | **+$2.429** realizado | el TP 2R + stop protector que puse SÍ capturó el fat-tail |
| **ORB resto** (PAGS, THRM, XP, PAX) | **−$251** (1/4 acierto) | PAGS −232, THRM −86; rupturas extendidas que se giraron; slippage entrada **+0,49%** |
| **SCALP maker** (36 cerradas) | **−0,25%/op, 31% acierto** (95% fill) | selección adversa confirmada EN VIVO = los 8/8 backtests |
| **FADE** (11 intentos) | 5 enviadas, **6 rechazadas HTB** (SAIQ, VEEA, APUS, JAGX, MI, ALVO) | no puedes shortear los gappers extremos: el short-parabólico es inejecutable retail |
| **PAIRS** (3 pares) | **+$6,52** | reversión real pero magnitud sub-coste |
| **Swing/TSMOM** (33 pos abiertas) | +$707 no realizado | beta de mercado (día alcista), no alfa intradía |

## Qué FUNCIONÓ (decisiones que suman, mantener)
1. **El fix de TP/SL realista + el fix de colisión de libros fueron los que capturaron QDEL.**
   Antes: TP fantasma +20% que casi nunca toca, y el gestor de momentum cancelaba el OCO del ORB.
   Hoy: QDEL tenía un TP 2R realista (12,03) y un OCO protegido que NADIE pisó → corrió a +9,7% y
   salió sola a 12,04. **Esa es la única expectativa positiva del ORB: la cola derecha gorda, y
   solo la capturas si el objetivo es alcanzable y el stop no lo borra otro libro.**
2. **El tope del 12% por nombre y la protección a breakeven** contuvieron el daño del resto.
3. **cerrar_intradia arreglado** (solo momentum) evitó que el cierre cancelara el overnight y
   aplanara TSMOM.

## Qué NO funcionó (y la lección para mañana)
1. **ORB resto: 3/4 perdedoras, −$251.** PRIMERA hipótesis (plausible): "entramos rupturas ya
   extendidas". **REFUTADA por los datos**: la extensión de la entrada sobre el máximo del rango
   fue PAGS +0,2% (perdió −232, la peor), PAX +0,4% (ganó +72), THRM +1,6% (−86), XP +1,9% (−6).
   **No hay relación extensión↔resultado**; el mayor perdedor fue la ruptura MENOS extendida.
   **Lección real:** NO añadir un "filtro anti-chase" — sería overfitting a 4 trades. Las
   perdedoras del ORB son el BASE RATE de una estrategia sin edge (las rupturas fallan), no un
   patrón corregible con una regla ajustada a un día. La disciplina es NO sobre-ajustar al ruido.
2. **SCALP: −0,25%/op en 36 ops.** Confirma el veredicto 8/8. **Lección:** PAUSAR el scalp, cero
   size extra; ya no aporta información nueva.
3. **FADE: 6/11 rechazadas por HTB.** **Lección:** pre-filtrar shortabilidad y no malgastar
   intentos en gappers no-shortables.
4. **Overnight MOC EXPIRÓ sin llenar** (Alpaca paper no ejecuta 'cls') → no medimos el único
   superviviente. **Lección/pendiente:** cambiar a orden de mercado pegada al cierre/apertura
   (pendiente OK de Eduardo).

## Decisiones concretas para mañana (perfeccionar el intradía = contener y seleccionar)
- **ORB:** NO añadir filtros ajustados a 4 trades (el anti-chase quedó refutado por los datos).
  Mantener lo estructural que SÍ funcionó: TP 2R, tope 12%, breakeven, OCO protegido. Dejar correr
  la corredora cuando aparezca (es donde vive la única expectativa positiva). Cualquier cambio de
  entrada se valida con backtest survivorship-free, nunca con el P&L de un día.
- **SCALP:** pausar (veredicto cerrado).
- **FADE:** saltar nombres HTB conocidos de entrada.
- **Overnight QQQ:** arreglar la ejecución (market cerca del cierre) — es el EV real.
- **Marco honesto:** un día QDEL no es edge intradía. El intradía sigue siendo MEDICIÓN; el
  retorno esperado de verdad está en el overnight + los diversificadores swing/TSMOM. No confundir
  suerte de cola + beta con habilidad intradía, y no arriesgar capital real en el intradía.

## Lo más valioso que aprendimos hoy
La estructura importa más que la señal: **un objetivo realista + un stop que nadie pisa +
dejar correr al ganador** convirtió la ÚNICA corredora buena del día en +$2.429 — y eso, no una
señal intradía mágica, es lo que hizo el día verde. Perfeccionar el intradía = perfeccionar la
GESTIÓN (selección, contención, protección del ganador), porque el edge de señal no existe.
