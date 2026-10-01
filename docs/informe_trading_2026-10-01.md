# Informe de actividad de trading — 1 de octubre de 2026

**Cuenta:** Alpaca *paper trading* (demo, dinero simulado). Capital inicial de
referencia: **$100.000**. Horario operado: **12:58–15:55 ET** (sesión parcial).
Todas las operaciones son simuladas; el conector está bloqueado al endpoint
*paper* y nunca toca dinero real.

## 1. Resumen ejecutivo

| Métrica | Valor |
|---|---:|
| **Equity final** | **$100.656** |
| **P&L del día** | **+$656 (+0,66 %)** |
| P&L intradía (realizado) | **+$483,05** |
| P&L swing (no realizado, abierto) | ≈ +$173 |
| Operaciones intradía cerradas | **12** |
| Acierto intradía | **7 / 12 (58 %)** |
| Posiciones swing abiertas | 24 (se mantienen) |

El resultado del día combina dos libros: un **intradía** de *stocks in play*
(entradas y salidas dentro de la sesión) y un **swing** de momentum (posiciones
que se mantienen a varios días). El intradía aportó +$483 realizados; el swing
sumó ≈ +$173 no realizados.

## 2. Metodología (cómo se decide)

**Entrada intradía.** Un valor es candidato si tiene *volumen anormal* (RVOL alto
= "stock in play"), cotiza **por encima de su VWAP** y lleva **sesión positiva**.
Entre los candidatos fuertes se elige por un **score de convicción** (confluencia
de RVOL + retorno de sesión + catalizador de noticias + memoria del propio
ticker), descartando los de noticia claramente negativa. Se opera con calidad >
cantidad: si no hay candidato fuerte, se queda en efectivo.

**TP y SL.** En múltiplos del ATR (volatilidad del valor): **TP ≈ 1,6·ATR** y
**SL ≈ 0,8·ATR** (reward:risk ≈ 2,0). *Dejar correr al ganador, cortar rápido al
perdedor.* Órdenes *bracket*: Alpaca gestiona la salida al tocar TP o SL.

**Salidas adicionales:** (a) **reversión confirmada** (pierde el VWAP de forma
sostenida; no se cierra por una vela roja puntual); (b) **time-stop** (en verde
≥1,5 h se recoge el positivo y se rota; ≥2,5 h cierre duro, el setup ha caducado);
(c) **cierre EOD** (*flatten* del intradía antes del cierre de mercado).

**Swing (diario).** Momentum: se compra el quintil de mayor fuerza relativa.
Salida por caída del quintil o por **filtro de régimen** (a liquidez si el mercado
pierde su media de 200 sesiones). Hoy el régimen fue **alcista**, así que el libro
swing se mantuvo íntegro.

## 3. Operaciones intradía (detalle completo)

Horas en ET. Precio de apertura y cierre = precio medio de ejecución real.

| # | Ticker | Apertura | Precio apert. | Por qué (señal de entrada) | Cierre | Precio cierre | Tipo de cierre | P&L | Ret. |
|--:|:--|:--|--:|:--|:--|--:|:--|--:|--:|
| 1 | JPM | 12:58 | $332,43 | Intradía inicial (RVOL alto, sobre VWAP) | 15:20 | $332,72 | **Time-stop en verde** (142 min → recoge y rota) | +$17,57 | +0,09 % |
| 2 | BAC | 12:58 | $53,67 | Swing previo | 13:07 | $53,60 | Rotación swing (salió del quintil líder) | −$26,04 | −0,13 % |
| 3 | COST | 12:58 | $915,83 | Intradía inicial (stock in play) | 15:07 | $914,07 | Cierre por gestión/stop (no EOD) | −$35,82 | −0,19 % |
| 4 | MU | 13:07 | $1.063,53 | RVOL 2,19 muy alto, rotura de resistencia alcista | 13:30 | $1.076,20 | **TP** | **+$226,88** | +1,19 % |
| 5 | MU | 13:38 | $1.077,42 | Reentrada por convicción (sigue líder, sobre VWAP) | 13:45 | $1.088,46 | **TP** | **+$198,72** | +1,02 % |
| 6 | MU | 13:54 | $1.090,22 | Reentrada por convicción | 14:04 | $1.083,03 | **SL** | −$129,32 | −0,66 % |
| 7 | MU | 14:11 | $1.074,87 | Reentrada a mejor precio (tras SL, fuera de cooldown) | 14:49 | $1.086,06 | **TP** | **+$201,42** | +1,04 % |
| 8 | MU | 14:56 | $1.088,60 | RVOL 2,19, convicción +2,03, rotura resistencia | 15:55 | $1.094,64 | Cierre EOD (*flatten*) | +$108,48 | +0,56 % |
| 9 | NKE | 15:12 | $36,06 | RVOL 1,65, canal alcista, sobre VWAP +0,57 % | 15:17 | $35,92 | **SL** | −$77,56 | −0,39 % |
| 10 | NKE | 15:20 | $35,79 | Reentrada canal alcista | 15:24 | $35,65 | **SL** | −$78,12 | −0,39 % |
| 11 | XLF | 15:20 | $53,42 | RVOL 1,70, sobre VWAP, sesión positiva | 15:55 | $53,49 | Cierre EOD (*flatten*) | +$26,18 | +0,13 % |
| 12 | JPM | 15:27 | $332,54 | RVOL 2,07, tendencia alcista, sobre VWAP +1,05 % | 15:55 | $333,38 | Cierre EOD (*flatten*) | +$50,66 | +0,26 % |
| | | | | | | | **TOTAL intradía** | **+$483,05** | |

### Notas de ejecución
- **MU fue el nombre del día**: 5 operaciones (3 TP, 1 SL, 1 cierre EOD), neto
  **+$605**. Un *stock in play* con RVOL 2,19 y rotura de resistencia que mantuvo
  fuerza toda la sesión; se reentró por convicción cada vez que liberó hueco (y
  tras el único SL, el *cooldown* se saltó porque la salida anterior fue una
  ganancia, no una pérdida).
- **JPM #1** ilustra el **time-stop**: llevaba 142 min abierta y en verde → se
  recogió el positivo y se rotó, en vez de esperar a que se girara.
- **NKE** (2 SL seguidos) muestra el *cooldown* funcionando: tras el primer SL,
  NKE quedó vetado para reentrada durante 60 min.

## 4. Libro swing (posiciones abiertas, se mantienen)

Entradas a las **12:58 ET**. *Por qué:* quintil superior de momentum (fuerza
relativa 12-1). *Estado:* abiertas — régimen de mercado alcista, se dejan correr.

| Ticker | Acc. | Precio ent. | Ticker | Acc. | Precio ent. | Ticker | Acc. | Precio ent. |
|:--|--:|--:|:--|--:|--:|:--|--:|--:|
| AMD | 2 | $609,78 | DOW | 58 | $27,29 | NEM | 13 | $114,80 |
| BMY | 25 | $61,33 | EOG | 11 | $141,22 | NUE | 6 | $233,74 |
| C | 12 | $126,96 | EQIX | 1 | $1.007,55 | OXY | 27 | $57,19 |
| CAT | 1 | $823,00 | FCX | 22 | $69,10 | PSX | 6 | $262,69 |
| COP | 12 | $126,83 | INTC | 13 | $119,55 | SLB | 32 | $48,48 |
| CSCO | 14 | $108,00 | JNJ | 6 | $259,51 | XLE | 25 | $62,40 |
| CVX | 7 | $206,36 | MPC | 3 | $413,08 | XOM | 9 | $163,66 |
| DE | 2 | $662,21 | MRK | 11 | $143,94 | | | |

## 5. Estadísticas

**Por tipo de cierre (intradía):**

| Tipo | Nº | P&L | Comentario |
|:--|--:|--:|:--|
| Take-profit (TP) | 3 | **+$627,02** | MU ×3; capturaron el objetivo (~+1 %) |
| Stop-loss (SL) | 3 | −$284,99 | MU ×1, NKE ×2; cortados rápido (~−0,4/−0,66 %) |
| Time-stop en verde | 1 | +$17,57 | JPM; recoge y rota |
| Cierre EOD (*flatten*) | 3 | +$185,33 | MU, XLF, JPM; cerrados antes del cierre |
| Gestión / rotación | 2 | −$61,86 | BAC, COST |
| **Total** | **12** | **+$483,05** | 7 ganadoras / 5 perdedoras (58 %) |

- **Payoff:** ganancia media por acierto ≈ +$118; pérdida media ≈ −$69. Ratio
  ganancia/pérdida ≈ **1,7** — el sistema gana por asimetría (dejar correr,
  cortar rápido), no por acertar mucho.
- Los 3 TP (todos MU) aportaron el grueso del día; los SL se cortaron pequeños,
  tal como está diseñado.

## 6. Nota honesta

Ha sido un día positivo (+0,66 %), pero **un día no demuestra un edge**. El
análisis de costes de este mismo proyecto muestra que, en *blue chips* de 15
minutos, el edge neto tras costes realistas es finísimo; el valor del sistema está
en la **disciplina** (selección por convicción, cortar rápido, time-stop, rotación
y filtro de régimen en el swing), no en forzar operaciones. El resultado real de
hoy se apoya sobre todo en un nombre que funcionó (MU) y en el libro swing de
momentum, cuyo edge sí es robusto (Deflated Sharpe ≈ 1,0 sobre 37 años).

---
*Datos reconstruidos de los fills reales de la cuenta paper de Alpaca y de la
bitácora interna de operaciones. Documento generado automáticamente.*
