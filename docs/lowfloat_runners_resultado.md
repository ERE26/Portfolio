# Small-cap *runners* (low-float): ¿hay edge intradía real para una cuenta pequeña?

Fecha: 2026-10-05 · Alcance: **SOLO research / paper**, cero gasto en órdenes.
Datos: Massive (plan Starter) — grouped diario de TODO el mercado US (incluidos
deslistados) + barras de 1 minuto. Código: `universo_runners_pit.py`,
`backtest_lowfloat_massive.py`, infra en `fuentes/massive.py`.

## TL;DR — veredicto honesto

**No hay edge robusto.** Probadas las tres setups reales del day-trader de runners
(gap-and-go, ORB del runner, short parabólico) sobre el universo **point-in-time y
sin supervivencia** de small-caps en juego, con ejecución realista:

| Estrategia | Expectancy @25 bps (pooled, 2 ventanas) | ¿IC95 sobre 0? | Muere a… |
|---|---|---|---|
| **Gap-and-go** (largo ruptura) | −0.07R / −0.12R | **No** (cruza 0) | 50 bps de slippage |
| **ORB runner** (ruptura confirmada) | −0.19R / +0.05R | **No** (cruza 0) | 50–100 bps |
| **Short parabólico** | −0.77R / −0.32R | **No — PIERDE** | ya pierde a 10–25 bps |

- Ninguna estrategia tiene un intervalo de confianza del 95% que despegue de cero
  en ninguna de las dos ventanas independientes (Q2-2024 y Q4-2023).
- Los dos *largos* están en **break-even ligeramente negativo** con slippage
  optimista (25 bps) y pasan a **claramente negativos** con slippage realista para
  estos nombres (50–100 bps). **Se los come la ejecución.**
- El *short* **pierde siempre** (te arrolla el runner: ~80% de stops), y encima en
  low-floats el borrow es caro o directamente *hard-to-borrow* (a veces ni se puede).
- La "señal" que aparece en muestras minúsculas (una semana da +0.65R) es un
  **espejismo de cola**: con 130–175 operaciones colapsa a cero. La media la cargan
  3–4 ganadores monstruosos; la **mediana de R es negativa** en las tres estrategias.

Esto **cierra** la hipótesis "el retail de bots gana en el rincón que el fondo no
toca": el rincón existe y es ineficiente, pero la ineficiencia **no es capturable**
con estas reglas una vez pagas el spread.

---

## 1. Qué se probó y por qué

Toda nuestra búsqueda intradía anterior usó el **universo líquido** (mid/large-cap),
el universo del fondo, el más arbitrado — y no apareció edge (ver
`docs/intradia_busqueda_edge.md`, `docs/statarb_fondo_resultado.md`). Pero el retail que monta
bots **no opera ahí**: opera en **small-cap / low-float runners** ($1–20, poca
flotación, catalizador/noticia, que se mueven 30–200% intradía). Es el rincón que
los fondos no pueden tocar (capacidad minúscula) pero una cuenta pequeña sí. **Nunca
lo habíamos probado.** Esto responde a esa pregunta, con honestidad: sin
supervivencia y con slippage alto y realista.

## 2. Universo PIT de runners, SIN supervivencia

Construido día a día desde el **grouped diario** de Massive (todo el mercado US en
una llamada; incluye los que cotizaron ese día **aunque hoy estén deslistados**).
Selección **as-of, sin look-ahead**: que un ticker sea "runner candidato" el día *t*
se decide con el **gap de apertura** (open *t* vs close *t−1*) y el **precio de
apertura**, ambos conocidos a las 09:30 — nunca con el resultado del día. Filtro:
precio $1–20, gap ≥ 10% (capado a 300% y excluyendo splits para no contar
reverse-splits como "gaps"), top-12 gappers del día.

| Ventana | Sesiones | Candidatos | Gap mediano | Precio mediano | **Haircut de supervivencia** |
|---|---|---|---|---|---|
| Q2-2024 (mar–may) | 63 | 753 | 21.6% | $3.19 | **50.1%** (377 de 753 ya no cotizan) |
| Q4-2023 (sep–nov) | 62 | 727 | 17.0% | $2.38 | **58.5%** (425 de 727 ya no cotizan) |

> **El haircut es enorme: entre la mitad y el 60% de los runners de hace ~2 años ya
> están deslistados o inactivos.** Un backtest que muestreara el universo de HOY
> (como hacían todos los nuestros anteriores) no vería ni la mitad de estos nombres.
> Esto, por sí solo, es la lección más importante del ejercicio: en small-caps el
> sesgo de supervivencia no es un detalle, es la mayor parte del universo.

## 3. Metodología (rigor)

- **Point-in-time, sin look-ahead.** Selección por gap+precio de apertura. Gates
  *in-play* (que confirman "stock en juego") con solo las primeras velas: **RVOL de
  apertura** (volumen del rango 09:30–09:35 vs una vela de 5m normal de ayer) ≥ 2.0 y
  **$-volumen del rango de apertura** ≥ $50k. La entrada se rellena al **precio de
  disparo** (ruptura) con slippage; el $-volumen de la sesión completa **no** se usa
  para seleccionar (sería look-ahead), solo para describir liquidez.
- **Sin supervivencia.** El universo incluye deslistados; se reporta el resultado en
  el universo **COMPLETO** y en **solo-supervivientes**, y el haircut entre ambos.
- **Suelo de riesgo 1R ≥ 2% del precio.** Sin esto, un rango de apertura de 1 centavo
  da R-fantasía (100R imposibles de llenar; el slippage se come un stop así). Es el
  detalle que separa el espejismo del resultado real.
- **Ejecución.** Barrido de slippage **10 / 25 / 50 / 100 bps por lado**
  (SLIP_STOP = 2×, el stop llena peor). El short paga **borrow/locate** (20 bps base).
- **3 semillas** (muestrean días), **bootstrap IC95** sobre la expectancy pooled.
  Una entrada por ticker-día (mide calidad de entrada, no re-entradas). Intrabar se
  asume el stop antes que el objetivo (conservador).
- Operaciones válidas (pasan gate) por ventana: 502 (Q2-2024), 346 (Q4-2023).

---

## 4. Resultados por estrategia

Todas las tablas: universo **PIT (completo, sin supervivencia)**, slippage base 25
bps salvo el barrido. R = múltiplos de riesgo (1R = distancia al stop).

### 4.1 Gap-and-go (largo a la ruptura del rango de apertura)

Gap up grande + RVOL de apertura alto + 1ª vela verde → largo al romper el máximo del
rango; stop en el mínimo del rango (suelo 2%); salida por **fade de momentum** (cierre
bajo VWAP de sesión) o EOD.

Por semilla y pooled (25 bps, PIT):

| Ventana | seed | ops | acierto | expectancy | IC95 |
|---|---|---|---|---|---|
| Q2-2024 | 0 | 56 | 28.6% | −0.130R | [−0.267, +0.019] |
| Q2-2024 | 1 | 55 | 16.4% | −0.306R | [−0.402, −0.209] |
| Q2-2024 | 2 | 56 | 21.4% | −0.209R | [−0.375, −0.012] |
| **Q2-2024** | **pooled** | **175** | **27.4%** | **−0.073R** | **[−0.204, +0.076]** |
| **Q4-2023** | **pooled** | **135** | **23.7%** | **−0.115R** | **[−0.244, +0.038]** |

Barrido de slippage (pooled PIT):

| Slippage | Q2-2024 expectancy (IC95) | Q4-2023 expectancy (IC95) |
|---|---|---|
| 10 bps | −0.04R [−0.17, +0.11] | −0.07R [−0.20, +0.09] |
| 25 bps | −0.07R [−0.20, +0.08] | −0.12R [−0.24, +0.04] |
| 50 bps | −0.12R [−0.25, +0.03] | **−0.19R [−0.31, −0.04] PIERDE** |
| 100 bps | **−0.22R [−0.35, −0.08] PIERDE** | **−0.32R [−0.45, −0.18] PIERDE** |

- Cierres (Q2): VWAP_FADE 152, BE 11, SL 8, EOD 4. Casi todo sale por el fade del
  VWAP: el runner pierde momentum y cruza el VWAP a la baja antes de dar mucho.
- **Perfil de cola:** acierto ~27%, R medio de las ganadoras ~0.9, **mediana de R
  −0.29**. La mayoría son pequeñas perdedoras; algún ganador modesto. No hay la
  asimetría enorme que la leyenda del gap-and-go promete (con el stop realista de 2%).
- **Veredicto: sin edge.** Marginalmente negativo a 25 bps; **muere a 50 bps**.

### 4.2 ORB runner (ruptura CONFIRMADA del rango de apertura)

Una vela **cierra** sobre el máximo del rango → largo; stop en el mínimo (suelo 2%);
se deja correr hasta EOD o stop (estilo Zarattini "ORB for stocks in play").

| Ventana | pooled ops | acierto | expectancy @25bps | IC95 | mediana R |
|---|---|---|---|---|---|
| Q2-2024 | 150 | 30.0% | −0.19R | [−0.42, +0.10] | −1.01 |
| Q4-2023 | 124 | 24.2% | +0.05R | [−0.37, +0.65] | −1.01 |

Barrido de slippage (pooled PIT):

| Slippage | Q2-2024 | Q4-2023 |
|---|---|---|
| 10 bps | −0.16R [−0.39, +0.14] | +0.10R [−0.33, +0.70] |
| 25 bps | −0.19R [−0.42, +0.10] | +0.05R [−0.37, +0.65] |
| 50 bps | −0.25R [−0.48, +0.04] | −0.03R [−0.44, +0.56] |
| 100 bps | **−0.35R [−0.57, −0.08] PIERDE** | −0.17R [−0.57, +0.38] |

- Cierres (Q2): EOD 73, SL 77 — **mitad y mitad**. Dejar correr al runner: la
  mayoría de las veces te saca el stop (mediana R = −1.0), y de vez en cuando corre
  todo el día (R medio ganadora 1.4–1.7).
- **Enorme varianza e inconsistencia entre ventanas** (Q2 −0.19R, Q4 +0.05R): el
  IC95 **siempre cruza cero**. Es ruido, no edge.
- **Veredicto: sin edge claro.** Cero neto con slippage optimista, negativo con
  slippage realista.

### 4.3 Short parabólico (fade del runner sobre-extendido)

Varias velas verdes consecutivas + extensión ≥10% del máximo sobre VWAP + vela de
agotamiento → corto; stop sobre el máximo del tramo (suelo 2%); cubre al volver a
VWAP o EOD. Paga **borrow/locate** (20 bps).

| Ventana | pooled ops | acierto | expectancy @25bps | IC95 | mediana R |
|---|---|---|---|---|---|
| Q2-2024 | 136 | 15.4% | **−0.77R** | [−0.94, −0.57] | −1.15 |
| Q4-2023 | 84 | 27.4% | **−0.32R** | [−0.61, +0.00] | −1.09 |

Barrido de slippage (pooled PIT):

| Slippage | Q2-2024 | Q4-2023 |
|---|---|---|
| 10 bps | **−0.69R PIERDE** | −0.24R [−0.53, +0.09] |
| 25 bps | **−0.77R PIERDE** | −0.32R [−0.61, +0.00] |
| 50 bps | **−0.88R PIERDE** | **−0.45R PIERDE** |
| 100 bps | **−1.15R PIERDE** | **−0.71R PIERDE** |

- Cierres (Q2): SL 113, VWAP_COVER 22, EOD 1 → **te arrolla el runner el ~83% de las
  veces.** Shortear fuerza en un low-float es ponerte delante del tren.
- **Veredicto: PIERDE claramente**, y es la peor de las tres. Además: muchos
  low-floats son *hard-to-borrow* o no hay localización — en la práctica esta
  estrategia **a menudo ni es ejecutable**, y el borrow modelado (20 bps) es
  optimista para estos nombres.

---

## 5. El muro de ejecución (¿a qué slippage muere cada una?)

| Estrategia | Marginal viable | Muere (expectancy claramente < 0) |
|---|---|---|
| Gap-and-go | ~10 bps (y aun así IC95 cruza 0) | **50 bps** |
| ORB runner | ~10–25 bps (IC95 cruza 0 siempre) | **50–100 bps** |
| Short parabólico | nunca | **ya a 10–25 bps** |

El precio **mediano** de estos runners es **$2.4–3.2**. En nombres así, el spread
real suele ser de **1–5 centavos**, es decir **30–150 bps**, no 25. Para **tamaño
pequeño el fill es plausible** (el gate exige ≥$50k en los primeros 5 min, así que
unos cientos de acciones entran), **pero ese mismo spread ancho es el que mata la
expectancy.** El problema no es la capacidad; es que la ineficiencia que mueve el
precio **es, en buena parte, el propio spread** — y te lo comes tú al entrar y salir.

## 6. Supervivencia: un matiz honesto

El haircut es del 50–58%. Pero su efecto **no** es el típico "el universo de hoy
infla el backtest". Para un **largo intradía en el día del gap**, los nombres que
luego mueren son justo los que más *pumpean* ese día:

| Estrategia | Expectancy PIT (completo) | Expectancy solo-supervivientes |
|---|---|---|
| Gap-and-go (Q2) | −0.073R (175 ops) | −0.144R (95 ops) |
| ORB runner (Q2) | −0.190R (150 ops) | −0.007R (80 ops) |
| Short parabólico (Q2) | −0.765R (136 ops) | −0.814R (80 ops) |

Es decir: quitar los muertos **empeora** el gap-and-go (los muertos aportaban las
subidas intradía más jugosas). **Ni así** ninguna versión es positiva. Pero el
haircut importa igual por **preservación de capital**: la mitad de estos nombres
acaba en cero, con *halts* y gaps a la baja — el día que el pump te falla estás
atrapado en un −60% halteado, y eso un backtest intradía de una sola entrada no lo
castiga del todo.

## 7. Veredicto final

> **Para una cuenta pequeña, en el universo de small-cap runners, con honestidad
> (sin supervivencia + slippage realista): NO hay edge robusto en ninguna de las tres
> setups.** Los largos (gap-and-go, ORB) están en break-even ligeramente negativo con
> slippage optimista y **mueren a 50–100 bps**, que es el rango realista de spread
> para nombres de $2–3. El short parabólico **pierde siempre** y encima suele ser
> inejecutable (hard-to-borrow). Lo poco que brilla en muestras pequeñas es **cola
> afortunada**, no señal: con n≥130 y dos ventanas independientes, la expectancy
> colapsa a cero/negativo y **ningún IC95 despega de cero**.

**¿Qué haría falta para que funcionara?** Un slippage efectivo de **≤10 bps por
lado** (irreal en $2–3; solo alcanzable con tamaño mínimo, fills pasivos y los
nombres más líquidos del grupo, p.ej. >$5 y >$20M de $-volumen) **y** además una
selección más fina que la de "top gappers" (catalizador concreto, ventana horaria,
float real vía shares outstanding) que **suba el acierto o la cola** de forma
**estable entre ventanas**. Nada de lo probado aquí lo consigue, y la base de
partida (50% de mortalidad, spreads anchos) es desfavorable. La carga de la prueba
está en encontrar esa variante; el resultado honesto de hoy es **negativo**.

## 8. Limitaciones y reproducibilidad

- Muestras moderadas (130–175 ops/estrategia/ventana), una entrada por ticker-día,
  parámetros **fijos y sin optimizar** (RVOL≥2, $OR≥50k, top-12 gappers, stop 2%).
  No se ha hecho búsqueda de hiperparámetros: evita overfitting, pero no descarta que
  una variante muy distinta tenga señal. El veredicto es sobre **estas** reglas.
- Barras de 1m de Massive (incluyen premarket/afterhours; se filtra a 09:30–16:00 ET).
  Los low-floats no cotizan cada minuto: se trabaja sobre las velas que existen.
- VWAP de sesión reconstruido del `vw` por barra (fallback precio típico).
- Dos ventanas (Q2-2024, Q4-2023), no solapadas, como chequeo out-of-sample mutuo.

Reproducir:

```
python universo_runners_pit.py --desde 2024-03-01 --hasta 2024-05-31      # inspección del universo + haircut
python backtest_lowfloat_massive.py --desde 2024-03-01 --hasta 2024-05-31 --topk 12 --seeds 3 --dias-seed 20
python backtest_lowfloat_massive.py --desde 2023-09-01 --hasta 2023-11-30 --topk 12 --seeds 3 --dias-seed 20
```

Solo lectura / paper. No toca `gestionar.py` ni `operativa_*.py`. No cablea nada.
