# La anomalía overnight, en limpio y sin sesgo de supervivencia (Massive EOD)

**Fecha:** 2026-10-05 · **Alcance:** SOLO research/paper, solo lectura. No cablea
nada en vivo, no toca `gestionar.py` ni `operativa_*.py`.
**Código:** `backtest_overnight_massive.py`, que reutiliza la infraestructura
ya commiteada `fuentes/massive.py::grouped_diario` (barra diaria de todas las
acciones en 1 llamada) e implementa su propio universo point-in-time
panel-vectorizado (misma idea y misma fuente que `universo_pit.universo_asof`,
con el que es consistente; verificación cruzada opcional en `--xcheck`).
**Datos:** Massive (plan Starter), EOD ajustado por splits, endpoint *grouped
daily* (1 petición/sesión = todas las acciones), caché local gitignored.

---

## TL;DR — veredicto honesto

> La anomalía existe en los datos y **se confirma en dirección**: el tramo
> **overnight** (cierre→apertura) aporta retorno positivo y el **intradía**
> (apertura→cierre) es un lastre. Pero como estrategia **long-only equal-weight
> con rebalanceo diario NO es explotable neta de costes**: el bruto (~+6,6 %/año,
> Sharpe ≈ 0,58) se lo come entero el **doble cruce diario**. El punto de
> equilibrio está en **~1,4 pb por lado**; con una ejecución realista en acciones
> líquidas (≥2 pb/lado entre medio-spread, comisiones y slippage de subasta) el
> neto **se vuelve negativo**. Y, lo que es peor para la tesis: **incluso en
> bruto rinde por debajo de comprar y mantener SPY** (+12,2 %/año con una sola
> operación), cuyo propio retorno overnight capturas con un único trade.
>
> Matices de honestidad: (1) el overnight bruto **ni siquiera es
> estadísticamente distinguible de cero** en esta ventana (IC95 del retorno medio
> diario = **[−1,4; +7,1] pb**, cruza el cero); (2) la anomalía **falló en el oso
> de 2022** (overnight también negativo); (3) el **haircut de supervivencia es
> casi nulo aquí** (~0 pb/día) — el asesino no es el sesgo de supervivencia, son
> las comisiones del cruce diario, exactamente como anticipaba la hipótesis.

---

## 1. Método (point-in-time, sin look-ahead)

- **Universo survivorship-free por construcción.** Con el *grouped daily* de
  Massive, una llamada por sesión devuelve la barra de **todas** las acciones que
  cotizaron ese día. Un ticker aparece en la fecha *t* si y solo si cotizó en *t*;
  cuando se deslista, deja de aparecer. El panel reconstruido contiene a los
  **deslistados en las fechas en que estuvieron vivos** → no hay sesgo de
  supervivencia. Se filtra a **acción común (type=CS)** vía
  `/v3/reference/tickers` (activos + deslistados) para excluir ETFs, warrants,
  unidades y preferentes.
- **Periodo:** 2021-10-06 → 2026-10-02, **1.253 sesiones** (límite real del plan
  Starter: 5 años exactos; anteriores devuelven HTTP 403). **8.341 tickers CS
  únicos** a lo largo del periodo; ~2.900–3.000 nombres/día tras filtro de
  liquidez.
- **Descomposición del retorno diario** (ajustado por splits):
  - `OVERNIGHT = open_t / close_{t-1} − 1`
  - `INTRADÍA  = close_t / open_t − 1`
  - (control: `close_t/close_{t-1}−1 = (1+ON)(1+ID)−1`)
  Se recortan |r|>50 % por barra como datos sucios.
- **Selección as-of sin look-ahead** (universo panel-vectorizado, misma idea que
  `universo_pit.universo_asof`): un valor entra en la cartera de *t* solo si, **con
  información del cierre de t−1**, precio ≥ 5 $ y dólar-volumen mediano móvil
  (21 d, *desplazado 1*) ≥ umbral. La decisión de operar en *t* nunca usa datos de
  *t*. Verificación cruzada (`--xcheck`) contra el selector ya commiteado
  `universo_pit.universo_asof` (top-500): **solape ~92–93 %** en fechas de muestra
  (2023, 2024, 2025); el resto es por umbrales de liquidez ligeramente distintos.
- **Carteras equal-weight, rebalanceo diario.** El retorno es la media de los
  nombres **ejecutables** ese día.
- **Modelo de coste.** Tanto overnight como intradía **cruzan dos veces al día**
  (entrada + salida), así que el coste diario = `2 × coste_por_lado`. Se reporta
  bruto y neto a **1, 2 y 5 pb/lado** (= 2, 4, 10 pb/día de ida y vuelta). SPY
  buy&hold cruza una sola vez en todo el periodo → coste ≈ 0.
- **Métricas:** retorno medio diario, anualizado (CAGR), Sharpe (rf=0), maxDD,
  % días positivos y **bootstrap IID (2.000 reps) para IC95** del retorno medio y
  del Sharpe.

---

## 2. Resultado principal (periodo completo, 2021-10 → 2026-10)

Retornos en **puntos básicos por día** (pb/d) y **% anualizado (CAGR)**.

| Estrategia (equal-weight, rebal. diario) | Bruto pb/d | Bruto CAGR | Sharpe | maxDD | Neto @1pb/lado | Neto @2pb/lado | Neto @5pb/lado |
|---|---:|---:|---:|---:|---:|---:|---:|
| **OVERNIGHT** (líq. ≥1 M$) | **+2,81** | **+6,6 %** | 0,58 | −22 % | +1,3 % | **−3,7 %** | −17,2 % |
| **INTRADÍA** (líq. ≥1 M$) | **−3,52** | **−10,0 %** | −0,49 | −46 % | −14,4 % | −18,6 % | −30,0 % |
| OVERNIGHT (líq. ≥10 M$) | +2,56 | +5,9 % | 0,53 | −24 % | +0,7 % | −4,3 % | −17,7 % |
| INTRADÍA (líq. ≥10 M$) | −0,86 | −3,7 % | −0,12 | −28 % | −8,4 % | −12,9 % | −25,2 % |
| OVERNIGHT top-500 dólar-vol | +2,88 | +6,8 % | 0,63 | −24 % | +1,6 % | −3,4 % | −17,0 % |
| OVERNIGHT ex-peor-momentum | +2,14 | +4,9 % | 0,47 | −22 % | −0,3 % | −5,2 % | −18,5 % |
| Cartera CC (cierre-cierre, líq.) | −0,94 | −4,5 % | −0,11 | −44 % | — | — | — |
| **SPY buy&hold (cierre-cierre)** | **+5,14** | **+12,2 %** | **0,75** | −25 % | ≈ bruto | ≈ bruto | ≈ bruto |
| SPY solo overnight | +2,69 | +6,4 % | 0,66 | −21 % | — | — | — |
| SPY solo intradía | +2,59 | +5,7 % | 0,46 | −14 % | — | — | — |

**Lectura:**

1. **La anomalía se confirma en dirección.** La cartera equal-weight de acciones
   líquidas gana overnight (+2,81 pb/d) y pierde intradía (−3,52 pb/d); la suma
   (cierre-cierre) es **negativa** (−0,94 pb/d). Es decir, **todo el retorno (y
   más) de esa cesta de acciones estuvo en la noche**, y el intradía lo destruyó.
2. **El bruto overnight es flojo y no significativo.** +6,6 %/año suena bien,
   pero con **Sharpe 0,58** y un **IC95 del retorno medio que cruza el cero
   ([−1,4; +7,1] pb/d; Sharpe IC95 [−0,29; +1,51])**: en esta ventana **no se
   puede rechazar que el overnight bruto sea cero**.
3. **El coste del cruce diario lo mata.** Punto de equilibrio ≈ **gross/2 =
   1,41 pb por lado**. Con 1 pb/lado queda un hilo (+1,3 %/año, Sharpe 0,17); con
   **2 pb/lado ya es negativo** (−3,7 %/año); con 5 pb/lado es ruina (−17 %/año y
   maxDD −62 %). Son ~**504 cruces/año** (2 × 252); a 2 pb/lado eso es **~10 %/año
   de drag puro de ejecución**.
4. **SPY buy&hold gana a todo, incluso en bruto.** +12,2 %/año con **una sola
   operación**. Y el propio overnight de SPY (+6,4 %/año) ≈ el overnight bruto de
   la cesta equal-weight: puedes capturar el retorno nocturno del mercado
   **manteniendo SPY**, sin 504 cruces anuales. La versión "churn diario" de la
   anomalía no aporta nada sobre comprar el índice y dormir.
5. **El intradía negativo es de small/mid caps, no del índice.** En SPY *ambos*
   tramos fueron positivos (ON +6,4 %/año, ID +5,7 %/año). El intradía fuertemente
   negativo aparece en la cesta amplia equal-weight (valores más pequeños y
   baratos), y se suaviza al subir el filtro de liquidez (líq ≥10 M$: intradía
   −3,7 % vs −10 % en ≥1 M$). Otro recordatorio de que equal-weight sobre 3.000
   nombres sobrepondera la cola ilíquida.

---

## 3. Año a año (bruto, pb/día) — la anomalía no es estable

| Año | OVERNIGHT | INTRADÍA | Cierre-cierre |
|---|---:|---:|---:|
| 2021 (Q4) | +4,74 | −15,08 | −10,39 |
| **2022 (oso)** | **−5,59** | **−5,38** | −11,16 |
| 2023 | +3,90 | +0,18 | +4,00 |
| 2024 | +8,65 | −7,20 | +1,17 |
| 2025 | +2,33 | −1,20 | +0,71 |
| 2026 (parcial) | +4,90 | −1,35 | +3,28 |

El patrón "overnight gana / intradía pierde" se cumple en 2021, 2024, 2025 y 2026,
es tibio en 2023 (ambos ~planos) y **falla en 2022**, cuando el overnight también
fue negativo (−5,6 pb/d): en el oso la noche **no protegió**. La media de 5 años
descansa en gran parte sobre **2024** (+8,65 pb/d). Un edge que depende de un año
y desaparece en el bear no es la máquina de imprimir dinero del folclore.

---

## 4. Haircut de supervivencia — aquí es casi nulo

| Overnight líq ≥1 M$ | Bruto pb/d |
|---|---:|
| **Survivorship-free** (incluye deslistados) | +2,815 |
| **Solo survivors** (tickers vivos al final de la muestra) | +2,834 |
| **Haircut** | **≈ −0,02 pb/d (≈ 0)** |

Contraintuitivo pero honesto: **el sesgo de supervivencia apenas mueve la aguja**
en esta estrategia y ventana. Razones: (a) la estrategia **sale cada mañana**, así
que no se queda con los muertos cayendo a cero (el daño típico de supervivencia se
concentra en el camino a la quiebra que un buy&hold sí sufre); (b) 5 años son
pocas bajas relativas a ~3.000 nombres; (c) a equal-weight sobre miles de nombres,
los retornos *overnight* de los que van a deslistarse no son sistemáticamente
distintos en ese tramo horario. Conclusión: **el enemigo de esta estrategia no es
el sesgo de supervivencia, son los costes.** (Haberlo medido en limpio es
justamente lo que nos deja afirmarlo sin sesgo.)

---

## 5. Variantes con filtro (apartado c)

- **top-500 por dólar-volumen:** marginalmente mejor en bruto (+6,8 % vs +6,6 %)
  y Sharpe (0,63), pero el mismo problema de coste (equilibrio 1,44 pb/lado,
  negativo a 2 pb/lado). No rescata el neto.
- **excluir peor quintil de momentum (21 d):** **empeora** el bruto (+4,9 %) — en
  este universo quitar los rezagados no ayuda al overnight.
- Subir liquidez a ≥10 M$ **no mejora**: menos nombres, mismo bruto, igual muerte
  por coste.

Ningún filtro cambia el veredicto: el margen bruto es demasiado pequeño frente al
peaje de operar todos los días a ambos lados.

---

## 6. Caveats (lo que este estudio NO prueba y por qué)

1. **Dividendos:** Massive ajusta solo splits, no dividendos. En el día ex-div la
   apertura cae por el dividendo → el overnight medido **infravalora** el retorno
   total (~1–1,5 %/año). Es un sesgo **conservador** (el overnight real bruto es
   algo mayor), pero **no cambia** el veredicto de costes, y afecta igual al bench
   SPY (misma convención).
2. **Capacidad / operabilidad:** equal-weight sobre ~3.000 tickers con rotación
   diaria total es **inoperable** para un libro pequeño (miles de MOC+MOO cada
   día). Es un backtest de laboratorio; en la práctica el coste efectivo sería
   mayor que el modelado, no menor.
3. **Microestructura del open:** `open_t/close_{t-1}` es ejecutable vía subasta
   MOC (compra) + MOO (venta) en nombres líquidos; en la cola ilíquida el "open"
   es un print fino. Los filtros ≥10 M$ y top-500 dan el mismo resultado, lo que
   sugiere que **no es puramente un artefacto de bid-ask bounce**.
4. **Retornos de deslistado:** la pérdida final más allá de la última barra no se
   modela; al salir cada mañana su impacto en esta estrategia es menor (y afecta
   igual a ON e ID).
5. **Ventana corta (5 años):** límite del plan Starter. Incluye un oso (2022) y
   dos toros, pero no es un ciclo largo. Los resultados son indicativos, no
   definitivos.

---

## 7. Veredicto

**¿Es el overnight un edge real y explotable NETO de costes?**

- **Real, sí — en dirección y en bruto:** en acciones el retorno se concentra en
  la noche y el intradía resta. Reproducido en limpio y sin sesgo de
  supervivencia.
- **Explotable neto, no:** como cartera long-only equal-weight con rebalanceo
  diario, el bruto (~+6,6 %/año, Sharpe 0,58, **ni significativo**) **se evapora
  con ~1,4 pb/lado** y se vuelve **negativo con una ejecución realista (≥2
  pb/lado)**. El doble cruce diario (~504 operaciones/año) es el sumidero.
- **Peor aún para la tesis:** **comprar y mantener SPY** (+12,2 %/año, una sola
  operación, Sharpe 0,75) **bate a la estrategia overnight incluso en bruto**, y
  su propio tramo nocturno te lo llevas sin rotar nada.

**Recomendación:** no perseguir el overnight como estrategia de rotación diaria.
El fenómeno es un hecho estilizado interesante (y una razón más para **no** estar
corto de mercado de un día para otro), pero **no es una fuente de alfa neto** a
este coste de ejecución. Si algo, sugiere que el *timing* de entradas/salidas de
otras estrategias debería evitar pagar el tramo intradía y favorecer mantener
posiciones a través de la noche — pero eso es *reducir* cruces, no añadir 504 al
año. La honestidad por delante del resultado: aquí el resultado neto es **no**.

---

*Reproducible:* `python backtest_overnight_massive.py 2021-10-06 2026-10-03`
(reusa la caché local gitignored; ~1.300 peticiones *grouped* la primera vez).
