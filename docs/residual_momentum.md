# Residual momentum (Blitz, Huij & Martens 2011) — investigado y **NO desplegado**

*2 de octubre de 2026.* Reproducible: `python validar_residual_momentum.py`.

Un resultado negativo, documentado a propósito. La disciplina del proyecto es *validar →
cablear solo si pasa*; esto **no pasa**, y registrar por qué vale tanto como un acierto.

## La idea

El momentum convencional compra lo que más ha subido. El problema: parte de esa subida es
simplemente **exposición a factores** (beta al mercado, a tamaño, a valor), no habilidad.
El **residual momentum** regresa cada acción contra los factores Fama-French 3 (Mkt-RF,
SMB, HML), se queda con el **residuo** (el movimiento idiosincrático) y hace momentum
*sobre eso*. En la literatura —universos amplios de miles de valores— da mayor Sharpe y,
sobre todo, evita los **"momentum crashes"** (los desplomes brutales del momentum en los
rebotes tras un crash).

Implementado de verdad: betas FF3 sobre ventana de 36 meses, señal = media de los residuos
de formación (11 meses, saltando el último) estandarizada por su vol, largo/corto por
quintiles, mensual. Factores bajados de la librería de Ken French (1926–2026).

## Qué sale en NUESTRO universo (111 mega-caps US locales)

| Estrategia | Sharpe | Vol | maxDD |
|---|---:|---:|---:|
| Residual momentum L/S | 0,21 | 12,3% | −40,7% |
| Convencional L/S | 0,20 | 22,7% | −71,2% |
| Residual momentum long-only\* | 1,16 | 15,6% | −40,5% |
| Convencional long-only\* | 1,25 | 18,5% | −45,0% |

\* *Long-only inflado por **sesgo de supervivencia** (el universo son los supervivientes de
hoy); no es un número fiable como alfa.*

## Lo que es real… y lo que no

**Real (el método funciona):**
- **Neutralidad a factores por construcción:** correlación del L/S con Mkt-RF −0,14, SMB
  −0,07, HML +0,03 — prácticamente cero, justo lo que promete el método (el convencional
  arrastra HML −0,18).
- **Crash-resistance:** mismo Sharpe que el convencional L/S (0,21 vs 0,20) pero con **la
  mitad de volatilidad (12% vs 23%) y la mitad de drawdown (−41% vs −71%)**. Esto reproduce
  el hallazgo central de Blitz.

**Lo que impide desplegarlo:**
- **No alcanza el bar:** DSR vs azar **0,63** (bar 0,95) y **PBO 0,85** (sobreajuste alto).
  Como libro propio, el L/S es demasiado débil y la elección de parámetros es ruido.
- **No es diversificador:** correlación **0,60** con el momentum convencional que ya corre
  el Libro B. Es una versión *más limpia* de lo mismo, no una fuente ortogonal (a
  diferencia de la rotación Europa, que estaba a −0,02 con el swing).
- **Universo equivocado:** el residual momentum brilla en cross-sections amplias (miles de
  valores). 111 mega-caps homogéneas y con sesgo de supervivencia es el peor banco de
  pruebas posible, y el DSR lo confirma.

## Decisión

**NO se cablea** —ni como libro nuevo ni como refinamiento— con los datos de hoy. Su única
virtud real (crash-resistance) sería un argumento para, en el futuro, **neutralizar por
factores la señal del Libro B** y así reducir su riesgo de desplome; pero eso exige un
**universo amplio y sin sesgo de supervivencia** (un vendor de datos tipo CRSP/Sharadar),
que no tenemos. Queda registrado y reproducible para revisitarlo si llega ese dato.

> Moraleja de método: el residual momentum es un *buen* algoritmo mal servido por nuestro
> universo. Rechazarlo aquí no es rechazar la idea; es negarse a desplegar un Sharpe 0,2
> con PBO 0,85 solo porque la teoría es bonita. Eso es exactamente la línea entre un
> proceso cuant y un cazador de backtests.
