# ¿Necesito otra fuente de datos aparte de Alpaca? — análisis honesto

Pregunta de Eduardo: *¿te vendría bien algún sitio aparte de Alpaca que te diera
más recursos (indicadores de gráficos u otras cosas)?*

Respuesta corta: **no necesito nada de pago.** Ya tengo cableado un stack gratuito
que cubre ~95% de lo que el sistema necesita. El único límite real no es de datos,
es de **ejecución** (Alpaca paper no llega a las bolsas europeas). Abajo el detalle,
todo comprobado hoy, no de memoria.

## 1. Lo que YA tengo cableado (y es más de lo que parece)

| Fuente | Módulo | Qué da | Estado |
|---|---|---|---|
| **Alpaca (paper)** | `alpaca.py`, `alpaca_trade.py` | Ejecución + intradía US (IEX) | ✅ ejecución |
| **Yahoo Finance** | `yahoo.py` | OHLCV diario/semanal/intradía, **índices y acciones europeas directas**, EUR/USD, **VIX y VIX3M**, cierre ajustado, historia larga | ✅ gratis |
| **TradingView** | `tradingview.py` | Snapshot técnico en vivo: RSI/MACD/ADX/Bollinger/Stochastic en varias temporalidades + recomendación agregada | ✅ gratis |
| **FRED** | `fred.py` | Macro (tipos, inflación, desempleo…) | ✅ gratis |
| **SEC / 13F** | `sec.py`, `form13f.py` | Posiciones institucionales (13F) | ✅ gratis |
| **Noticias** | `noticias_*.py`, `forexfactory.py` | Titulares + calendario macro | ✅ gratis |

**Hallazgo clave de hoy:** Yahoo me da cosas que yo creía que me faltaban:
- **Indicadores de gráfico:** no necesito un "sitio de indicadores" tipo TradingView
  Pro. Los indicadores (RVOL, VWAP, ATR, RSI, MACD, SMA/EMA, Bollinger, ADX,
  Chandelier…) **los calculo yo** a partir del OHLCV, que es lo correcto para
  backtesting (controlo la fórmula, evito look-ahead). Y para una "segunda opinión
  en vivo" ya tengo el scanner de TradingView gratis.
- **Intradía europeo:** Yahoo sirve velas de 15m del DAX, ASML, SAP… actualizándose
  en vivo (comprobado: DAX 15m a las 07:36 de hoy con Europa abierta). Puedo
  **analizar** Europa intradía directamente, no solo vía proxies US.
- **Estructura temporal del VIX:** ^VIX (16,2) y ^VIX3M (18,6) → desbloquea el
  filtro de régimen (idea #6) **gratis**.

## 2. El único límite REAL: ejecución, no datos

Alpaca paper **no puede ejecutar en bolsas europeas** (Euronext, Xetra, BME…). Solo
opera instrumentos cotizados en EE.UU. Por eso operamos Europa vía **ADRs y ETFs
US-listed** (ASML, SAP, ETFs de país EWG/EWQ/EWP…). Es un buen proxy, pero:
- Los ADRs cotizan en horario US, con su propia liquidez (a veces menor).
- Hay empresas europeas sin ADR líquido → inaccesibles.

**Si algún día quisieras ejecución europea de verdad** (no es necesario ahora), la
opción sería una cuenta **paper de Interactive Brokers**: sí llega a Euronext/Xetra/
BME, tiene API de Python (`ib_insync`) y cuenta demo gratis. Es la única "fuente
aparte" que cambiaría algo de fondo — y es un broker, no un proveedor de datos.
Mi recomendación: **no hace falta todavía**; los ADRs/ETFs cubren el aprendizaje y
la estrategia perfectamente. Lo dejo anotado por si en el futuro quieres dar el salto.

## 3. Huecos menores y cómo taparlos (todo gratis)

| Hueco | Para qué | Solución gratis | ¿Merece la pena? |
|---|---|---|---|
| Barras **diarias** en Alpaca (IEX no las da) | backtests | Ya resuelto: uso Yahoo / caché `datos/mercado/1d` | ✅ hecho |
| **Calendario de resultados** (earnings) | módulo PEAD (idea #3) | Yahoo (`/calendar`) o Alpha Vantage free (`EARNINGS`) — Alpha Vantage responde (HTTP 200 comprobado) | 🔸 sí, si montamos PEAD |
| **VIX term structure** | filtro de régimen (idea #6) | Yahoo ^VIX/^VIX3M — comprobado | ✅ ya disponible |
| **Fundamentales** (value/quality) | factores fundamentales | — (requiere feed fiable de pago) | ❌ los descartamos ya; OHLCV basta |
| **Order book L2 / opciones** | market-making, vol selling | — (infra de pago) | ❌ no hacemos esas estrategias |

## 4. ¿Y una fuente de PAGO (Polygon, Tiingo, EODHD…)?

Darían datos más limpios y **sin rate-limit** (Yahoo a veces devuelve 429 y hay que
reintentar). Para un proyecto de dinero real con backtesting masivo, valdría la pena.
**Para esto (paper + aprendizaje, coste cero), NO.** El rate-limit de Yahoo se
gestiona con reintentos y caché; no justifica pagar. Fiel a tu principio: *aquí no
hay nada que cueste dinero.*

## 5. Recomendación final

1. **No contratar nada.** El stack gratuito (Yahoo + TradingView + FRED + SEC +
   Alpaca) cubre datos, indicadores y análisis europeo.
2. **Aprovechar mejor Yahoo:** cablear VIX term-structure (régimen) y, si montamos
   PEAD, el calendario de earnings. Ambos gratis.
3. **Anotar IB paper** como única mejora futura *de ejecución* (no de datos) si
   algún día queremos operar bolsas europeas de verdad. Hoy no hace falta.

El cuello de botella de este sistema no es la información — es la **selección** y la
**gestión de riesgo**. Ahí es donde seguimos mejorando.
