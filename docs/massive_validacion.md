# Validación en vivo: API de acciones de Massive (tier gratis "Stocks Basic")

Fecha: 2026-10-03 · Alcance: SOLO LECTURA, PAPER, cero gasto (solo tier gratis).
Validador: `validar_massive.py` · Cliente: `fuentes/massive.py`.

Esta es la validación **empírica** que faltaba en `docs/massive_smoketest.md`:
allí la `MASSIVE_API_KEY` era un placeholder (`<...>`) y Massive la rechazaba con
401 "Unknown API Key", así que no se descargó ni un dato. El usuario ya corrigió
la clave (32 caracteres, sin `<`/`>`), y ahora la API responde. Se ejecutaron 12
peticiones reales respetando el límite de 5 req/min.

## Resultados clave

| Prueba | Resultado | OK |
|---|---|:---:|
| **Conector** (`/v3/reference/tickers/AAPL`) | `status: OK`, Apple Inc., XNAS, CIK 0000320193, active — **sin 401** | ✅ |
| **EOD SPY vs Yahoo** (~120 sesiones) | 126/126 barras comunes, 0 ausentes; dif. retorno mediana 0,0 pb (p99 18,9 pb, 2 días >10 pb); dif. cierre último 0,0%; dif. volumen mediana 0,0% | ✅ |
| **EOD AAPL vs Yahoo** (~120 sesiones) | 126/126 barras comunes, 0 ausentes; dif. retorno mediana 0,0 pb (p99 6,5 pb, 0 días >10 pb); dif. cierre último 0,0%; dif. volumen mediana 0,0% | ✅ |
| **Última barra** | 2026-10-02 (sesión previa; hoy es 2026-10-03) → confirma **EOD / fin de día** | ✅ |
| **Profundidad de histórico** | Pedí SPY desde 2015-01-01 → sólo 501 barras, primera **2024-10-03** → **~2 años** exactos (coincide con lo documentado del plan gratis) | ✅ |
| **Ajuste por splits** | ZCMD 2:1 el 2026-10-02 → ratio raw/adj 0,5000 (esperado 0,5000) → `adjusted=true` cuadra | ✅ |
| **Survivorship: deslistados** | 2000 deslistados (CS) en 2 páginas; **97,2%** con `delisted_utc`; bajas de 2004-06-29 a 2026-10-02 | ✅ |
| **Deslistados con barras descargables** | **3/3** de la muestra reciente traen OHLCV | ✅ |

### Profundidad real del histórico (plan gratis)

**~2 años.** Aunque pedí SPY desde `2015-01-01`, Massive sólo devolvió **501
barras** empezando el **2024-10-03** (exactamente dos años antes de hoy). Es el
recorte del tier gratis "Stocks Basic", no un fallo del cliente. Para backtests
que necesiten más de ~2 años de EOD habría que subir de plan (fuera de alcance:
cero gasto).

### Survivorship-free confirmado (con ejemplo)

Los deslistados **sí** traen barras, que es lo que elimina el sesgo de
supervivencia frente a Yahoo. Muestra de los 3 deslistados más recientes:

| Ticker | Nombre | Baja | Barras últimos 90d | Última barra |
|---|---|---|---:|---|
| COLA | Columbus Acquisition Corp | 2026-10-02 | 56 | 2026-10-01 |
| **DBRG** | **DigitalBridge Group, Inc.** | **2026-10-01** | **61** | **2026-09-29** |
| AIXC | AIxCrypto Holdings, Inc. | 2026-09-30 | 62 | 2026-09-29 |

Ejemplo: **DBRG** (DigitalBridge) se deslistó el 2026-10-01 y Massive sirve sus
**61 sesiones** previas (última 2026-09-29). Yahoo ya no lo serviría limpio.

## VEREDICTO

**DATOS BUENOS — survivorship-free confirmado.** El conector autentica sin 401;
el EOD de SPY/AAPL calca a Yahoo (OHLCV y retornos coherentes, 0 sesiones
ausentes); el ajuste por splits cuadra; el plan gratis da **~2 años** de
histórico EOD (first bar 2024-10-03); y los **deslistados traen barras** (3/3 de
la muestra, p. ej. DBRG con 61 barras hasta su baja), que es justo lo que quita
el sesgo de supervivencia. La única limitación operativa es la profundidad de
~2 años del tier gratis.
