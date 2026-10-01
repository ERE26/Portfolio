# Intradía europeo: cómo operamos Europa (y la restricción honesta)

## La restricción (importante)

Nuestro bróker es **Alpaca paper**, que **solo** da acceso a acciones y ETFs
cotizados en **EE.UU.** (y cripto). **No** podemos operar directamente en las
bolsas europeas (Xetra, Euronext París/Ámsterdam, Borsa Italiana, BME Madrid,
LSE…). Para eso haría falta otro bróker (p. ej. Interactive Brokers) y otra
infraestructura.

## La vía real: ADRs europeos + ETFs de Europa (cotizados en EE.UU.)

Sí podemos tener **exposición europea intradía** operando:

- **ADRs** de grandes empresas europeas que cotizan en NYSE/NASDAQ: **ASML** (NL),
  **SAP** (DE), **Novo Nordisk NVO** (DK), **AstraZeneca AZN** (UK), **Shell SHEL**,
  **TotalEnergies TTE**, **Novartis NVS** (CH), **Unilever UL**, **Diageo DEO**,
  **RELX**, **Sanofi SNY**, **Stellantis STLA**, **Spotify SPOT**, **Ericsson ERIC**,
  **ING**, **Barclays BCS**, **Nokia NOK**, **Philips PHG**, **Eni E**,
  **AB InBev BUD**.
- **ETFs de países/región de Europa**: **VGK** (Europa), **FEZ** (Euro Stoxx 50),
  **EZU** (Eurozona), **HEDJ** (Europa con divisa cubierta), **EWG** (Alemania),
  **EWU** (Reino Unido), **EWQ** (Francia), **EWP** (España), **EWL** (Suiza),
  **EWN** (Países Bajos), **EWI** (Italia), **EWD** (Suecia).

Verificado: 38 de 39 candidatos son *tradable* en Alpaca y tienen datos de 15m
(feed IEX). La lista operada vive en `gestionar.EUROPA` (33 nombres) y entra en
`universo_dinamico()`: los europeos **compiten de igual a igual** con los de EE.UU.
en la selección por confluencia — se opera el mejor, venga de donde venga.

## Cómo encaja con la estrategia

- **Misma estrategia** (confluencia): RVOL + sobre VWAP + fuerza relativa +
  no sobre-extendido + catalizador + sesión con sustancia. Calidad > cantidad.
- **Horario:** se operan en **horario de EE.UU.** (15:30–22:00 hora de Madrid en
  verano / 15:30–22:00 ≈ 9:30–16:00 ET). No en horario europeo.
- **Gap por la sesión europea:** el subyacente europeo cierra antes (≈17:30 CET),
  así que estos ADRs/ETFs suelen **abrir con gap** recogiendo la sesión europea y
  sus noticias. Eso los hace interesantes para catalizadores, y encaja con el
  filtro de *stock in play* (RVOL alto por noticia europea).

## Refinamiento pendiente (honesto)

La **fuerza relativa** se mide hoy contra el **SPY** (el mercado en el que cotizan
durante el horario de EE.UU.). Para los nombres europeos sería algo más preciso
medirla contra un benchmark europeo (**VGK** o **FEZ**). Es una mejora menor
pendiente; de momento el SPY es razonable porque operan en el mercado de EE.UU.

## Qué NO es esto

No es trading en bolsas europeas en horario europeo. Es exposición a empresas y
mercados europeos a través de instrumentos cotizados en EE.UU. Si en el futuro se
quiere operar Xetra/Euronext de verdad, hay que cambiar de bróker (IBKR) — un
proyecto aparte.
