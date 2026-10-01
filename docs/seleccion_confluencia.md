# Selección por confluencia: elegir solo las mejores (no "el mínimo")

Principio: hay miles de tickers × toda la sesión = muchísimas oportunidades. **No
queremos todas, queremos las mejores.** Por eso una operación ya no entra por
cumplir *un* mínimo (p. ej. RVOL ≥ 1,5), sino por pasar **varias confirmaciones
independientes a la vez** (confluencia). Si nada las pasa, **efectivo**.

## Universo amplio, listón alto

- **Universo ampliado** (`gestionar.UNIVERSO` + watchlist dinámica): ~95 valores
  líquidos de tecnología, semis, consumo, energía, salud, financieras e índices.
  Cuantos más miramos, mejor podemos **elegir**.
- La selección decide, no el tamaño del universo.

## Las 6 confirmaciones (gates de calidad) para comprar

Un candidato debe cumplir **todas** (`rankear_conviccion`):

1. **RVOL ≥ umbral fuerte** — volumen anormal (*stock in play*), no un valor dormido.
2. **Sobre VWAP** (dist_vwap > 0) — compradores al mando.
3. **No sobre-extendido** (ext < 4 ATR sobre el VWAP) — *no perseguir* un valor ya
   estirado (entrar tarde es comprar caro).
4. **Fuerza relativa > +0,2 %** — **bate al mercado hoy** (retorno del valor − retorno
   del SPY). Un verdadero *stock in play* sube MÁS que la marea, no solo con ella. ←
   indicador nuevo.
5. **Sesión del valor ≥ +0,3 %** — movimiento con sustancia, no un +0,05 % de ruido.
6. **Catalizador no muy negativo** — no comprar contra una mala noticia.

Entre los que pasan, se ordena por **convicción**:

    convicción = z(RVOL) + z(ret_sesión) + z(fuerza_relativa) + 0,5·catalizador + memoria

y se opera el top; si no hay huecos o nadie pasa, **efectivo**. Esto implementa el
único caso con edge neto positivo del estudio (figura 26): la **confluencia**, no un
filtro suelto. Ejemplo de por qué sirve: las operaciones flojas de XLF (sesión
+0,15 %, convicción negativa) o un candidato que no bate al mercado quedan
**descartadas** — ya no entran "por cumplir el mínimo".

## Cooldown a prueba de timing (el caso NKE)

NKE se compró, saltó el SL, y se **re-compró 3 min después** → volvió a saltar. El
cooldown debía evitarlo pero falló por un problema de orden: `gestionar` miraba el
cooldown **antes** de que `registro` reconciliara la pérdida. Arreglado con tres
capas:

1. **Reconciliar ANTES** de reinvertir (el SL reciente ya está en el log).
2. **Cooldown en vivo desde los fills de Alpaca** (`_sl_recientes_fills`):
   reconstruye round-trips FIFO y marca cualquier venta a pérdida reciente **al
   instante**, sin depender del CSV.
3. **Nunca re-comprar lo cerrado este mismo ciclo** (se añade `a_cerrar` al cooldown).

Así, un nombre que acaba de saltar el SL queda vetado 60 min aunque la foto siga
guapa: un valor que te saca suele estar en *whipsaw*, no en tendencia.
