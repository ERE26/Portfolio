"""Interfaz visual del buscador de patrones (Streamlit).

Metes uno o varios tickers y la app:
  - dibuja el tramo actual del grafico,
  - busca en la memoria los momentos historicos mas parecidos (de OTROS valores),
  - explica POR QUE se parecen (que indicadores coinciden),
  - muestra lo que paso despues de cada patron parecido,
  - da una prediccion con su nivel de confianza,
  - y compara/ordena varios tickers por la fuerza y fiabilidad de su senal.

Es un envoltorio del motor que ya existe (caso_vivo + MemoriaPatrones): no hay
logica de prediccion nueva aqui, solo visualizacion. Para analizar cada ticker
como "no visto" se excluye su propio historico de los vecinos (excluir_ticker),
sin reconstruir la memoria.

Arranque:  streamlit run app.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from patrones import contexto as mod_ctx
from patrones import ventanas as mod_v
from patrones.explicar import por_que_parecido
from patrones.figuras import describir, texto
from patrones.memoria import MemoriaPatrones

# ---- paleta coherente con las figuras del trabajo
AZUL, NAR, VER, ROJO, GRIS, TINTA = "#2563eb", "#e8710a", "#059669", "#dc2626", "#64748b", "#1e293b"

# ---- parametros por defecto segun la temporalidad
PRESETS = {
    "1d": dict(carpeta="datos/mercado", largo=40, horizonte=10, paso=3, anos=10, etq="1 dia"),
    "1Hour": dict(carpeta="datos/intraday", largo=40, horizonte=6, paso=4, anos=5, etq="1 hora"),
    "15Min": dict(carpeta="datos/intraday", largo=40, horizonte=13, paso=13, anos=5, etq="15 min"),
}

# ====================================================================== datos
def _leer_intraday(carpeta, intervalo):
    from fuentes.descarga import _unsafe
    from fuentes.universo import catalogo
    sub = os.path.join(carpeta, intervalo)
    cat = catalogo()
    pre, filas = {}, []
    for f in sorted(os.listdir(sub)):
        if not f.endswith(".csv"):
            continue
        t = _unsafe(f[:-4])
        df = pd.read_csv(os.path.join(sub, f), parse_dates=["fecha"]).set_index("fecha")
        if len(df) < 500:
            continue
        pre[t] = df
        info = cat.get(t, {"clase": "accion", "sector": "desconocido"})
        filas.append({"ticker": t, "clase": info.get("clase", "accion"),
                      "sector": info.get("sector", "desconocido"), "nombre": t})
    return pre, pd.DataFrame(filas)


@st.cache_resource(show_spinner=False)
def cargar_universo(intervalo):
    """Carga el universo de precios y su metadata para una temporalidad."""
    pre = PRESETS[intervalo]
    if intervalo == "1d":
        from fuentes.descarga import cargar_mercado
        precios, meta = cargar_mercado(pre["carpeta"], intervalo="1d")
    else:
        precios, meta = _leer_intraday(pre["carpeta"], intervalo)
    return precios, meta


@st.cache_resource(show_spinner=False)
def construir_memoria(intervalo, anos, largo, horizonte, paso, k):
    """Construye el banco de casos y la memoria UNA vez (se cachea)."""
    precios, meta = cargar_universo(intervalo)
    # calibracion de escala con pocas muestras (solo afecta a la confianza, no a
    # los vecinos): hace manejable un universo grande, como en bot.py buscar.
    import patrones.memoria as _mm
    if not getattr(_mm, "_cal_parcheado", False):
        _orig = _mm.MemoriaPatrones._calibrar_escala
        _mm.MemoriaPatrones._calibrar_escala = \
            lambda self, n_muestra=60, semilla=0: _orig(self, min(n_muestra, 60), semilla)
        _mm._cal_parcheado = True

    cfg = mod_v.Config(largo=largo, horizonte=horizonte, paso=paso)
    recorte = {}
    if anos and anos > 0:
        corte = pd.Timestamp.now() - pd.Timedelta(days=int(anos * 365))
        minimo = cfg.minimo_historia + cfg.largo + cfg.horizonte + 5
        for t, d in precios.items():
            d2 = d[d.index >= corte]
            if len(d2) >= minimo:
                recorte[t] = d2
    else:
        recorte = precios
    ctx = mod_ctx.Contexto(recorte, meta)
    banco = mod_v.construir(recorte, ctx, None, cfg, verbose=False)
    mem = MemoriaPatrones(cfg, k=k).ajustar(banco)
    return dict(precios=recorte, meta=meta, ctx=ctx, cfg=cfg, mem=mem,
                tickers=sorted(recorte))


def forma_vecino(mem, ticker, fecha):
    """Recupera la forma normalizada (log-cierre) de un vecino concreto."""
    pos = np.flatnonzero((mem.tickers == ticker) &
                         (mem.fechas == np.datetime64(pd.Timestamp(fecha))))
    if len(pos):
        return mem.banco.formas[pos[0]], mem.banco.formas_vol[pos[0]]
    return None, None


@st.cache_data(show_spinner=False)
def _descargar_yahoo(ticker, intervalo):
    """Descarga el historico de un ticker nuevo desde Yahoo (solo diario)."""
    from fuentes import yahoo
    df, _ = yahoo.historico(ticker, intervalo="1d", rango="max")
    return df


@st.cache_data(show_spinner=False)
def analizar(intervalo, anos, largo, horizonte, paso, k, ticker, n_similares):
    """Analiza un ticker: tramo actual, vecinos, explicacion y prediccion.

    Devuelve un dict serializable para cachear por (parametros, ticker)."""
    B = construir_memoria(intervalo, anos, largo, horizonte, paso, k)
    mem, cfg, precios, ctx = B["mem"], B["cfg"], B["precios"], B["ctx"]
    if ticker not in precios:
        # ticker nuevo: se descarga de Yahoo al vuelo (solo diario) y se analiza
        # como "no visto" (no entra en la memoria). ctx se reconstruye con el.
        if intervalo != "1d":
            return {"error": f"{ticker}: los tickers nuevos solo se pueden añadir en diario."}
        try:
            dfy = _descargar_yahoo(ticker, intervalo)
        except Exception as e:
            return {"error": f"No pude descargar {ticker} de Yahoo: {str(e)[:80]}"}
        if dfy is None or len(dfy) < cfg.minimo_historia:
            return {"error": f"{ticker}: histórico insuficiente desde Yahoo."}
        precios = {**precios, ticker: dfy}
        meta_ext = pd.concat([B["meta"], pd.DataFrame([{"ticker": ticker,
                              "sector": "desconocido", "nombre": ticker}])], ignore_index=True)
        ctx = mod_ctx.Contexto(precios, meta_ext)
    vivo = mod_v.caso_vivo(precios, ctx, None, cfg, ticker)
    pred = mem.predecir(vivo, 0, excluir_ticker=ticker)
    # evidencia: indicadores crudos del caso + voto de cada familia de indicadores
    indicadores = leer_indicadores(vivo.feat[0], vivo.nombres_feat, intra=(intervalo != "1d"))
    votos = [{"grupo": g, "label": GRUPO_ETQ.get(g, g), "prob": float(pr),
              "peso": float(mem.pesos.get(g, 0.0))} for g, pr in pred.probs_grupo.items()]
    votos.sort(key=lambda d: -d["peso"])
    vec = pred.vecinos.head(n_similares).copy()
    # forma del tramo actual y de cada vecino (para el dibujo de parecido)
    forma_obj = vivo.formas[0].tolist()
    desc_obj = texto(describir(vivo.formas[0], vivo.formas_vol[0]))
    filas = []
    for _, v in vec.iterrows():
        fv, _ = forma_vecino(mem, v["ticker"], v["fecha"])
        dv = texto(describir(fv)) if fv is not None else ""
        filas.append({
            "ticker": v["ticker"], "fecha": pd.Timestamp(v["fecha"]),
            "similitud": float(1 - v["distancia"]),
            "ret": float(v["ret"]), "mfe": float(v["mfe"]), "mae": float(v["mae"]),
            "porque": por_que_parecido(v, mem.pesos), "figura": dv,
            "forma": (fv.tolist() if fv is not None else None),
        })
    # precios recientes para el candlestick
    df = precios[ticker].tail(largo + horizonte)
    ohlc = {"fecha": [str(x) for x in df.index], "open": df["open"].tolist(),
            "high": df["high"].tolist(), "low": df["low"].tolist(),
            "close": df["close"].tolist()}
    atr = float(pred.atr_pct)
    sector = ctx.sector_de(ticker) if ctx is not None else "desconocido"
    return {
        "ticker": ticker, "fecha": str(pred.fecha), "figura": desc_obj, "sector": sector,
        "prob_sube": float(pred.prob_sube), "ret_esperado": float(pred.ret_esperado),
        "confianza": float(pred.confianza), "abstencion": bool(pred.abstencion),
        "atr_pct": atr, "forma_obj": forma_obj,
        "camino": (np.asarray(pred.camino) * atr * 100).tolist(),
        "camino_p10": (np.asarray(pred.camino_p10) * atr * 100).tolist(),
        "camino_p90": (np.asarray(pred.camino_p90) * atr * 100).tolist(),
        "pct_vecinos_suben": float((vec["ret"] > 0).mean()) if len(vec) else float("nan"),
        "n_vecinos": int(len(vec)), "vecinos": filas, "ohlc": ohlc,
        "indicadores": indicadores, "votos": votos,
    }


# ================================================================ componentes
def tarjeta_prediccion(r):
    sube = r["prob_sube"] > 0.5
    signo = "AL ALZA ▲" if sube else "A LA BAJA ▼"
    color = VER if sube else ROJO
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"**Sesgo**<br><span style='color:{color};font-size:1.5em;font-weight:700'>"
                f"{signo}</span>", unsafe_allow_html=True)
    c2.metric("Prob. de subida", f"{r['prob_sube']:.0%}")
    c3.metric("Retorno esperado", f"{r['ret_esperado']*100:+.2f}%")
    fiab = "BAJA ⚠️" if r["abstencion"] else "OK ✓"
    c4.metric("Confianza", f"{r['confianza']:.2f}", fiab)
    if r["abstencion"]:
        st.warning("Confianza baja: no hay patrones suficientemente parecidos. "
                   "Tómalo con cautela (el sistema 'se abstendría').")


def grafico_parecido(r):
    """Superpone la forma del tramo actual con la de los vecinos: se VE el parecido."""
    fig = go.Figure()
    x = list(range(len(r["forma_obj"])))
    for v in r["vecinos"]:
        if v["forma"] is not None:
            fig.add_trace(go.Scatter(x=x, y=v["forma"], mode="lines",
                                     line=dict(color=GRIS, width=1), opacity=0.45,
                                     name=f"{v['ticker']} {v['fecha'].date()}",
                                     hovertemplate="%{fullData.name}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=r["forma_obj"], mode="lines",
                             line=dict(color=AZUL, width=3.5), name=f"{r['ticker']} (ahora)"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=36, b=10),
                      title="Forma actual vs. patrones parecidos (cierre normalizado)",
                      showlegend=False, plot_bgcolor="white",
                      yaxis_title="forma (media 0, desv. 1)", xaxis_title="velas de la ventana")
    fig.update_xaxes(showgrid=True, gridcolor="#eef2f7")
    fig.update_yaxes(showgrid=True, gridcolor="#eef2f7")
    return fig


def grafico_velas_proyeccion(r, horizonte):
    """Candlestick del tramo actual + proyeccion esperada con banda p10-p90.

    Eje X numerico (indice de vela) para no mezclar fechas con etiquetas futuras;
    las marcas se re-etiquetan con la fecha historica y '+N' para la proyeccion.
    """
    o = r["ohlc"]
    n = len(o["close"])
    xh = list(range(n))
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=xh, open=o["open"], high=o["high"], low=o["low"],
                                 close=o["close"], name=r["ticker"], text=o["fecha"],
                                 increasing_line_color=VER, decreasing_line_color=ROJO))
    # proyeccion: parte del ultimo cierre, en % acumulado convertido a precio
    c0 = o["close"][-1]
    xf = list(range(n - 1, n + horizonte))

    def a_precio(camino):
        return [c0] + [c0 * (1 + p / 100.0) for p in camino]
    fig.add_trace(go.Scatter(x=xf, y=a_precio(r["camino_p90"]), mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xf, y=a_precio(r["camino_p10"]), mode="lines", fill="tonexty",
                             line=dict(width=0), fillcolor="rgba(37,99,235,0.12)",
                             name="rango p10-p90", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xf, y=a_precio(r["camino"]), mode="lines",
                             line=dict(color=AZUL, width=2.5, dash="dot"), name="esperado"))
    fig.add_vline(x=n - 1, line_width=1, line_dash="dot", line_color=GRIS)
    # marcas: algunas fechas historicas + la proyeccion
    paso = max(1, n // 5)
    tv = list(range(0, n, paso)) + [n - 1 + horizonte]
    tt = [o["fecha"][i][:10] for i in range(0, n, paso)] + [f"+{horizonte}"]
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=36, b=10),
                      title=f"{r['ticker']}: tramo actual y proyección a {horizonte} velas",
                      xaxis_rangeslider_visible=False, plot_bgcolor="white",
                      legend=dict(orientation="h", y=1.02, x=0))
    fig.update_xaxes(showgrid=False, tickmode="array", tickvals=tv, ticktext=tt, tickangle=-30)
    fig.update_yaxes(showgrid=True, gridcolor="#eef2f7")
    return fig


def mini_forma(forma, color):
    fig = go.Figure(go.Scatter(y=forma, mode="lines", line=dict(color=color, width=2)))
    fig.update_layout(height=90, margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor="white",
                      showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


TONO_COL = {"alza": VER, "baja": ROJO, "neutro": GRIS}


def grafico_votos(r):
    """Diverge alrededor del 50%: qué dice CADA familia de indicadores por su cuenta.

    Es el 'por qué' cuantitativo: cada barra es lo que predeciría ese grupo con sus
    propios vecinos; el ancho de la familia en el sistema es su peso."""
    vs = [v for v in r["votos"] if v["peso"] > 0.005][:10]
    if not vs:
        return None
    labels = [f"{v['label']}  ·  peso {v['peso']:.0%}" for v in vs]
    x = [v["prob"] - 0.5 for v in vs]
    cols = [VER if v["prob"] >= 0.5 else ROJO for v in vs]
    txt = [f"{v['prob']:.0%} ↑" for v in vs]
    fig = go.Figure(go.Bar(y=labels, x=x, orientation="h", marker_color=cols,
                           text=txt, textposition="outside", cliponaxis=False))
    fig.update_layout(height=max(230, 32 * len(vs) + 70), margin=dict(l=10, r=10, t=40, b=10),
                      title="Voto por familia de indicadores", plot_bgcolor="white",
                      yaxis=dict(autorange="reversed"))
    fig.update_xaxes(range=[-0.55, 0.55], tickvals=[-0.5, -0.25, 0, 0.25, 0.5],
                     ticktext=["baja", "", "50%", "", "sube"], zeroline=True,
                     zerolinecolor=TINTA, zerolinewidth=1, gridcolor="#eef2f7")
    return fig


def ficha_tecnica_html(r):
    """Rejilla de indicadores del caso actual con su lectura y color."""
    celdas = []
    for ind in r["indicadores"]:
        c = TONO_COL.get(ind["tono"], GRIS)
        celdas.append(
            f"<div style='border:1px solid #e5e9f0;border-radius:8px;padding:7px 10px;background:#fff'>"
            f"<div style='font-size:0.78em;color:{GRIS}'>{ind['label']}</div>"
            f"<div style='font-size:1.1em;font-weight:700;color:{c}'>{ind['valor']}</div>"
            f"<div style='font-size:0.76em;color:{c}'>{ind['lectura']}</div></div>")
    return ("<div style='display:grid;grid-template-columns:repeat(auto-fill,minmax(145px,1fr));"
            "gap:8px'>" + "".join(celdas) + "</div>")


def vista_detalle(r, horizonte):
    st.subheader(f"{r['ticker']}  ·  {r['fecha'][:10]}")
    st.caption(f"Figura detectada: **{r['figura']}**")
    tarjeta_prediccion(r)
    g1, g2 = st.columns([1.1, 1])
    g1.plotly_chart(grafico_velas_proyeccion(r, horizonte), use_container_width=True)
    g2.plotly_chart(grafico_parecido(r), use_container_width=True)

    st.markdown("#### ¿Por qué esta predicción? — la evidencia")
    v1, v2 = st.columns([1, 1.15])
    fig_v = grafico_votos(r)
    if fig_v is not None:
        v1.plotly_chart(fig_v, use_container_width=True)
    else:
        v1.info("No hay voto por familia para este caso.")
    with v2:
        st.caption("Ficha técnica — indicadores del tramo actual")
        st.markdown(ficha_tecnica_html(r), unsafe_allow_html=True)

    st.markdown(f"**Patrones más parecidos** ({r['n_vecinos']}) — de los más parecidos, "
                f"subieron el **{r['pct_vecinos_suben']:.0%}**:")
    for v in r["vecinos"]:
        col = VER if v["ret"] > 0 else ROJO
        a, b = st.columns([1, 4])
        if v["forma"] is not None:
            a.plotly_chart(mini_forma(v["forma"], col), use_container_width=True)
        signo = "subió" if v["ret"] > 0 else "bajó"
        b.markdown(
            f"**{v['ticker']}** · {v['fecha'].date()} · similitud **{v['similitud']:.0%}**  \n"
            f"<span style='color:{GRIS}'>{v['figura']}</span>  \n"
            f"*Por qué se parece:* {v['porque']}  \n"
            f"*Después:* <span style='color:{col};font-weight:600'>{signo} {v['ret']*100:+.2f}%</span> "
            f"en {horizonte} velas (recorrido +{v['mfe']:.1f}/−{abs(v['mae']):.1f} ATR)",
            unsafe_allow_html=True)


# sectores donde el sistema funciona mejor (del desglose walk-forward: ciclicos
# y tendenciales; los defensivos rinden peor). Se resaltan en la tabla.
SECTORES_FUERTES = {"financiero", "etf_indice", "tecnologia", "industrial",
                    "materiales", "energia"}


def tabla_comparativa(resultados):
    filas = []
    for r in resultados:
        if "error" in r:
            continue
        sec = r.get("sector", "desconocido")
        fuerte = sec in SECTORES_FUERTES
        filas.append({
            "Ticker": r["ticker"],
            "Sector": ("✓ " if fuerte else "") + sec,
            "Sesgo": "▲ sube" if r["prob_sube"] > 0.5 else "▼ baja",
            "Prob. subida": r["prob_sube"],
            "Ret. esperado": r["ret_esperado"],
            "Confianza": r["confianza"],
            "% vecinos ↑": r["pct_vecinos_suben"],
            "Fiable": "no" if r["abstencion"] else "sí",
        })
    if not filas:
        return None
    df = pd.DataFrame(filas).sort_values("Confianza", ascending=False).reset_index(drop=True)
    return df


# etiquetas legibles de cada familia de indicadores (para el "por que")
GRUPO_ETQ = {
    "forma": "Forma del gráfico", "forma_volumen": "Forma del volumen",
    "tf_semanal": "Gráfico semanal", "tf_mensual": "Gráfico mensual",
    "osciladores": "Osciladores (RSI/MACD)", "momento": "Momento / tendencia",
    "volumen": "Volumen", "volatilidad": "Volatilidad", "contexto": "Contexto de mercado",
    "noticias": "Noticias", "fundamentales": "Fundamentales", "macro": "Macro",
    "institucional": "Institucional (13F)", "factores": "Factores (value/quality)",
    "intradia": "Intradía (VWAP/RVOL)",
}


def leer_indicadores(feat, nombres, intra):
    """Convierte el vector de indicadores del caso actual en lecturas legibles.

    Devuelve lista de dicts {label, valor, lectura, tono} donde tono in
    {alza, baja, neutro} para colorear. Es la EVIDENCIA cruda tras la prediccion.
    """
    def v(n):
        return float(feat[nombres.index(n)]) if n in nombres else None
    out = []

    def add(label, valor_txt, lectura, tono):
        out.append({"label": label, "valor": valor_txt, "lectura": lectura, "tono": tono})

    rsi = v("rsi14")
    if rsi is not None:
        t = "baja" if rsi > 70 else ("alza" if rsi < 30 else "neutro")
        lec = "sobrecompra" if rsi > 70 else ("sobreventa" if rsi < 30 else "neutral")
        add("RSI(14)", f"{rsi:.0f}", lec, t)
    rsi2 = v("rsi2")
    if rsi2 is not None:
        t = "baja" if rsi2 > 90 else ("alza" if rsi2 < 10 else "neutro")
        add("RSI(2) corto plazo", f"{rsi2:.0f}",
            "muy sobrecomprado" if rsi2 > 90 else ("muy sobrevendido" if rsi2 < 10 else "neutral"), t)
    mh = v("macd_hist_atr")
    if mh is not None:
        add("MACD (histograma)", f"{mh:+.2f}",
            "momento alcista" if mh > 0 else "momento bajista", "alza" if mh > 0 else "baja")
    adx = v("adx14")
    if adx is not None:
        add("ADX(14)", f"{adx:.0f}",
            "tendencia fuerte" if adx > 25 else ("sin tendencia" if adx < 20 else "tendencia débil"),
            "neutro")
    p50 = v("pos_sma50")
    if p50 is not None:
        add("vs. media 50", f"{p50*100:+.1f}%",
            "por encima" if p50 > 0 else "por debajo", "alza" if p50 > 0 else "baja")
    p200 = v("pos_sma200")
    if p200 is not None:
        add("vs. media 200", f"{p200*100:+.1f}%",
            "tendencia de fondo alcista" if p200 > 0 else "tendencia de fondo bajista",
            "alza" if p200 > 0 else "baja")
    mom = v("mom_12_1")
    if mom is not None:
        add("Momento 12-1", f"{mom*100:+.1f}%",
            "momento positivo" if mom > 0 else "momento negativo", "alza" if mom > 0 else "baja")
    bb = v("boll_b20")
    if bb is not None:
        t = "baja" if bb > 1 else ("alza" if bb < 0 else "neutro")
        lec = "sobre banda superior" if bb > 1 else ("bajo banda inferior" if bb < 0 else "dentro de bandas")
        add("Bollinger %b", f"{bb:.2f}", lec, t)
    atr = v("atr_pct")
    if atr is not None:
        add("Volatilidad (ATR)", f"{atr*100:.2f}%", "del precio por vela", "neutro")
    zv = v("z_volumen20")
    if zv is not None and abs(zv) > 1e-6:
        add("Volumen (z-score)", f"{zv:+.1f}",
            "volumen alto" if zv > 1 else ("volumen bajo" if zv < -1 else "normal"), "neutro")
    reg = v("mkt_sobre_sma200")
    if reg is not None:
        add("Régimen de mercado", "alcista" if reg > 0 else "bajista",
            "el índice sobre su media 200" if reg > 0 else "el índice bajo su media 200",
            "alza" if reg > 0 else "baja")
    fr = v("fuerza_rel_indice_60")
    if fr is not None and abs(fr) > 1e-6:
        add("Fuerza relativa (índice)", f"{fr*100:+.1f}%",
            "más fuerte que el mercado" if fr > 0 else "más débil que el mercado",
            "alza" if fr > 0 else "baja")
    if intra:
        dv = v("intra_dist_vwap")
        if dv is not None and abs(dv) > 1e-9:
            add("Distancia al VWAP", f"{dv*100:+.2f}%",
                "sobre el VWAP de la sesión" if dv > 0 else "bajo el VWAP de la sesión",
                "alza" if dv > 0 else "baja")
        rv = v("intra_rvol")
        if rv is not None and abs(rv) > 1e-9:
            import math
            rvol = math.exp(rv)
            add("RVOL (volumen relativo)", f"{rvol:.2f}×",
                "volumen anormal (stock in play)" if rvol > 1.2 else "volumen normal",
                "alza" if rvol > 1.2 else "neutro")
    return out


# ============================================================= cartera demo (web)
@st.cache_data(ttl=20, show_spinner=False)
def _estado_cuenta():
    from fuentes import alpaca_trade as AT
    import requests
    cta = AT.cuenta()
    pos = AT.posiciones()
    ordenes = AT.ordenes_abiertas()
    try:
        abierto = requests.get("https://paper-api.alpaca.markets/v2/clock",
                               headers=AT._headers(), timeout=15).json().get("is_open")
    except Exception:
        abierto = None
    return cta, pos, {o["symbol"] for o in ordenes}, abierto


@st.cache_data(ttl=120, show_spinner=False)
def _reanalisis_intradia():
    from gestionar import revisar_intradia
    return revisar_intradia()


@st.cache_data(ttl=120, show_spinner=False)
def _oportunidades():
    from gestionar import escanear, escaneo_diario
    from descubrir import candidatos_nuevos
    intra = escanear("15Min")
    nuevos = candidatos_nuevos(n=6)
    diario = escaneo_diario(n=8)
    return intra, nuevos, diario


@st.cache_data(ttl=300, show_spinner=False)
def _noticias(ticker):
    from fuentes import noticias_vivo
    return noticias_vivo.noticias(ticker, n=5)


def dashboard_cartera():
    from fuentes import alpaca_trade as AT
    st.subheader("💼 Cartera demo — Alpaca paper (en vivo)")
    try:
        cta, pos, sim_bracket, abierto = _estado_cuenta()
    except Exception as e:
        st.error(f"No pude conectar con la cuenta paper de Alpaca: {str(e)[:120]}")
        st.info("Comprueba que `datos/.alpaca_key` tiene las claves paper (empiezan por PK).")
        return
    eq = float(cta["equity"]); cash = float(cta["cash"])
    pl_dia = eq - float(cta.get("last_equity", eq))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Equity", f"${eq:,.0f}", f"{pl_dia:+,.0f} hoy")
    c2.metric("Cash", f"${cash:,.0f}")
    c3.metric("Invertido", f"${float(cta.get('long_market_value', 0)):,.0f}")
    c4.metric("Mercado", "🟢 ABIERTO" if abierto else "🔴 cerrado")

    # --- posiciones
    filas = [{"Ticker": t, "Libro": "Intradía" if t in sim_bracket else "Swing",
              "Acciones": int(v["qty"]), "Valor": v["valor"], "P&L": v["pl"]}
             for t, v in pos.items()]
    if filas:
        dfp = pd.DataFrame(filas).sort_values(["Libro", "P&L"], ascending=[True, False])
        st.markdown(f"**Posiciones ({len(filas)})** · Intradía: "
                    f"{sum(1 for f in filas if f['Libro']=='Intradía')} · "
                    f"Swing: {sum(1 for f in filas if f['Libro']=='Swing')}")
        st.dataframe(dfp.style.format({"Valor": "${:,.0f}", "P&L": "${:+,.0f}"})
                     .background_gradient(subset=["P&L"], cmap="RdYlGn"),
                     use_container_width=True, hide_index=True, height=240)
    else:
        st.info("Sin posiciones abiertas.")

    # --- controles para maniobrar
    st.markdown("**Maniobrar**")
    b1, b2, b3 = st.columns(3)
    if b1.button("🔄 Gestión activa ahora", use_container_width=True):
        with st.spinner("Re-analizando intradía y rotando…"):
            from gestionar import main as gest
            import io
            import contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                gest(gestionar=True, rotar=True)
            _estado_cuenta.clear(); _reanalisis_intradia.clear()
        st.code(buf.getvalue() or "sin salida")
    if b2.button("🧹 Cerrar intradía", use_container_width=True):
        r = AT.cerrar_intradia(confirmar=True)
        _estado_cuenta.clear()
        st.success(f"Intradía cerrado: {r.get('cerradas', r)}")
    cerrar_t = b3.selectbox("Cerrar posición", ["—"] + sorted(pos), key="cerrar_sel")
    if b3.button("Cerrar la seleccionada", use_container_width=True) and cerrar_t != "—":
        r = AT.cerrar_posiciones([cerrar_t], confirmar=True)
        _estado_cuenta.clear()
        st.success(f"Cerrada {cerrar_t}: {r.get('cerradas', r)}")

    # --- re-analisis intradia (por que se mantiene o se cierra)
    st.divider()
    st.markdown("**Re-análisis intradía** (mantener bajada orgánica vs cerrar reversión confirmada)")
    if st.button("Analizar posiciones intradía ahora"):
        _reanalisis_intradia.clear()
    try:
        rev = _reanalisis_intradia()
    except Exception as e:
        rev = []
        st.caption(f"(no se pudo analizar: {str(e)[:80]})")
    if rev:
        dr = pd.DataFrame([{"Ticker": d["ticker"], "P&L": d["pl"],
                            "dist VWAP %": d["dist_vwap"] * 100, "pend %": d["pend"] * 100,
                            "bajo VWAP %": d["bajo_vwap"] * 100,
                            "Decisión": "🔴 CERRAR" if d["cerrar"] else "🟢 mantener",
                            "Motivo": d["motivo"]} for d in rev])
        st.dataframe(dr.style.format({"P&L": "${:+,.0f}", "dist VWAP %": "{:+.2f}",
                                      "pend %": "{:+.3f}", "bajo VWAP %": "{:.0f}"}),
                     use_container_width=True, hide_index=True)
    else:
        st.caption("Sin posiciones intradía con bracket ahora mismo.")

    # --- oportunidades + noticias
    st.divider()
    st.markdown("**Oportunidades** (intradía, tickers nuevos y momentum diario)")
    if st.button("📡 Buscar oportunidades + noticias"):
        _oportunidades.clear()
        with st.spinner("Escaneando universo ampliado, tickers nuevos y noticias…"):
            intra, nuevos, diario = _oportunidades()
        o1, o2 = st.columns(2)
        with o1:
            st.caption("Intradía (15m, universo ampliado)")
            if len(intra):
                st.dataframe(intra[["ticker", "dist_vwap", "rvol", "ret_sesion"]]
                             .style.format({"dist_vwap": "{:+.3f}", "rvol": "{:.2f}",
                                            "ret_sesion": "{:+.3f}"}),
                             use_container_width=True, hide_index=True)
            else:
                st.caption("ninguna ahora")
            st.caption("Tickers NUEVOS (no vistos) + noticias")
            if len(nuevos):
                cols = [c for c in ["ticker", "rvol", "ret_sesion", "tono", "sent_noticias"] if c in nuevos]
                st.dataframe(nuevos[cols], use_container_width=True, hide_index=True)
            else:
                st.caption("ninguno ahora")
        with o2:
            st.caption("Top momentum diario (oportunidades swing)")
            dd = pd.DataFrame({"ticker": diario.index, "momento %": diario.to_numpy() * 100})
            dd["en cartera"] = dd["ticker"].isin(pos).map({True: "sí", False: ""})
            st.dataframe(dd.style.format({"momento %": "{:+.1f}"}),
                         use_container_width=True, hide_index=True)

    # --- noticias de la cartera
    st.divider()
    st.markdown("**Noticias de la cartera**")
    tsel = st.selectbox("Ver titulares de", ["—"] + sorted(pos), key="news_sel")
    if tsel != "—":
        nv = _noticias(tsel)
        st.caption(f"Tono medio: {nv['etiqueta']} ({nv['sentimiento']:+.2f}) · {nv['n']} titulares")
        for h in nv["titulares"]:
            col = VER if h["punt"] > 0.1 else (ROJO if h["punt"] < -0.1 else GRIS)
            st.markdown(f"<span style='color:{col}'>[{h['punt']:+.2f}]</span> {h['titular']} "
                        f"<span style='color:{GRIS};font-size:0.8em'>· {h['fecha']} · {h['fuente']}</span>",
                        unsafe_allow_html=True)


# ======================================================================== app
def _secrets_a_env():
    """Pasa las claves de los Secrets del hosting (Streamlit Cloud, etc.) a variables
    de entorno, para que los conectores (Alpaca, FRED) funcionen desplegados."""
    import os
    try:
        for k in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "FRED_API_KEY"):
            if k in st.secrets and not os.environ.get(k):
                os.environ[k] = str(st.secrets[k])
    except Exception:
        pass


def main():
    st.set_page_config(page_title="Buscador de patrones", page_icon="📈", layout="wide")
    _secrets_a_env()
    st.markdown(f"<h1 style='margin-bottom:0'>📈 Buscador de patrones de trading</h1>"
                f"<p style='color:{GRIS};margin-top:2px'>Trabajo de Fin de Grado · CUNEF Universidad"
                f"  ·  sistema de aprendizaje de patrones con validación walk-forward</p>",
                unsafe_allow_html=True)
    st.caption("Metes varios tickers → encuentra momentos históricos parecidos, explica por qué, "
               "y predice. Herramienta de estudio, no es recomendación de inversión.")

    vista = st.sidebar.radio("Vista", ["🔎 Buscador de patrones", "💼 Cartera demo (en vivo)"])
    if vista.startswith("💼"):
        dashboard_cartera()
        return

    with st.sidebar:
        st.header("Configuración")
        intervalo = st.selectbox("Temporalidad", list(PRESETS), index=0,
                                 format_func=lambda k: PRESETS[k]["etq"])
        P = PRESETS[intervalo]
        try:
            disponibles = sorted(cargar_universo(intervalo)[0])
        except Exception as e:
            disponibles = []
            st.error(f"No pude cargar los datos de {P['etq']}: {e}")
        if not disponibles:
            cmd = ("python bot.py descargar --intervalos 1d" if intervalo == "1d"
                   else f"python bot.py descargar-intraday --intervalo {intervalo}")
            st.warning("No hay datos para esta temporalidad. Descárgalos primero:\n\n"
                       f"```\n{cmd}\n```")
            st.stop()
        defecto = [t for t in ["NVDA", "AAPL", "TSLA"] if t in disponibles][:3] or disponibles[:3]
        seleccion = st.multiselect(f"Tickers ({len(disponibles)} disponibles)", disponibles,
                                   default=defecto)
        extra_txt = ""
        if intervalo == "1d":
            extra_txt = st.text_input("Otros tickers (Yahoo)", "",
                                      placeholder="p.ej. INTC, SHOP",
                                      help="Analiza valores fuera del universo; se descargan "
                                           "de Yahoo al vuelo (solo diario).")
        extra = [t.strip().upper() for t in extra_txt.replace(";", ",").split(",") if t.strip()]
        seleccion = list(dict.fromkeys(seleccion + extra))   # une y quita duplicados
        n_similares = st.slider("Patrones parecidos a mostrar", 3, 12, 6)
        with st.expander("Parámetros avanzados"):
            largo = st.slider("Velas de la ventana (forma)", 20, 60, P["largo"])
            horizonte = st.slider("Horizonte (velas a predecir)", 3, 20, P["horizonte"])
            anos = st.slider("Años de histórico", 2, 12, P["anos"])
            paso = P["paso"]
        lanzar = st.button("Analizar", type="primary", use_container_width=True)

    st.markdown(f"**Temporalidad:** {P['etq']}  ·  **Universo:** {len(disponibles)} valores  ·  "
                f"ventana {largo} velas, horizonte {horizonte}")

    if not seleccion:
        st.info("Elige al menos un ticker en la barra lateral y pulsa **Analizar**.")
        st.stop()

    if not (lanzar or st.session_state.get("auto", True)):
        return
    st.session_state["auto"] = False
    kmem = max(n_similares * 4, 30)
    with st.spinner(f"Construyendo memoria de patrones ({P['etq']})…"):
        try:
            construir_memoria(intervalo, anos, largo, horizonte, paso, kmem)
        except Exception as e:
            st.error(f"Error construyendo la memoria: {e}")
            st.stop()
    resultados = []
    barra = st.progress(0.0, text="Analizando tickers…")
    for i, t in enumerate(seleccion, 1):
        try:
            resultados.append(analizar(intervalo, anos, largo, horizonte, paso, kmem, t, n_similares))
        except Exception as e:
            resultados.append({"error": str(e), "ticker": t})
        barra.progress(i / len(seleccion), text=f"Analizando {t} ({i}/{len(seleccion)})")
    barra.empty()

    if len(seleccion) > 1:
        st.header("Comparación entre tickers")
        df = tabla_comparativa(resultados)
        if df is not None:
            st.dataframe(
                df.style.format({"Prob. subida": "{:.0%}", "Ret. esperado": "{:+.2%}",
                                 "Confianza": "{:.2f}", "% vecinos ↑": "{:.0%}"})
                .background_gradient(subset=["Confianza"], cmap="Greens"),
                use_container_width=True, hide_index=True)
            mejor = df.iloc[0]
            st.success(f"Señal más fiable ahora: **{mejor['Ticker']}** "
                       f"({mejor['Sesgo']}, confianza {mejor['Confianza']:.2f}).")
            st.caption("El ✓ marca sectores donde el sistema funciona mejor "
                       "(cíclicos/tendenciales); los defensivos rinden peor.")
            st.download_button("⬇ Descargar comparación (CSV)", df.to_csv(index=False),
                               file_name="comparacion_tickers.csv", mime="text/csv")
        st.divider()

    for r in resultados:
        if "error" in r:
            st.error(f"{r.get('ticker','?')}: {r['error']}")
            continue
        with st.container(border=True):
            vista_detalle(r, horizonte)

    st.divider()
    st.caption("Herramienta de estudio con fines académicos (TFG, CUNEF Universidad). "
               "No es recomendación de inversión. La confianza mostrada proviene de la "
               "memoria de casos; el meta-modelo calibrado se usa en la validación.")


if __name__ == "__main__":
    main()
