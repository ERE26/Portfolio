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
def analizar(intervalo, anos, largo, horizonte, paso, k, ticker, n_similares):
    """Analiza un ticker: tramo actual, vecinos, explicacion y prediccion.

    Devuelve un dict serializable para cachear por (parametros, ticker)."""
    B = construir_memoria(intervalo, anos, largo, horizonte, paso, k)
    mem, cfg, precios, ctx = B["mem"], B["cfg"], B["precios"], B["ctx"]
    if ticker not in precios:
        return {"error": f"{ticker} no esta en el universo cargado."}
    vivo = mod_v.caso_vivo(precios, ctx, None, cfg, ticker)
    pred = mem.predecir(vivo, 0, excluir_ticker=ticker)
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
    return {
        "ticker": ticker, "fecha": str(pred.fecha), "figura": desc_obj,
        "prob_sube": float(pred.prob_sube), "ret_esperado": float(pred.ret_esperado),
        "confianza": float(pred.confianza), "abstencion": bool(pred.abstencion),
        "atr_pct": atr, "forma_obj": forma_obj,
        "camino": (np.asarray(pred.camino) * atr * 100).tolist(),
        "camino_p10": (np.asarray(pred.camino_p10) * atr * 100).tolist(),
        "camino_p90": (np.asarray(pred.camino_p90) * atr * 100).tolist(),
        "pct_vecinos_suben": float((vec["ret"] > 0).mean()) if len(vec) else float("nan"),
        "n_vecinos": int(len(vec)), "vecinos": filas, "ohlc": ohlc,
        "probs_grupo": {k_: float(v_) for k_, v_ in pred.probs_grupo.items()},
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


def vista_detalle(r, horizonte):
    st.subheader(f"{r['ticker']}  ·  {r['fecha'][:10]}")
    st.caption(f"Figura detectada: **{r['figura']}**")
    tarjeta_prediccion(r)
    g1, g2 = st.columns([1.1, 1])
    g1.plotly_chart(grafico_velas_proyeccion(r, horizonte), use_container_width=True)
    g2.plotly_chart(grafico_parecido(r), use_container_width=True)
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


def tabla_comparativa(resultados):
    filas = []
    for r in resultados:
        if "error" in r:
            continue
        filas.append({
            "Ticker": r["ticker"],
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


# ======================================================================== app
def main():
    st.set_page_config(page_title="Buscador de patrones", page_icon="📈", layout="wide")
    st.title("📈 Buscador de patrones de trading")
    st.caption("Metes varios tickers → encuentra momentos históricos parecidos, explica por qué, "
               "y predice. Herramienta de estudio, no es recomendación de inversión.")

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
        st.divider()

    for r in resultados:
        if "error" in r:
            st.error(f"{r.get('ticker','?')}: {r['error']}")
            continue
        with st.container(border=True):
            vista_detalle(r, horizonte)


if __name__ == "__main__":
    main()
