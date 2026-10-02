"""Cache de 'inteligencia' del modelo para el terminal: importancia de features del
meta-labeling + edge matrix (confianza del meta vs resultado real). Es LENTO (corre el
walk-forward del meta), asi que se cachea en datos/modelo_cache.json y el dashboard lo
lee; se regenera a diario, no en cada ciclo.

  python modelo.py            # regenera datos/modelo_cache.json
Usa el primario GBM (rapido en lote) para no tardar como el kNN.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ANOS = float(sys.argv[1]) if len(sys.argv) > 1 else 3
CACHE = "datos/modelo_cache.json"


def generar(anos=ANOS):
    from fuentes.descarga import cargar_mercado
    from patrones import ventanas as mod_v, contexto as mod_ctx, evaluacion

    pre, meta = cargar_mercado("datos/mercado", "1d")
    msi = meta.set_index("ticker")
    tickers = [t for t in pre if ("clase" not in meta.columns) or msi.loc[t, "clase"] in ("accion", "fondo")]
    corte = pd.Timestamp.now() - pd.Timedelta(days=int(anos * 365))
    pre = {t: pre[t][pre[t].index >= corte] for t in tickers}
    pre = {t: d for t, d in pre.items() if len(d) > 320}
    ctx = mod_ctx.Contexto(pre, meta)
    cfg = mod_v.Config()
    b = mod_v.construir(pre, ctx, None, cfg, verbose=False, enriquecer=None)
    res, imp = evaluacion.evaluar_meta(b, n_pliegues=3, primario="gbm", verbose=False)

    # importancia de features (red neural-shell). Nombres legibles.
    NOMBRE = {"prob_cruda": "P. memoria", "conviccion": "Convicción", "confianza": "Confianza",
              "dist_media": "Distancia vecinos", "disp_ret_vecinos": "Dispersión vecinos",
              "frac_vecinos_suben": "Acuerdo vecinos", "mfe_esp": "MFE esp.", "mae_esp": "MAE esp.",
              "ratio_mfe_mae": "Ratio MFE/MAE", "prob_gbm": "P. GBM", "acuerdo_modelos": "Acuerdo modelos",
              "conviccion_gbm": "Convicción GBM", "disp_grupos": "Disp. grupos",
              "frac_grupos_acuerdo": "Acuerdo grupos", "atr_pct": "Volatilidad (ATR)",
              "vol_ratio": "Régimen vol", "ret_estand": "Retorno estand.", "pendiente_fig": "Pendiente figura",
              "rango_ret_vecinos": "Rango vecinos", "n_vecinos": "Nº vecinos"}
    imp_lista = sorted(imp.items(), key=lambda kv: -abs(kv[1]))[:12]
    importancias = [{"feat": NOMBRE.get(k, k.replace("fig_", "fig ").replace("_", " ")),
                     "peso": round(float(v), 4)} for k, v in imp_lista]

    # edge matrix: confianza del meta (x) vs resultado real (y = retorno triple-barrera)
    r = res.dropna(subset=["prob_meta"]).copy()
    if "tb_ret" in r.columns:
        r = r.dropna(subset=["tb_ret"])
    n = len(r)
    muestra = r.sample(min(380, n), random_state=0) if n > 380 else r
    edge = [{"p": round(float(p), 3), "r": round(float(t), 3), "ok": int(a)}
            for p, t, a in zip(muestra["prob_meta"], muestra.get("tb_ret", pd.Series([0] * len(muestra))),
                               muestra["acierto_primario"])]
    # calibracion por deciles: P(acierto real) por tramo de prob del meta
    r["bin"] = (r["prob_meta"] * 10).clip(0, 9).astype(int)
    calib = [{"p": (i + 0.5) / 10, "acierto": round(float(g["acierto_primario"].mean()), 3), "n": int(len(g))}
             for i, g in r.groupby("bin") if len(g) >= 5]
    try:
        from patrones.robustez import sharpe_no_anualizado
        auc = evaluacion._auc(r["acierto_primario"].to_numpy(), r["prob_meta"].to_numpy())
    except Exception:
        auc = float("nan")

    cache = {"generado": pd.Timestamp.now(tz="UTC").isoformat(),
             "n_casos": int(n), "auc": round(float(auc), 4) if np.isfinite(auc) else None,
             "importancias": importancias, "edge": edge, "calibracion": calib}
    os.makedirs("datos", exist_ok=True)
    json.dump(cache, open(CACHE, "w"), ensure_ascii=False, indent=2)
    print(f"modelo_cache: {len(importancias)} features | {len(edge)} puntos edge | "
          f"{len(calib)} tramos calib | AUC {cache['auc']} | {n} casos")
    return cache


if __name__ == "__main__":
    generar()
