"""Indicadores tecnicos en numpy/pandas puro (sin TA-Lib ni dependencias raras).

REGLA DEL PROYECTO: ningun indicador mira al futuro. Todos usan ventanas que
terminan en la barra actual, asi que el valor en la fecha t solo depende de
datos de t y anteriores. Esto es lo que evita que los resultados de la
evaluacion salgan magicamente buenos por fuga de informacion.
"""
import numpy as np
import pandas as pd


def _s(x):
    return x if isinstance(x, pd.Series) else pd.Series(x)


def sma(close, n):
    return _s(close).rolling(n, min_periods=n).mean()


def ema(close, n):
    return _s(close).ewm(span=n, adjust=False, min_periods=n).mean()


def wilder(x, n):
    """Suavizado de Wilder, el que usan de verdad RSI, ATR y ADX."""
    return _s(x).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(close, n=14):
    d = _s(close).diff()
    subidas = wilder(d.clip(lower=0), n)
    bajadas = wilder(-d.clip(upper=0), n)
    rs = subidas / bajadas.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # Si no ha habido ninguna bajada en la ventana, el RSI es 100 por definicion.
    return out.where(bajadas.notna() & (bajadas != 0), out.where(bajadas != 0, 100.0))


def true_range(high, low, close):
    h, l, c = _s(high), _s(low), _s(close)
    cierre_previo = c.shift(1)
    return pd.concat([h - l, (h - cierre_previo).abs(), (l - cierre_previo).abs()], axis=1).max(axis=1)


def atr(high, low, close, n=14):
    return wilder(true_range(high, low, close), n)


def adx(high, low, close, n=14):
    """Fuerza de la tendencia. Alto = tendencia clara, bajo = lateral."""
    h, l = _s(high), _s(low)
    up, dn = h.diff(), -l.diff()
    mas = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=h.index)
    menos = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=h.index)
    tr = wilder(true_range(high, low, close), n)
    di_mas = 100 * wilder(mas, n) / tr.replace(0.0, np.nan)
    di_menos = 100 * wilder(menos, n) / tr.replace(0.0, np.nan)
    dx = 100 * (di_mas - di_menos).abs() / (di_mas + di_menos).replace(0.0, np.nan)
    return wilder(dx, n), di_mas, di_menos


def macd(close, rapida=12, lenta=26, senal=9):
    linea = ema(close, rapida) - ema(close, lenta)
    sig = linea.ewm(span=senal, adjust=False, min_periods=senal).mean()
    return linea, sig, linea - sig


def bollinger_b(close, n=20, k=2.0):
    """%B: 0 = banda inferior, 1 = banda superior. Donde esta el precio en su rango."""
    c = _s(close)
    m = c.rolling(n, min_periods=n).mean()
    s = c.rolling(n, min_periods=n).std(ddof=0)
    inf, sup = m - k * s, m + k * s
    return (c - inf) / (sup - inf).replace(0.0, np.nan)


def obv(close, volume):
    """On Balance Volume: acumula volumen con el signo del dia."""
    c, v = _s(close), _s(volume)
    return (np.sign(c.diff().fillna(0.0)) * v).cumsum()


def volatilidad_realizada(close, n=20, periodos_ano=252):
    r = np.log(_s(close)).diff()
    return r.rolling(n, min_periods=n).std(ddof=0) * np.sqrt(periodos_ano)


def z_score(x, n):
    s = _s(x)
    m = s.rolling(n, min_periods=n).mean()
    d = s.rolling(n, min_periods=n).std(ddof=0)
    return (s - m) / d.replace(0.0, np.nan)


def distancia_extremo(close, n=252):
    """Cuanto falta para el maximo y el minimo de la ventana, en tanto por uno."""
    c = _s(close)
    mx = c.rolling(n, min_periods=max(20, n // 5)).max()
    mn = c.rolling(n, min_periods=max(20, n // 5)).min()
    return c / mx - 1.0, c / mn - 1.0


def pendiente(x, n):
    """Pendiente de la recta de regresion de los ultimos n puntos, normalizada."""
    s = _s(x)
    t = np.arange(n, dtype=float)
    t = t - t.mean()
    den = (t ** 2).sum()

    def _fit(w):
        return float(np.dot(w - w.mean(), t) / den)

    cruda = s.rolling(n, min_periods=n).apply(_fit, raw=True)
    return cruda / s.rolling(n, min_periods=n).mean().abs().replace(0.0, np.nan)


def momentum_12_1(close):
    """Momentum 12-1 (Jegadeesh-Titman): retorno de t-252 a t-21 (12 meses saltando
    el ultimo, para evitar la reversion a corto plazo). Solo usa el pasado."""
    c = _s(close)
    return c.shift(21) / c.shift(252) - 1.0
