"""Các phép tính chỉ số kỹ thuật & thống kê trên dữ liệu giá OHLCV.

Không phụ thuộc vnstock hay Streamlit - chỉ nhận/trả về pandas.DataFrame,
nên có thể test độc lập với dữ liệu bất kỳ.

Input mong đợi: DataFrame có cột time, open, high, low, close, volume,
đã sắp xếp tăng dần theo thời gian.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def add_moving_averages(df: pd.DataFrame, windows: tuple[int, ...] = (20, 50)) -> pd.DataFrame:
    """Thêm cột sma_{n}: trung bình động đơn giản của giá đóng cửa."""
    out = df.copy()
    for w in windows:
        out[f"sma_{w}"] = out["close"].rolling(window=w, min_periods=w).mean()
    return out


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Thêm dải Bollinger: bb_mid (SMA), bb_upper, bb_lower."""
    out = df.copy()
    mid = out["close"].rolling(window=window, min_periods=window).mean()
    std = out["close"].rolling(window=window, min_periods=window).std()
    out["bb_mid"] = mid
    out["bb_upper"] = mid + num_std * std
    out["bb_lower"] = mid - num_std * std
    return out


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Thêm cột rsi: chỉ số sức mạnh tương đối (0-100).

    RSI > 70 thường được coi là vùng quá mua, RSI < 30 là vùng quá bán -
    đây chỉ là kinh nghiệm tham khảo, không phải quy tắc tuyệt đối.
    """
    out = df.copy()
    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi"] = 100 - (100 / (1 + rs))
    out.loc[avg_loss == 0, "rsi"] = 100
    return out


def compute_macd(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """Thêm cột macd, macd_signal, macd_hist (đường trung bình động hội tụ phân kỳ)."""
    out = df.copy()
    ema_fast = out["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = out["close"].ewm(span=slow, adjust=False).mean()
    out["macd"] = ema_fast - ema_slow
    out["macd_signal"] = out["macd"].ewm(span=signal, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    return out


def compute_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm cột daily_return: tỷ suất sinh lời theo ngày (%) và cumulative_return (%)."""
    out = df.copy()
    out["daily_return"] = out["close"].pct_change() * 100
    out["cumulative_return"] = (out["close"] / out["close"].iloc[0] - 1) * 100 if len(out) else np.nan
    return out


def compute_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Thêm cột volatility: độ lệch chuẩn của tỷ suất sinh lời ngày, quy đổi theo năm (%)."""
    out = df.copy()
    daily_ret = out["close"].pct_change()
    out["volatility"] = daily_ret.rolling(window=window, min_periods=window).std() * np.sqrt(
        TRADING_DAYS_PER_YEAR
    ) * 100
    return out


def compute_max_drawdown(df: pd.DataFrame) -> float:
    """Mức sụt giảm tối đa (%) từ đỉnh gần nhất trong toàn bộ giai đoạn."""
    if df.empty:
        return float("nan")
    running_max = df["close"].cummax()
    drawdown = (df["close"] / running_max - 1) * 100
    return float(drawdown.min())


def enrich(df: pd.DataFrame, ma_windows: tuple[int, ...] = (20, 50)) -> pd.DataFrame:
    """Áp dụng toàn bộ các chỉ số ở trên vào một DataFrame giá, trả về bản đã bổ sung cột."""
    out = add_moving_averages(df, ma_windows)
    out = add_bollinger_bands(out)
    out = compute_rsi(out)
    out = compute_macd(out)
    out = compute_daily_returns(out)
    out = compute_volatility(out)
    return out


# ---------------------------------------------------------------------------
# Phân tích kỹ thuật chuyên sâu: OBV, thống kê volume, trạng thái đa khung
# ---------------------------------------------------------------------------
def compute_obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume: cộng dồn khối lượng theo hướng giá đóng cửa.

    OBV tăng khi dòng tiền vào (phiên tăng có khối lượng lớn), giảm khi dòng
    tiền rút ra — dùng để xác nhận xu hướng giá bằng khối lượng.
    """
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum()


def volume_stats(df: pd.DataFrame, window: int = 20) -> dict:
    """Thống kê khối lượng phục vụ nhận định dòng tiền (trên `window` phiên cuối).

    - vol_ratio      : KL phiên gần nhất / trung bình `window` phiên
    - accum_days     : số phiên TĂNG giá kèm KL > trung bình (phiên tích lũy)
    - distrib_days   : số phiên GIẢM giá kèm KL > trung bình (phiên phân phối)
    - obv_rising     : OBV đang nằm trên trung bình 20 phiên của chính nó
    """
    if len(df) < window + 1:
        return {}
    vol_ma = df["volume"].rolling(window).mean()
    ret = df["close"].diff()
    high_vol = df["volume"] > vol_ma
    tail = df.index[-window:]
    obv = compute_obv(df)
    return {
        "vol_ratio": float(df["volume"].iloc[-1] / vol_ma.iloc[-1]) if vol_ma.iloc[-1] else float("nan"),
        "avg_volume": float(vol_ma.iloc[-1]),
        "accum_days": int(((ret > 0) & high_vol).loc[tail].sum()),
        "distrib_days": int(((ret < 0) & high_vol).loc[tail].sum()),
        "obv_rising": bool(obv.iloc[-1] > obv.rolling(window).mean().iloc[-1]),
    }


def frame_status(df: pd.DataFrame, rsi_period: int = 14) -> dict:
    """Trạng thái kỹ thuật của MỘT khung thời gian (ngày/tuần/tháng).

    Trả về dict để dashboard hiển thị thẻ trạng thái đa khung:
    - trend    : 'up' (giá > MA10 > MA20) / 'down' (giá < MA10 < MA20) / 'side'
    - rsi      : giá trị RSI hiện tại; rsi_state: 'overbought'/'oversold'/'neutral'
    - macd_ok  : MACD đang nằm trên đường signal
    - vol_ratio: KL thanh gần nhất so với trung bình 20 thanh
    """
    if len(df) < 21:
        return {}
    close = df["close"]
    ma10 = close.rolling(10).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    last = float(close.iloc[-1])

    if last > ma10 > ma20:
        trend = "up"
    elif last < ma10 < ma20:
        trend = "down"
    else:
        trend = "side"

    rsi_now = float(compute_rsi(df, period=rsi_period)["rsi"].iloc[-1])
    if rsi_now >= 70:
        rsi_state = "overbought"
    elif rsi_now <= 30:
        rsi_state = "oversold"
    else:
        rsi_state = "neutral"

    macd_df = compute_macd(df)
    vol_ma = df["volume"].rolling(20).mean().iloc[-1]

    return {
        "trend": trend,
        "rsi": rsi_now,
        "rsi_state": rsi_state,
        "macd_ok": bool(macd_df["macd"].iloc[-1] > macd_df["macd_signal"].iloc[-1]),
        "vol_ratio": float(df["volume"].iloc[-1] / vol_ma) if vol_ma else float("nan"),
        "last_close": last,
    }


# ---------------------------------------------------------------------------
# Bộ lọc cổ phiếu (tích hợp từ vn-stock-screener_1, giữ nguyên 6 tiêu chí gốc)
# ---------------------------------------------------------------------------
def volume_ratio(volume: pd.Series, window: int = 20) -> float:
    """Khối lượng phiên gần nhất / trung bình `window` phiên."""
    avg = volume.rolling(window).mean().iloc[-1]
    if not avg or pd.isna(avg):
        return float("nan")
    return float(volume.iloc[-1] / avg)


def distance_from_52w_high(close: pd.Series) -> float:
    """% cách đỉnh 52 tuần (0.10 = đang thấp hơn đỉnh 10%)."""
    high = close.tail(252).max()
    return float((high - close.iloc[-1]) / high)


def return_over(close: pd.Series, sessions: int = 63) -> float:
    """% thay đổi giá qua `sessions` phiên (63 phiên ~ 3 tháng)."""
    n = min(sessions, len(close))
    return float(close.iloc[-1] / close.iloc[-n] - 1)


def score_symbol(
    symbol: str,
    df: pd.DataFrame,
    index_ret_3m: float,
    rsi_min: float = 50.0,
    rsi_max: float = 70.0,
    vol_breakout: float = 1.5,
    max_dist_52w: float = 0.15,
) -> dict:
    """Chấm 6 tiêu chí kỹ thuật cho một mã, trả dict phẳng để đưa thẳng vào bảng.

    6 tiêu chí (mỗi tiêu chí 1 điểm):
    - trend_ok    : giá > MA50 > MA200 (xu hướng tăng dài hạn)
    - rsi_ok      : RSI trong vùng [rsi_min, rsi_max] (khỏe nhưng chưa quá mua)
    - macd_ok     : đường MACD nằm trên đường signal (động lượng dương)
    - volume_ok   : khối lượng phiên gần nhất >= vol_breakout x trung bình 20 phiên
    - near_high_ok: giá cách đỉnh 52 tuần không quá max_dist_52w
    - rs_ok       : 3 tháng qua tăng mạnh hơn VN-Index (sức mạnh tương đối)
    """
    close, vol = df["close"], df["volume"]
    last = float(close.iloc[-1])

    ma50 = close.rolling(50).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1]
    rsi_now = float(compute_rsi(df)["rsi"].iloc[-1])
    macd_df = compute_macd(df)
    vratio = volume_ratio(vol)
    dist = distance_from_52w_high(close)
    rs = return_over(close) - index_ret_3m

    checks = {
        "trend_ok": bool(last > ma50 > ma200),
        "rsi_ok": bool(rsi_min <= rsi_now <= rsi_max),
        "macd_ok": bool(macd_df["macd"].iloc[-1] > macd_df["macd_signal"].iloc[-1]),
        "volume_ok": bool(vratio >= vol_breakout),
        "near_high_ok": bool(dist <= max_dist_52w),
        "rs_ok": bool(rs > 0),
    }
    return {
        "symbol": symbol,
        "price": round(last, 2),
        "rsi": round(rsi_now, 1),
        "vol_ratio": round(vratio, 2),
        "dist_52w_high": round(dist, 4),
        "rs_vs_index_3m": round(rs, 4),
        **checks,
        "score": int(sum(checks.values())),
        "max_score": len(checks),
    }


def summary_stats(df: pd.DataFrame) -> dict:
    """Tóm tắt các con số chính của giai đoạn đang xem: giá, biến động, khối lượng..."""
    if df.empty:
        return {}
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    change = last["close"] - prev["close"]
    pct_change = (change / prev["close"] * 100) if prev["close"] else np.nan
    return {
        "last_close": float(last["close"]),
        "change": float(change),
        "pct_change": float(pct_change),
        "period_high": float(df["high"].max()),
        "period_low": float(df["low"].min()),
        "last_volume": float(last["volume"]),
        "avg_volume": float(df["volume"].mean()),
        "mean_daily_return_pct": float(df["close"].pct_change().mean() * 100),
        "annualized_volatility_pct": float(
            df["close"].pct_change().std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
        ),
        "max_drawdown_pct": compute_max_drawdown(df),
    }
