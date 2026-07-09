# -*- coding: utf-8 -*-
"""
Module phân tích KHỐI LƯỢNG (volume) — tích hợp với mtf_vn30.py
Input: DataFrame OHLCV index thời gian, cột: open, high, low, close, volume

4 lớp phân tích:
  1. Volume Ratio  — khối lượng hôm nay so với nền trung bình 20 phiên
  2. OBV           — dòng tiền tích lũy theo hướng giá (phát hiện phân kỳ)
  3. MFI           — "RSI có trọng số volume", đo áp lực mua/bán thật
  4. Xác nhận sự kiện — breakout/breakdown có volume đi kèm hay không
"""

import numpy as np
import pandas as pd


# ---------- 1. Volume Ratio & trạng thái nền khối lượng ----------

def volume_ratio(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Khối lượng / trung bình N phiên. >1.5 = đột biến, <0.6 = cạn kiệt."""
    return df["volume"] / df["volume"].rolling(window).mean()


def volume_state(df: pd.DataFrame) -> dict:
    vr = volume_ratio(df).iloc[-1]
    trend5 = df["volume"].tail(5).mean() / df["volume"].rolling(20).mean().iloc[-1]
    if vr >= 2.0:   label = "Đột biến mạnh"
    elif vr >= 1.5: label = "Đột biến"
    elif vr >= 0.8: label = "Bình thường"
    elif vr >= 0.6: label = "Thấp"
    else:           label = "Cạn kiệt"
    return {"vol_ratio": round(float(vr), 2),
            "vol_trend_5d": round(float(trend5), 2),  # >1: nền vol đang dày lên
            "vol_state": label}


# ---------- 2. OBV (On-Balance Volume) ----------

def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum()


def obv_divergence(df: pd.DataFrame, lookback: int = 40) -> str:
    """So sánh xu hướng giá và xu hướng OBV trong N phiên gần nhất.
    Trả về: 'Đồng thuận' / 'Phân kỳ dương' (giá giảm, OBV tăng = gom hàng)
            / 'Phân kỳ âm' (giá tăng, OBV giảm = phân phối)."""
    if len(df) < lookback:
        return "Thiếu dữ liệu"
    seg = df.tail(lookback)
    o = obv(df).tail(lookback)
    # Hồi quy tuyến tính đơn giản lấy dấu độ dốc
    x = np.arange(lookback)
    price_slope = np.polyfit(x, seg["close"].values, 1)[0]
    obv_slope = np.polyfit(x, o.values, 1)[0]
    if price_slope <= 0 and obv_slope > 0:
        return "Phân kỳ dương (gom hàng?)"
    if price_slope >= 0 and obv_slope < 0:
        return "Phân kỳ âm (phân phối?)"
    return "Đồng thuận"


# ---------- 3. MFI (Money Flow Index, 14) ----------

def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3          # typical price
    mf = tp * df["volume"]                                    # money flow
    pos = mf.where(tp > tp.shift(), 0.0)
    neg = mf.where(tp < tp.shift(), 0.0)
    ratio = pos.rolling(period).sum() / neg.rolling(period).sum().replace(0, np.nan)
    return 100 - 100 / (1 + ratio)


# ---------- 4. Xác nhận sự kiện giá bằng volume ----------

def confirm_breakout(df: pd.DataFrame, base_window: int = 20,
                     vol_mult: float = 1.5) -> dict:
    """Phiên gần nhất có phá đỉnh/thủng đáy N phiên không, và volume có xác nhận?"""
    if len(df) < base_window + 1:
        return {"event": "Thiếu dữ liệu", "vol_confirm": None}
    last = df.iloc[-1]
    prior = df.iloc[-(base_window + 1):-1]
    vr = last["volume"] / prior["volume"].mean()
    if last["close"] > prior["high"].max():
        event = "Breakout đỉnh %d phiên" % base_window
    elif last["close"] < prior["low"].min():
        event = "Breakdown đáy %d phiên" % base_window
    else:
        return {"event": "Không có", "vol_confirm": None, "vol_ratio_event": round(float(vr), 2)}
    return {"event": event,
            "vol_confirm": bool(vr >= vol_mult),
            "vol_ratio_event": round(float(vr), 2)}


# ---------- Chấm điểm volume (-100..+100) để ghép vào analyze_frame ----------

def volume_score(df: pd.DataFrame) -> dict:
    """Điểm dương = volume ủng hộ phe mua; âm = ủng hộ phe bán."""
    out = volume_state(df)
    out["obv_signal"] = obv_divergence(df)
    m = mfi(df).iloc[-1]
    out["mfi"] = round(float(m), 1) if not np.isnan(m) else np.nan
    bo = confirm_breakout(df)
    out.update({"event": bo["event"], "event_vol_confirm": bo.get("vol_confirm")})

    score, weight = 0.0, 0.0
    # a) Giá & volume cùng chiều phiên gần nhất
    if len(df) >= 2:
        up = df["close"].iloc[-1] > df["close"].iloc[-2]
        heavy = out["vol_ratio"] >= 1.2
        light = out["vol_ratio"] <= 0.7
        if up and heavy:   score += 30      # tăng có tiền
        elif up and light: score += 5       # tăng yếu ớt
        elif (not up) and heavy: score -= 30 # giảm có cung chủ động
        elif (not up) and light: score -= 5  # giảm cạn cung (lành mạnh)
        weight += 30
    # b) OBV
    if "dương" in out["obv_signal"]:   score += 25
    elif "âm" in out["obv_signal"]:    score -= 25
    weight += 25
    # c) MFI
    if not np.isnan(out.get("mfi", np.nan)):
        if out["mfi"] >= 60:   score += 20
        elif out["mfi"] <= 40: score -= 20
        weight += 20
    # d) Sự kiện có volume xác nhận
    if out["event_vol_confirm"] is True:
        score += 25 if "Breakout" in out["event"] else -25
        weight += 25
    elif out["event_vol_confirm"] is False:
        # breakout/breakdown KHÔNG volume = nghi ngờ tín hiệu giả, trừ nhẹ theo chiều ngược
        score += -10 if "Breakout" in out["event"] else 10
        weight += 25
    out["volume_score"] = round(100 * score / weight, 1) if weight else np.nan
    return out


# ---------- Hướng dẫn tích hợp vào mtf_vn30.py ----------
INTEGRATION_GUIDE = """
1) Trong analyze_frame(df) của mtf_vn30.py, thêm cuối hàm:
       from volume_analysis import volume_score
       vs = volume_score(df)
       # trộn 20% điểm volume vào điểm khung:
       if not np.isnan(vs["volume_score"]) and not np.isnan(final):
           final = round(0.8 * final + 0.2 * vs["volume_score"], 1)
       kết quả trả thêm: "vol_ratio": vs["vol_ratio"], "obv": vs["obv_signal"], "mfi": vs["mfi"]

2) Trong analyze_symbol(), thêm cột hiển thị:
       "Vol": d_vs["vol_state"], "OBV": d_vs["obv_signal"], "MFI_D": d_vs["mfi"]

3) Trong render_mtf_tab(), thêm bộ lọc nhanh:
       st.checkbox("Chỉ hiện mã có volume xác nhận (breakout + vol>=1.5x)")

4) Quy tắc đọc nhanh:
   - Điểm khung DƯƠNG + volume_score DƯƠNG  -> tín hiệu tin cậy cao
   - Điểm khung DƯƠNG + volume_score ÂM     -> nghi bull trap, giảm tỷ trọng tin
   - Điểm khung ÂM  + Phân kỳ dương OBV     -> theo dõi gom hàng (kiểu HPG hiện tại?)
   - Volume 'Cạn kiệt' kéo dài trong nền giá hẹp -> thường đứng trước biến động lớn
"""

if __name__ == "__main__":
    # Demo với dữ liệu giả lập
    np.random.seed(7)
    idx = pd.bdate_range("2025-01-01", periods=300)
    close = 25 * np.exp(np.cumsum(np.random.normal(0.0004, 0.018, 300)))
    vol = np.random.randint(2e6, 8e6, 300).astype(float)
    vol[-1] *= 2.2  # giả lập phiên đột biến
    df = pd.DataFrame({"open": close * 0.995, "high": close * 1.012,
                       "low": close * 0.988, "close": close, "volume": vol}, index=idx)
    from pprint import pprint
    pprint(volume_score(df))
    print(INTEGRATION_GUIDE)
