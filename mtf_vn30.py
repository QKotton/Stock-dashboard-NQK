# -*- coding: utf-8 -*-
"""
Phân tích trạng thái thị trường ĐA KHUNG (Daily / Weekly / Monthly) cho rổ VN30
Dữ liệu: vnstock (nguồn VCI/TCBS) - chỉ cần dữ liệu ngày, tự resample lên tuần/tháng

Cách dùng:
  1) Chạy độc lập:    python mtf_vn30.py
  2) Nhúng Streamlit: from mtf_vn30 import render_mtf_tab; render_mtf_tab()

Chỉ báo mỗi khung: MA20/MA50/MA200, RSI(14), MACD(12,26,9)
Trạng thái mỗi khung được chấm điểm -100..+100 rồi phân loại:
  Tăng mạnh / Tăng / Trung tính / Giảm / Giảm mạnh
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ------------------------- Cấu hình -------------------------

VN30_FALLBACK = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "LPB", "MBB", "MSN", "MWG", "PLX", "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
]

LOOKBACK_YEARS = 5          # đủ dài để MA200 khung tuần có nghĩa
SOURCE = "VCI"              # đổi sang "TCBS" nếu VCI lỗi
SLEEP_BETWEEN_CALLS = 0.6   # tránh bị rate-limit


def get_vn30_symbols():
    """Lấy danh sách VN30 mới nhất từ vnstock, lỗi thì dùng fallback."""
    try:
        from vnstock import Vnstock
        listing = Vnstock().stock(symbol="ACB", source=SOURCE).listing
        syms = listing.symbols_by_group("VN30")
        syms = sorted(set(map(str, syms.tolist())))
        if 25 <= len(syms) <= 35:
            return syms
    except Exception as e:
        print(f"[!] Không lấy được VN30 từ vnstock ({e}) -> dùng danh sách dự phòng")
    return VN30_FALLBACK


def fetch_daily(symbol: str) -> pd.DataFrame:
    """Tải OHLCV khung ngày cho 1 mã."""
    from vnstock import Vnstock
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=365 * LOOKBACK_YEARS)).strftime("%Y-%m-%d")
    stock = Vnstock().stock(symbol=symbol, source=SOURCE)
    df = stock.quote.history(start=start, end=end, interval="1D")
    df = df.rename(columns=str.lower)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    return df[["open", "high", "low", "close", "volume"]].astype(float)


# ------------------------- Resample & chỉ báo -------------------------

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """rule: 'W-FRI' (tuần) hoặc 'ME' (tháng)."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = df.resample(rule).agg(agg).dropna()
    return out


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


# ------------------------- Chấm điểm 1 khung -------------------------

def analyze_frame(df: pd.DataFrame) -> dict:
    """Trả về điểm số & nhãn trạng thái cho 1 khung thời gian."""
    n = len(df)
    if n < 30:
        return {"score": np.nan, "state": "Thiếu dữ liệu", "rsi": np.nan,
                "close_vs_ma20": np.nan, "close_vs_ma50": np.nan, "macd_hist": np.nan}

    close = df["close"]
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean() if n >= 50 else None
    ma200 = close.rolling(200).mean() if n >= 200 else None
    r = rsi(close).iloc[-1]
    macd_line, sig_line, hist = macd(close)

    c = close.iloc[-1]
    score, weight = 0.0, 0.0

    # 1) Vị trí giá so với MA (trọng số lớn nhất)
    score += (25 if c > ma20.iloc[-1] else -25); weight += 25
    if ma50 is not None and not np.isnan(ma50.iloc[-1]):
        score += (20 if c > ma50.iloc[-1] else -20); weight += 20
        # MA20 vs MA50 (cấu trúc xu hướng)
        score += (10 if ma20.iloc[-1] > ma50.iloc[-1] else -10); weight += 10
    if ma200 is not None and not np.isnan(ma200.iloc[-1]):
        score += (10 if c > ma200.iloc[-1] else -10); weight += 10

    # 2) RSI: >55 tích cực, <45 tiêu cực, giữa = trung tính
    if not np.isnan(r):
        if r >= 55:   score += 15
        elif r <= 45: score -= 15
        weight += 15

    # 3) MACD: đường MACD vs signal + histogram đang mở rộng hay thu hẹp
    if not np.isnan(hist.iloc[-1]):
        score += (10 if macd_line.iloc[-1] > sig_line.iloc[-1] else -10); weight += 10
        if len(hist) >= 3:
            momentum_up = hist.iloc[-1] > hist.iloc[-2]
            score += (10 if momentum_up else -10); weight += 10

    final = round(100 * score / weight, 1) if weight else np.nan

    if final >= 60:    state = "Tăng mạnh"
    elif final >= 20:  state = "Tăng"
    elif final > -20:  state = "Trung tính"
    elif final > -60:  state = "Giảm"
    else:              state = "Giảm mạnh"

    return {
        "score": final, "state": state, "rsi": round(float(r), 1),
        "close_vs_ma20": round((c / ma20.iloc[-1] - 1) * 100, 1),
        "close_vs_ma50": round((c / ma50.iloc[-1] - 1) * 100, 1) if ma50 is not None else np.nan,
        "macd_hist": round(float(hist.iloc[-1]), 3),
    }


def analyze_symbol(symbol: str, daily: pd.DataFrame) -> dict:
    weekly = resample_ohlcv(daily, "W-FRI")
    monthly = resample_ohlcv(daily, "ME")

    d = analyze_frame(daily)
    w = analyze_frame(weekly)
    m = analyze_frame(monthly)

    # Điểm tổng hợp: khung lớn trọng số cao hơn (tháng 0.5, tuần 0.3, ngày 0.2)
    parts, weights = [], []
    for res, wt in [(m, 0.5), (w, 0.3), (d, 0.2)]:
        if not np.isnan(res["score"]):
            parts.append(res["score"] * wt); weights.append(wt)
    composite = round(sum(parts) / sum(weights), 1) if weights else np.nan

    # Đồng thuận đa khung: cả 3 khung cùng chiều?
    states = [d["state"], w["state"], m["state"]]
    if all(s in ("Tăng", "Tăng mạnh") for s in states):
        alignment = "Đồng thuận TĂNG"
    elif all(s in ("Giảm", "Giảm mạnh") for s in states):
        alignment = "Đồng thuận GIẢM"
    else:
        alignment = "Phân kỳ khung"

    change_1d = round((daily["close"].iloc[-1] / daily["close"].iloc[-2] - 1) * 100, 2) if len(daily) > 1 else np.nan

    return {
        "Mã": symbol,
        "Giá": daily["close"].iloc[-1],
        "%1D": change_1d,
        "Ngày": d["state"], "RSI_D": d["rsi"],
        "Tuần": w["state"], "RSI_W": w["rsi"],
        "Tháng": m["state"], "RSI_M": m["rsi"],
        "Điểm D": d["score"], "Điểm W": w["score"], "Điểm M": m["score"],
        "Điểm tổng hợp": composite,
        "Đa khung": alignment,
    }


# ------------------------- Chạy toàn bộ VN30 -------------------------

def run_vn30_mtf(progress_callback=None) -> pd.DataFrame:
    symbols = get_vn30_symbols()
    rows, errors = [], []
    for i, sym in enumerate(symbols):
        try:
            daily = fetch_daily(sym)
            rows.append(analyze_symbol(sym, daily))
        except Exception as e:
            errors.append((sym, str(e)))
        if progress_callback:
            progress_callback((i + 1) / len(symbols), sym)
        time.sleep(SLEEP_BETWEEN_CALLS)

    df = pd.DataFrame(rows).sort_values("Điểm tổng hợp", ascending=False).reset_index(drop=True)
    if errors:
        print("Lỗi khi tải:", errors)
    return df


def market_breadth(df: pd.DataFrame) -> dict:
    """Độ rộng thị trường theo từng khung."""
    out = {}
    for col in ["Ngày", "Tuần", "Tháng"]:
        up = df[col].isin(["Tăng", "Tăng mạnh"]).sum()
        down = df[col].isin(["Giảm", "Giảm mạnh"]).sum()
        out[col] = {"Tăng": int(up), "Giảm": int(down), "Trung tính": int(len(df) - up - down)}
    out["Đồng thuận tăng"] = int((df["Đa khung"] == "Đồng thuận TĂNG").sum())
    out["Đồng thuận giảm"] = int((df["Đa khung"] == "Đồng thuận GIẢM").sum())
    return out


# ------------------------- Giao diện Streamlit -------------------------

def render_mtf_tab():
    """Gọi hàm này trong 1 tab của dashboard Streamlit."""
    import streamlit as st

    st.subheader("📊 Trạng thái thị trường đa khung – VN30")
    st.caption("Khung Ngày / Tuần / Tháng · MA20-50-200, RSI(14), MACD(12,26,9)")

    if st.button("🔄 Quét VN30", type="primary") or "mtf_df" not in st.session_state:
        bar = st.progress(0.0, text="Đang tải dữ liệu...")
        df = run_vn30_mtf(lambda p, s: bar.progress(p, text=f"Đang xử lý {s}..."))
        bar.empty()
        st.session_state["mtf_df"] = df

    df = st.session_state["mtf_df"]
    breadth = market_breadth(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tăng (khung Ngày)", f"{breadth['Ngày']['Tăng']}/{len(df)}")
    c2.metric("Tăng (khung Tuần)", f"{breadth['Tuần']['Tăng']}/{len(df)}")
    c3.metric("Đồng thuận TĂNG", breadth["Đồng thuận tăng"])
    c4.metric("Đồng thuận GIẢM", breadth["Đồng thuận giảm"])

    def color_state(v):
        colors = {"Tăng mạnh": "#0a7a2f", "Tăng": "#3fa35c",
                  "Giảm": "#c05555", "Giảm mạnh": "#8f1d1d",
                  "Trung tính": "#8a8a8a"}
        return f"color: white; background-color: {colors.get(v, '')}" if v in colors else ""

    styled = df.style.map(color_state, subset=["Ngày", "Tuần", "Tháng"]) \
                     .background_gradient(subset=["Điểm tổng hợp"], cmap="RdYlGn", vmin=-100, vmax=100) \
                     .format({"Giá": "{:,.0f}", "%1D": "{:+.2f}%"})
    st.dataframe(styled, use_container_width=True, height=900)

    st.download_button("⬇️ Tải CSV", df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="vn30_mtf.csv", mime="text/csv")


if __name__ == "__main__":
    result = run_vn30_mtf(lambda p, s: print(f"  {p:>5.0%}  {s}"))
    print("\n=== KẾT QUẢ ĐA KHUNG VN30 ===")
    print(result.to_string(index=False))
    print("\n=== ĐỘ RỘNG THỊ TRƯỜNG ===")
    for k, v in market_breadth(result).items():
        print(f"{k}: {v}")
    result.to_csv("vn30_mtf.csv", index=False, encoding="utf-8-sig")
    print("\nĐã lưu vn30_mtf.csv")
