# -*- coding: utf-8 -*-
"""Xuất báo cáo Excel phân tích kỹ thuật cho một hoặc nhiều mã chứng khoán.

Theo chỉ dẫn HUONG_DAN_XUAT_EXCEL.md:
- Import nguyên trạng logic chấm điểm từ mtf_vn30.py và volume_analysis.py (không sửa).
- Một mã  -> build_symbol_report(): 3 sheet "Danh gia ky thuat" / "So sanh RS & RRG"
  / "Nguon & phuong phap".
- Nhiều mã -> build_multi_report(): thêm sheet "Tong quan so sanh" + biểu đồ RRG chung,
  mỗi mã một sheet đánh giá riêng.

Style bằng openpyxl trực tiếp (không dùng pandas .style.to_excel).
Tên sheet không dấu; nội dung ô tiếng Việt có dấu đầy đủ.
"""
from __future__ import annotations

import datetime as dt
import inspect
import io

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import Reference, ScatterChart, Series
from openpyxl.chart.marker import Marker
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

import volume_analysis
from mtf_vn30 import analyze_frame, analyze_symbol, fetch_daily, resample_ohlcv

# ---------------------------------------------------------------------------
# Tham số phân tích (đọc từ code thực của các module — không hardcode lại)
# ---------------------------------------------------------------------------
RSI_PERIOD = inspect.signature(volume_analysis.mfi).parameters["period"].default  # cùng 14
MFI_PERIOD = inspect.signature(volume_analysis.mfi).parameters["period"].default
VOL_CONFIRM_MULT = inspect.signature(volume_analysis.confirm_breakout).parameters["vol_mult"].default
BREAKOUT_WINDOW = inspect.signature(volume_analysis.confirm_breakout).parameters["base_window"].default
OBV_LOOKBACK = inspect.signature(volume_analysis.obv_divergence).parameters["lookback"].default
MA_WINDOWS = (20, 50, 200)          # theo analyze_frame của mtf_vn30.py
FRAME_WEIGHTS = "Tháng 0.5 / Tuần 0.3 / Ngày 0.2"  # theo analyze_symbol của mtf_vn30.py
RRG_RS_WINDOW = 40                  # tuần — SMA của RS
RRG_MOM_WINDOW = 10                 # tuần — SMA của RS-Ratio
RS_PERIODS = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252}
MIN_ROWS_RELIABLE = 250

# ---------------------------------------------------------------------------
# Màu & style theo quy ước của file chỉ dẫn
# ---------------------------------------------------------------------------
NAVY = "1F4E79"
GROUP_BG = "D9E2F3"
BORDER_GRAY = "BFBFBF"
STATE_COLORS = {
    "Tăng mạnh": "0A7A2F",
    "Tăng": "3FA35C",
    "Trung tính": "8A8A8A",
    "Giảm": "C05555",
    "Giảm mạnh": "8F1D1D",
}
RS_GREEN = "C6EFCE"
RS_RED = "FFC7CE"
RS_YELLOW = "FFEB9C"
RRG_COLORS = {"Leading": RS_GREEN, "Improving": RS_YELLOW, "Weakening": "FFD9B3", "Lagging": RS_RED}

FONT = "Arial"
_thin = Side(style="thin", color=BORDER_GRAY)
BORDER = Border(left=_thin, right=_thin, top=_thin, bottom=_thin)


def _cell(ws, row, col, value, *, bold=False, size=10, color="000000", bg=None,
          num_fmt=None, wrap=True, align="left"):
    c = ws.cell(row=row, column=col, value=value)
    c.font = Font(name=FONT, size=size, bold=bold, color=color)
    c.alignment = Alignment(wrap_text=wrap, vertical="center", horizontal=align)
    c.border = BORDER
    if bg:
        c.fill = PatternFill("solid", fgColor=bg)
    if num_fmt:
        c.number_format = num_fmt
    return c


def _title(ws, row, text, subtitle=None):
    c = ws.cell(row=row, column=1, value=text)
    c.font = Font(name=FONT, size=14, bold=True, color=NAVY)
    if subtitle:
        s = ws.cell(row=row + 1, column=1, value=subtitle)
        s.font = Font(name=FONT, size=9, italic=True, color="666666")
        return row + 3
    return row + 2


def _group_header(ws, row, text, span=2):
    for col in range(1, span + 1):
        _cell(ws, row, col, text if col == 1 else None, bold=True, color=NAVY, bg=GROUP_BG)
    return row + 1


def _pair(ws, row, label, value, *, value_bg=None, value_color="000000",
          value_bold=False, num_fmt=None):
    _cell(ws, row, 1, label)
    _cell(ws, row, 2, value, bg=value_bg, color=value_color, bold=value_bold, num_fmt=num_fmt)
    return row + 1


# ---------------------------------------------------------------------------
# Dữ liệu vào
# ---------------------------------------------------------------------------
COLUMN_ALIASES = {
    "time": {"time", "date", "ngay", "ngày", "datetime", "tradingdate"},
    "open": {"open", "mo cua", "mở cửa", "gia mo cua"},
    "high": {"high", "cao nhat", "cao nhất"},
    "low": {"low", "thap nhat", "thấp nhất"},
    "close": {"close", "dong cua", "đóng cửa", "gia dong cua", "adjclose", "adj close"},
    "volume": {"volume", "vol", "khoi luong", "khối lượng", "kl"},
}


def normalize_ohlcv(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Chuẩn hóa OHLCV người dùng upload: nhận diện tên cột hoa/thường, parse ngày
    dd/mm/yyyy và yyyy-mm-dd, sắp tăng dần, loại trùng ngày.

    Trả về (df chuẩn index thời gian, danh sách cảnh báo). Raise ValueError nếu thiếu cột.
    """
    warnings: list[str] = []
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower()
        for std, aliases in COLUMN_ALIASES.items():
            if key in aliases:
                rename[col] = std
                break
    df = df.rename(columns=rename)
    missing = [c for c in ("time", "open", "high", "low", "close", "volume") if c not in df.columns]
    if missing:
        raise ValueError(
            f"File thiếu cột bắt buộc: {', '.join(missing)}. "
            "Cần tối thiểu: time/date, open, high, low, close, volume."
        )
    ts = pd.to_datetime(df["time"], errors="coerce", dayfirst=False, format="mixed")
    if ts.isna().mean() > 0.5:  # đa số không parse được -> thử dd/mm/yyyy
        ts = pd.to_datetime(df["time"], errors="coerce", dayfirst=True, format="mixed")
    df = df.assign(time=ts).dropna(subset=["time"])
    df = df.drop_duplicates(subset=["time"], keep="last").sort_values("time")
    df = df.set_index("time")[["open", "high", "low", "close", "volume"]].astype(float)
    if len(df) < MIN_ROWS_RELIABLE:
        warnings.append(
            f"Dữ liệu chỉ có {len(df)} phiên (<{MIN_ROWS_RELIABLE}) — "
            "MA200 và RRG kém tin cậy, kết quả chỉ mang tính tham khảo."
        )
    return df, warnings


def fetch_daily_dong(symbol: str) -> pd.DataFrame:
    """fetch_daily của mtf_vn30 trả giá theo nghìn đồng — quy về đồng cho báo cáo."""
    df = fetch_daily(symbol)
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]] * 1000
    return df


def fetch_benchmark(index_symbol: str = "VNINDEX") -> pd.DataFrame:
    """OHLCV chỉ số (điểm) làm chuẩn so sánh RS/RRG."""
    import data_loader

    end = dt.date.today()
    start = end - dt.timedelta(days=365 * 5)
    df = data_loader.get_index_history(index_symbol, str(start), str(end))
    df = df.assign(time=pd.to_datetime(df["time"])).set_index("time").sort_index()
    return df[["open", "high", "low", "close", "volume"]].astype(float)


# ---------------------------------------------------------------------------
# Tính RS & RRG
# ---------------------------------------------------------------------------
def rs_performance(stock_close: pd.Series, bench_close: pd.Series) -> dict[str, float | None]:
    """Hiệu suất mã trừ hiệu suất benchmark (điểm %) theo 4 kỳ hạn; thiếu -> None."""
    out: dict[str, float | None] = {}
    joined = pd.concat([stock_close.rename("s"), bench_close.rename("b")], axis=1).dropna()
    for label, sessions in RS_PERIODS.items():
        if len(joined) < sessions + 1:
            out[label] = None
            continue
        s_ret = joined["s"].iloc[-1] / joined["s"].iloc[-sessions - 1] - 1
        b_ret = joined["b"].iloc[-1] / joined["b"].iloc[-sessions - 1] - 1
        out[label] = round((s_ret - b_ret) * 100, 1)
    return out


def rrg_weekly(stock_daily: pd.DataFrame, bench_daily: pd.DataFrame) -> pd.DataFrame | None:
    """Chuỗi RRG tuần (W-FRI): RS-Ratio (SMA 40w) và RS-Momentum (SMA 10w).

    Trả None nếu lịch sử chung < 50 tuần.
    """
    s = resample_ohlcv(stock_daily, "W-FRI")["close"].rename("s")
    b = resample_ohlcv(bench_daily, "W-FRI")["close"].rename("b")
    joined = pd.concat([s, b], axis=1).dropna()
    if len(joined) < RRG_RS_WINDOW + RRG_MOM_WINDOW:
        return None
    rs = joined["s"] / joined["b"]
    rs_ratio = 100 * rs / rs.rolling(RRG_RS_WINDOW).mean()
    rs_mom = 100 * rs_ratio / rs_ratio.rolling(RRG_MOM_WINDOW).mean()
    out = pd.DataFrame({"rs_ratio": rs_ratio, "rs_momentum": rs_mom}).dropna()
    return out if not out.empty else None


def rrg_quadrant(ratio: float, momentum: float) -> str:
    if ratio >= 100 and momentum >= 100:
        return "Leading"
    if ratio < 100 and momentum >= 100:
        return "Improving"
    if ratio >= 100 and momentum < 100:
        return "Weakening"
    return "Lagging"


# ---------------------------------------------------------------------------
# Nhận định tự động
# ---------------------------------------------------------------------------
def auto_warnings(d_frame: dict, vs: dict, daily: pd.DataFrame) -> str:
    """Hàng 'Cảnh báo tự động' theo tổ hợp điểm khung ngày × volume."""
    notes = []
    d_score = d_frame.get("score")
    v_score = vs.get("volume_score")
    if d_score is not None and v_score is not None and not (np.isnan(d_score) or np.isnan(v_score)):
        if d_score > 0 and v_score < 0:
            notes.append("Nghi bull trap — điểm xu hướng dương nhưng dòng tiền âm")
        if d_score < 0 and "dương" in str(vs.get("obv_signal", "")):
            notes.append("Theo dõi gom hàng — giá yếu nhưng OBV phân kỳ dương")
    if vs.get("vol_state") == "Cạn kiệt" and len(daily) >= 20:
        rng = (daily["high"].tail(20).max() - daily["low"].tail(20).min()) / daily["close"].iloc[-1]
        if rng < 0.08:
            notes.append("Nền tích lũy (volume cạn kiệt + biên độ hẹp), chờ biến động")
    return "; ".join(notes) if notes else "Không có"


def action_hint(alignment: str, vs: dict, rs_3m: float | None) -> str:
    """HÀNH ĐỘNG THAM KHẢO — tối đa 2 câu, không dùng từ mua/bán."""
    v_score = vs.get("volume_score")
    v_pos = v_score is not None and not np.isnan(v_score) and v_score > 0
    rs_pos = rs_3m is not None and rs_3m > 0
    if alignment == "Đồng thuận TĂNG" and v_pos and rs_pos:
        return ("Xu hướng, dòng tiền và sức mạnh tương đối cùng ủng hộ — đưa vào vùng quan sát "
                "ưu tiên. Canh nhịp điều chỉnh về MA20 khung ngày để theo dõi phản ứng giá.")
    if alignment == "Đồng thuận TĂNG":
        return ("Xu hướng tăng nhưng dòng tiền/sức mạnh tương đối chưa xác nhận. "
                "Chờ phiên bùng nổ khối lượng hoặc RS chuyển dương trước khi nâng mức chú ý.")
    if alignment == "Đồng thuận GIẢM":
        return ("Cả ba khung cùng chiều giảm — đứng ngoài quan sát. Chờ tín hiệu tạo đáy "
                "(phân kỳ dương OBV, volume cạn kiệt) và xác nhận từ khung tuần.")
    if "dương" in str(vs.get("obv_signal", "")):
        return ("Các khung chưa đồng thuận nhưng có dấu hiệu gom hàng theo OBV. "
                "Theo dõi vùng hỗ trợ gần nhất, chờ xác nhận từ khung tuần.")
    return ("Tín hiệu giữa các khung còn trái chiều — tiếp tục theo dõi. "
            "Chờ đồng thuận giữa xu hướng, khối lượng và sức mạnh tương đối rồi mới hành động.")


# ---------------------------------------------------------------------------
# Sheet 1 — Danh gia ky thuat (bảng dọc)
# ---------------------------------------------------------------------------
def _sheet_danh_gia(ws, symbol: str, daily: pd.DataFrame, mtf: dict, vs: dict,
                    rs_bench: dict | None, exchange: str = "—",
                    warnings: list[str] | None = None,
                    source_note: str | None = None) -> None:
    ws.column_dimensions["A"].width = 26
    ws.column_dimensions["B"].width = 45

    row = _title(
        ws, 1, f"ĐÁNH GIÁ KỸ THUẬT ĐA KHUNG: {symbol.upper()}",
        f"Chạy lúc {dt.datetime.now():%d/%m/%Y %H:%M} · Nguồn: {source_note or 'vnstock (VCI)'} · "
        "Tham khảo, không phải khuyến nghị đầu tư.",
    )
    if warnings:
        for w in warnings:
            _cell(ws, row, 1, f"⚠ {w}", color="9C6500", bg=RS_YELLOW)
            ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=2)
            row += 1
        row += 1

    close = daily["close"]
    last = float(close.iloc[-1])
    chg_1d = (last / close.iloc[-2] - 1) * 100 if len(close) > 1 else np.nan
    hi52 = float(close.tail(252).max())
    lo52 = float(close.tail(252).min())
    vol_ma20 = float(daily["volume"].rolling(20).mean().iloc[-1]) if len(daily) >= 20 else np.nan

    # --- THÔNG TIN CHUNG ---
    row = _group_header(ws, row, "THÔNG TIN CHUNG")
    row = _pair(ws, row, "Sàn niêm yết", exchange)
    row = _pair(ws, row, "Giá gần nhất (đ)", round(last), num_fmt="#,##0")
    row = _pair(ws, row, "Thay đổi 1 phiên", f"{chg_1d:+.2f}%" if not np.isnan(chg_1d) else "—")
    row = _pair(
        ws, row, "Vị trí vùng 52 tuần",
        f"Thấp hơn đỉnh {((hi52 - last) / hi52 * 100):.1f}% · cao hơn đáy "
        f"{((last - lo52) / lo52 * 100):.1f}%",
    )
    row = _pair(ws, row, "Khối lượng TB 20 phiên",
                round(vol_ma20) if not np.isnan(vol_ma20) else "—", num_fmt="#,##0")
    row += 1

    # --- TRẠNG THÁI ĐA KHUNG ---
    row = _group_header(ws, row, "TRẠNG THÁI ĐA KHUNG")
    for label, state_key, score_key in [("Khung Ngày", "Ngày", "Điểm D"),
                                        ("Khung Tuần", "Tuần", "Điểm W"),
                                        ("Khung Tháng", "Tháng", "Điểm M")]:
        state = mtf.get(state_key, "—")
        score = mtf.get(score_key)
        bg = STATE_COLORS.get(state)
        text = f"{state} ({score:+.1f} điểm)" if score is not None and not np.isnan(score) else state
        row = _pair(ws, row, label, text,
                    value_bg=bg, value_color="FFFFFF" if bg else "000000", value_bold=bool(bg))
    alignment = mtf.get("Đa khung", "—")
    composite = mtf.get("Điểm tổng hợp")
    align_bg = (STATE_COLORS["Tăng"] if alignment == "Đồng thuận TĂNG"
                else STATE_COLORS["Giảm"] if alignment == "Đồng thuận GIẢM" else None)
    row = _pair(
        ws, row, "Kết luận đa khung",
        f"{alignment} · Điểm tổng hợp {composite:+.1f}"
        if composite is not None and not np.isnan(composite) else alignment,
        value_bg=align_bg, value_color="FFFFFF" if align_bg else "000000", value_bold=True,
    )
    row += 1

    # --- CHỈ BÁO KỸ THUẬT (khung ngày, gộp volume) ---
    row = _group_header(ws, row, "CHỈ BÁO KỸ THUẬT (KHUNG NGÀY)")
    for w in MA_WINDOWS:
        if len(close) >= w:
            ma = close.rolling(w).mean().iloc[-1]
            row = _pair(ws, row, f"Giá so với MA{w}", f"{(last / ma - 1) * 100:+.1f}%")
        else:
            row = _pair(ws, row, f"Giá so với MA{w}", "— (không đủ dữ liệu)")

    d_frame = analyze_frame(daily)
    r = d_frame.get("rsi", np.nan)
    rsi_note = "tích cực" if r >= 55 else ("tiêu cực" if r < 45 else "trung tính")
    row = _pair(ws, row, f"RSI({RSI_PERIOD})",
                f"{r:.1f} — {rsi_note}" if not np.isnan(r) else "—")

    from mtf_vn30 import macd as _macd

    macd_line, sig_line, hist = _macd(close)
    macd_pos = macd_line.iloc[-1] > sig_line.iloc[-1]
    hist_widen = len(hist) >= 2 and abs(hist.iloc[-1]) > abs(hist.iloc[-2])
    row = _pair(
        ws, row, "MACD(12,26,9)",
        f"{'Trên' if macd_pos else 'Dưới'} đường signal · histogram đang "
        f"{'mở rộng' if hist_widen else 'thu hẹp'}",
    )

    row = _pair(ws, row, "Volume Ratio (KL/TB20)",
                f"×{vs['vol_ratio']:.2f} — nền khối lượng: {vs['vol_state']}")
    obv_sig = str(vs.get("obv_signal", "—"))
    obv_bg = RS_GREEN if "dương" in obv_sig else (RS_RED if "âm" in obv_sig else None)
    row = _pair(ws, row, f"Tín hiệu OBV ({OBV_LOOKBACK} phiên)", obv_sig, value_bg=obv_bg)

    m = vs.get("mfi", np.nan)
    mfi_note = "tiền vào" if (not np.isnan(m) and m >= 60) else (
        "tiền ra" if (not np.isnan(m) and m <= 40) else "trung tính")
    row = _pair(ws, row, f"MFI({MFI_PERIOD})",
                f"{m:.1f} — {mfi_note}" if not np.isnan(m) else "—")
    conflict = (not np.isnan(r) and not np.isnan(m)
                and ((r >= 55 and m <= 40) or (r < 45 and m >= 60)))
    row = _pair(
        ws, row, "RSI vs MFI",
        "MÂU THUẪN — giá và dòng tiền thật đang kể hai câu chuyện khác nhau, thận trọng"
        if conflict else "Đồng thuận",
        value_bg=RS_YELLOW if conflict else None,
    )

    event = str(vs.get("event", "Không có"))
    confirm = vs.get("event_vol_confirm")
    if "Breakout" in event or "Breakdown" in event:
        if confirm:
            ev_bg = RS_GREEN if "Breakout" in event else RS_RED
            ev_text = f"{event} — CÓ volume xác nhận (≥{VOL_CONFIRM_MULT}x)"
        else:
            ev_bg = RS_YELLOW
            ev_text = f"{event} — KHÔNG có volume xác nhận (nghi tín hiệu giả)"
    else:
        ev_bg, ev_text = None, "Không có breakout/breakdown trong phạm vi 20 phiên"
    row = _pair(ws, row, f"Sự kiện giá ({BREAKOUT_WINDOW} phiên)", ev_text, value_bg=ev_bg)

    v_score = vs.get("volume_score")
    row = _pair(ws, row, "Điểm volume tổng hợp",
                f"{v_score:+.1f} / ±100" if v_score is not None and not np.isnan(v_score) else "—")
    row = _pair(ws, row, "Cảnh báo tự động", auto_warnings(d_frame, vs, daily),
                value_bg=RS_YELLOW if auto_warnings(d_frame, vs, daily) != "Không có" else None)
    row += 1

    # --- HÀNH ĐỘNG THAM KHẢO ---
    row = _group_header(ws, row, "HÀNH ĐỘNG THAM KHẢO")
    rs_3m = rs_bench.get("3M") if rs_bench else None
    _pair(ws, row, "Nhận định", action_hint(alignment, vs, rs_3m))


# ---------------------------------------------------------------------------
# Sheet 2 — So sanh RS & RRG
# ---------------------------------------------------------------------------
def _rs_fill(v: float | None) -> str | None:
    if v is None:
        return None
    if v >= 1:
        return RS_GREEN
    if v <= -1:
        return RS_RED
    return RS_YELLOW


def _sheet_rs_rrg(ws, symbol: str, daily: pd.DataFrame,
                  benchmark: pd.DataFrame | None,
                  benchmark2: pd.DataFrame | None = None,
                  bench2_name: str = "VN30") -> None:
    ws.column_dimensions["A"].width = 26
    for col in "BCDEF":
        ws.column_dimensions[col].width = 16

    row = _title(ws, 1, f"SỨC MẠNH TƯƠNG ĐỐI (RS) & RRG: {symbol.upper()}",
                 "RS = hiệu suất mã trừ hiệu suất chỉ số (điểm %). RRG tính trên chuỗi tuần.")

    if benchmark is None or benchmark.empty:
        _cell(ws, row, 1, "Không có dữ liệu chỉ số chuẩn — bỏ qua phần RS/RRG.", bg=RS_YELLOW)
        return

    # --- Bảng RS ---
    headers = ["Kỳ hạn", "RS vs VN-Index (điểm %)"]
    if benchmark2 is not None and not benchmark2.empty:
        headers.append(f"RS vs {bench2_name} (điểm %)")
    for j, h in enumerate(headers, start=1):
        _cell(ws, row, j, h, bold=True, color="FFFFFF", bg=NAVY)
    row += 1
    rs1 = rs_performance(daily["close"], benchmark["close"])
    rs2 = (rs_performance(daily["close"], benchmark2["close"])
           if benchmark2 is not None and not benchmark2.empty else None)
    for label in RS_PERIODS:
        _cell(ws, row, 1, label)
        v = rs1[label]
        _cell(ws, row, 2, v if v is not None else "—",
              bg=_rs_fill(v), num_fmt="+0.0;-0.0", align="center")
        if rs2 is not None:
            v2 = rs2[label]
            _cell(ws, row, 3, v2 if v2 is not None else "—",
                  bg=_rs_fill(v2), num_fmt="+0.0;-0.0", align="center")
        row += 1
    row += 1

    # --- RRG ---
    rrg = rrg_weekly(daily, benchmark)
    if rrg is None:
        _cell(ws, row, 1, f"Không đủ dữ liệu cho RRG (cần ≥{RRG_RS_WINDOW + RRG_MOM_WINDOW} "
                          "tuần lịch sử chung với chỉ số).", bg=RS_YELLOW)
        row += 2
    else:
        latest = rrg.iloc[-1]
        quad = rrg_quadrant(latest["rs_ratio"], latest["rs_momentum"])
        for j, h in enumerate(["RS-Ratio", "RS-Momentum", "Góc phần tư (mốc 100)"], start=1):
            _cell(ws, row, j, h, bold=True, color="FFFFFF", bg=NAVY)
        row += 1
        _cell(ws, row, 1, round(float(latest["rs_ratio"]), 2), num_fmt="0.00", align="center")
        _cell(ws, row, 2, round(float(latest["rs_momentum"]), 2), num_fmt="0.00", align="center")
        _cell(ws, row, 3, quad, bg=RRG_COLORS[quad], bold=True, align="center")
        row += 2

        # Đuôi quỹ đạo 8 tuần
        _cell(ws, row, 1, "Quỹ đạo 8 tuần gần nhất", bold=True, color=NAVY, bg=GROUP_BG)
        _cell(ws, row, 2, None, bg=GROUP_BG)
        _cell(ws, row, 3, None, bg=GROUP_BG)
        row += 1
        for j, h in enumerate(["Tuần", "RS-Ratio", "RS-Momentum"], start=1):
            _cell(ws, row, j, h, bold=True, color="FFFFFF", bg=NAVY)
        row += 1
        tail = rrg.tail(8)
        tail_start = row
        for ts, r_ in tail.iterrows():
            _cell(ws, row, 1, ts.strftime("%d/%m/%Y"))
            _cell(ws, row, 2, round(float(r_["rs_ratio"]), 2), num_fmt="0.00", align="center")
            _cell(ws, row, 3, round(float(r_["rs_momentum"]), 2), num_fmt="0.00", align="center")
            row += 1
        tail_end = row - 1

        chart = ScatterChart()
        chart.title = f"RRG — quỹ đạo 8 tuần của {symbol.upper()}"
        chart.x_axis.title = "RS-Ratio (>100 = mạnh hơn chỉ số)"
        chart.y_axis.title = "RS-Momentum (>100 = đang cải thiện)"
        chart.height, chart.width = 10, 16
        xs = Reference(ws, min_col=2, min_row=tail_start, max_row=tail_end)
        ys = Reference(ws, min_col=3, min_row=tail_start, max_row=tail_end)
        s = Series(ys, xs, title="Quỹ đạo")
        s.marker = Marker(symbol="circle", size=6)
        s.graphicalProperties.line.width = 15000
        chart.series.append(s)
        xs_last = Reference(ws, min_col=2, min_row=tail_end, max_row=tail_end)
        ys_last = Reference(ws, min_col=3, min_row=tail_end, max_row=tail_end)
        s_last = Series(ys_last, xs_last, title="Tuần mới nhất")
        s_last.marker = Marker(symbol="diamond", size=12)
        s_last.graphicalProperties.line.noFill = True
        chart.series.append(s_last)
        ws.add_chart(chart, f"E{tail_start - 2}")
        row += 1

    _cell(ws, row, 1,
          "Ghi chú: RS là thước đo TƯƠNG ĐỐI — mã ở góc Leading vẫn có thể giảm giá tuyệt đối. "
          "VN-Index có thể bị méo bởi nhóm vốn hóa lớn, nên so thêm với VN30 hoặc chỉ số ngành.",
          color="666666")
    ws.merge_cells(start_row=row, start_column=1, end_row=row + 1, end_column=4)


# ---------------------------------------------------------------------------
# Sheet 3 — Nguon & phuong phap
# ---------------------------------------------------------------------------
def _sheet_nguon(ws, meta: dict) -> None:
    ws.column_dimensions["A"].width = 26
    ws.column_dimensions["B"].width = 60
    row = _title(ws, 1, "NGUỒN DỮ LIỆU & PHƯƠNG PHÁP")

    row = _group_header(ws, row, "NGUỒN DỮ LIỆU")
    row = _pair(ws, row, "Nguồn", meta.get("source", "vnstock (VCI)"))
    row = _pair(ws, row, "Thời điểm chạy", f"{dt.datetime.now():%d/%m/%Y %H:%M}")
    row = _pair(ws, row, "Số phiên dữ liệu", meta.get("n_rows", "—"))
    row = _pair(ws, row, "Khoảng ngày", meta.get("date_range", "—"))
    row = _pair(
        ws, row, "Lưu ý giá điều chỉnh",
        "Giá vnstock đã điều chỉnh cổ tức/chia tách. Khi đối chiếu nguồn ngoài phải kiểm tra "
        "sự kiện chia tách; ưu tiên CafeF/Vietstock/FireAnt, thận trọng với "
        "Investing/TradingView cho cổ phiếu VN.",
    )
    row += 1

    row = _group_header(ws, row, "THAM SỐ PHÂN TÍCH")
    row = _pair(ws, row, "Cửa sổ MA", " / ".join(f"MA{w}" for w in MA_WINDOWS))
    row = _pair(ws, row, "Chu kỳ RSI / MFI", f"{RSI_PERIOD} / {MFI_PERIOD}")
    row = _pair(ws, row, "Ngưỡng volume xác nhận", f"≥ {VOL_CONFIRM_MULT}x trung bình 20 phiên")
    row = _pair(ws, row, "Cửa sổ breakout/breakdown", f"{BREAKOUT_WINDOW} phiên")
    row = _pair(ws, row, "Trọng số khung (điểm tổng hợp)", FRAME_WEIGHTS)
    row = _pair(ws, row, "Cửa sổ RRG", f"RS-Ratio SMA {RRG_RS_WINDOW} tuần / "
                                       f"RS-Momentum SMA {RRG_MOM_WINDOW} tuần")
    _pair(ws, row, "Kỳ hạn RS", " / ".join(f"{k}={v} phiên" for k, v in RS_PERIODS.items()))


# ---------------------------------------------------------------------------
# API chính
# ---------------------------------------------------------------------------
def build_symbol_report(
    symbol: str,
    daily: pd.DataFrame | None = None,
    benchmark: pd.DataFrame | None = None,
    benchmark2: pd.DataFrame | None = None,
    output: str | io.BytesIO = "bao_cao_{symbol}.xlsx",
    source_note: str | None = None,
    upload_warnings: list[str] | None = None,
    exchange: str = "—",
) -> str | io.BytesIO:
    """Tạo báo cáo Excel 3 sheet cho MỘT mã. Trả về đường dẫn hoặc BytesIO đã ghi."""
    symbol = symbol.strip().upper()
    if daily is None:
        daily = fetch_daily_dong(symbol)
    if benchmark is None:
        try:
            benchmark = fetch_benchmark("VNINDEX")
        except Exception:
            benchmark = None

    mtf = analyze_symbol(symbol, daily)
    vs = volume_analysis.volume_score(daily)
    rs_bench = (rs_performance(daily["close"], benchmark["close"])
                if benchmark is not None and not benchmark.empty else None)

    wb = Workbook()
    ws1 = wb.active
    ws1.title = "Danh gia ky thuat"
    _sheet_danh_gia(ws1, symbol, daily, mtf, vs, rs_bench,
                    exchange=exchange, warnings=upload_warnings, source_note=source_note)
    ws2 = wb.create_sheet("So sanh RS & RRG")
    _sheet_rs_rrg(ws2, symbol, daily, benchmark, benchmark2)
    ws3 = wb.create_sheet("Nguon & phuong phap")
    _sheet_nguon(ws3, {
        "source": source_note or "vnstock (VCI)",
        "n_rows": len(daily),
        "date_range": f"{daily.index[0]:%d/%m/%Y} → {daily.index[-1]:%d/%m/%Y}",
    })

    if isinstance(output, str):
        path = output.format(symbol=symbol)
        wb.save(path)
        return path
    wb.save(output)
    output.seek(0)
    return output


def build_multi_report(
    symbols: list[str],
    dailies: dict[str, pd.DataFrame],
    benchmark: pd.DataFrame | None = None,
    output: str | io.BytesIO = "so_sanh.xlsx",
    source_note: str | None = None,
    exchanges: dict[str, str] | None = None,
) -> str | io.BytesIO:
    """Báo cáo SO SÁNH nhiều mã: sheet tổng quan xếp hạng + RRG chung
    + mỗi mã một sheet đánh giá chi tiết + sheet nguồn/phương pháp."""
    symbols = [s.strip().upper() for s in symbols]
    exchanges = exchanges or {}
    if benchmark is None:
        try:
            benchmark = fetch_benchmark("VNINDEX")
        except Exception:
            benchmark = None

    results = []
    for sym in symbols:
        daily = dailies[sym]
        mtf = analyze_symbol(sym, daily)
        vs = volume_analysis.volume_score(daily)
        rs = (rs_performance(daily["close"], benchmark["close"])
              if benchmark is not None and not benchmark.empty else {k: None for k in RS_PERIODS})
        rrg = rrg_weekly(daily, benchmark) if benchmark is not None else None
        quad = (rrg_quadrant(rrg.iloc[-1]["rs_ratio"], rrg.iloc[-1]["rs_momentum"])
                if rrg is not None else "—")
        results.append({"symbol": sym, "daily": daily, "mtf": mtf, "vs": vs,
                        "rs": rs, "rrg": rrg, "quad": quad})

    results.sort(
        key=lambda r_: r_["mtf"].get("Điểm tổng hợp")
        if r_["mtf"].get("Điểm tổng hợp") is not None
        and not np.isnan(r_["mtf"].get("Điểm tổng hợp")) else -999,
        reverse=True,
    )

    wb = Workbook()
    ws = wb.active
    ws.title = "Tong quan so sanh"
    row = _title(ws, 1, f"SO SÁNH KỸ THUẬT {len(symbols)} MÃ",
                 f"Chạy lúc {dt.datetime.now():%d/%m/%Y %H:%M} · sắp theo Điểm tổng hợp giảm dần "
                 "· Tham khảo, không phải khuyến nghị đầu tư.")

    headers = ["Mã", "Giá (đ)", "%1 phiên", "Khung Ngày", "Khung Tuần", "Khung Tháng",
               "Điểm tổng hợp", "Đa khung", "Điểm volume", "RS 1M", "RS 3M", "RS 6M",
               "RS 1Y", "RRG"]
    for j, h in enumerate(headers, start=1):
        _cell(ws, row, j, h, bold=True, color="FFFFFF", bg=NAVY, align="center")
        ws.column_dimensions[get_column_letter(j)].width = 13
    ws.column_dimensions["A"].width = 9
    row += 1

    for r_ in results:
        mtf, vs, rs = r_["mtf"], r_["vs"], r_["rs"]
        j = 1
        _cell(ws, row, j, r_["symbol"], bold=True); j += 1
        _cell(ws, row, j, round(float(r_["daily"]["close"].iloc[-1])), num_fmt="#,##0"); j += 1
        chg = mtf.get("%1D")
        _cell(ws, row, j, chg if chg is not None and not np.isnan(chg) else "—",
              num_fmt="+0.00;-0.00", align="center"); j += 1
        for key in ("Ngày", "Tuần", "Tháng"):
            state = mtf.get(key, "—")
            bg = STATE_COLORS.get(state)
            _cell(ws, row, j, state, bg=bg, color="FFFFFF" if bg else "000000",
                  bold=bool(bg), align="center")
            j += 1
        comp = mtf.get("Điểm tổng hợp")
        _cell(ws, row, j, comp if comp is not None and not np.isnan(comp) else "—",
              num_fmt="+0.0;-0.0", align="center"); j += 1
        _cell(ws, row, j, mtf.get("Đa khung", "—"), align="center"); j += 1
        v_sc = vs.get("volume_score")
        _cell(ws, row, j, v_sc if v_sc is not None and not np.isnan(v_sc) else "—",
              num_fmt="+0.0;-0.0", align="center"); j += 1
        for label in RS_PERIODS:
            v = rs[label]
            _cell(ws, row, j, v if v is not None else "—",
                  bg=_rs_fill(v), num_fmt="+0.0;-0.0", align="center")
            j += 1
        _cell(ws, row, j, r_["quad"],
              bg=RRG_COLORS.get(r_["quad"]), align="center", bold=r_["quad"] in RRG_COLORS)
        row += 1
    row += 1

    # --- RRG chung: mỗi mã một quỹ đạo 8 tuần ---
    with_rrg = [r_ for r_ in results if r_["rrg"] is not None]
    if with_rrg:
        _cell(ws, row, 1, "Dữ liệu quỹ đạo RRG (8 tuần gần nhất mỗi mã)",
              bold=True, color=NAVY, bg=GROUP_BG)
        row += 1
        chart = ScatterChart()
        chart.title = "RRG — so sánh vị thế các mã so với VN-Index"
        chart.x_axis.title = "RS-Ratio (>100 = mạnh hơn chỉ số)"
        chart.y_axis.title = "RS-Momentum (>100 = đang cải thiện)"
        chart.height, chart.width = 12, 20
        for r_ in with_rrg:
            tail = r_["rrg"].tail(8)
            _cell(ws, row, 1, r_["symbol"], bold=True)
            start = row
            for k, (_, rr) in enumerate(tail.iterrows()):
                _cell(ws, row, 2 + 0, round(float(rr["rs_ratio"]), 2), num_fmt="0.00")
                _cell(ws, row, 3, round(float(rr["rs_momentum"]), 2), num_fmt="0.00")
                row += 1
            xs = Reference(ws, min_col=2, min_row=start, max_row=row - 1)
            ys = Reference(ws, min_col=3, min_row=start, max_row=row - 1)
            s = Series(ys, xs, title=r_["symbol"])
            s.marker = Marker(symbol="circle", size=5)
            chart.series.append(s)
        ws.add_chart(chart, f"E{max(4, row - len(with_rrg) * 8)}")
        row += 1

    # --- Sheet chi tiết từng mã + sheet nguồn ---
    for r_ in results:
        ws_d = wb.create_sheet(f"DG {r_['symbol']}"[:31])
        _sheet_danh_gia(ws_d, r_["symbol"], r_["daily"], r_["mtf"], r_["vs"], r_["rs"],
                        exchange=exchanges.get(r_["symbol"], "—"))

    all_rows = [len(r_["daily"]) for r_ in results]
    ws3 = wb.create_sheet("Nguon & phuong phap")
    _sheet_nguon(ws3, {
        "source": source_note or "vnstock (VCI)",
        "n_rows": f"{min(all_rows)}–{max(all_rows)} phiên/mã ({len(symbols)} mã)",
        "date_range": f"{min(r_['daily'].index[0] for r_ in results):%d/%m/%Y} → "
                      f"{max(r_['daily'].index[-1] for r_ in results):%d/%m/%Y}",
    })

    if isinstance(output, str):
        wb.save(output)
        return output
    wb.save(output)
    output.seek(0)
    return output
