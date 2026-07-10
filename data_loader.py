"""Lấy dữ liệu chứng khoán Việt Nam qua thư viện vnstock (nguồn mặc định: VCI).

Mọi hàm ở đây chỉ chịu trách nhiệm lấy và làm sạch dữ liệu thô (trả về
pandas.DataFrame / dict). Toàn bộ phần tính toán chỉ số nằm ở analysis.py,
phần hiển thị nằm ở app.py.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import subprocess
import sys
import time

import pandas as pd

# vnstock tự print thông báo chứa emoji/tiếng Việt trong lúc gọi API. Trên
# Windows, console mặc định (cp1252) không mã hóa được các ký tự đó, khiến
# print bên trong thư viện ném UnicodeEncodeError và làm hỏng luôn lệnh lấy
# dữ liệu. Chuyển stdout/stderr sang UTF-8 trước khi import vnstock để tránh.
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
from vnstock.api.company import Company
from vnstock.api.financial import Finance
from vnstock.api.listing import Listing
from vnstock.api.quote import Quote
from vnstock.api.trading import Trading

import analysis

logger = logging.getLogger(__name__)

# Nguồn dữ liệu: VCI đầy đủ nhất nhưng chặn IP nước ngoài (deploy cloud sẽ lỗi
# ConnectionError). KBS làm nguồn dự phòng — dữ liệu nghèo trường hơn nhưng đủ
# cho giá/bảng giá/danh sách mã. Đặt biến môi trường VNSTOCK_SOURCE=KBS nếu
# muốn ưu tiên KBS ngay từ đầu (ví dụ khi chạy trên server nước ngoài).
DEFAULT_SOURCE = os.environ.get("VNSTOCK_SOURCE", "VCI").upper()
FALLBACK_SOURCES = ["VCI", "KBS"]


def _source_chain(source: str | None) -> list[str]:
    """Danh sách nguồn sẽ thử theo thứ tự. source chỉ định -> chỉ dùng nguồn đó."""
    if source:
        return [source.upper()]
    return [DEFAULT_SOURCE] + [s for s in FALLBACK_SOURCES if s != DEFAULT_SOURCE]


def _try_sources(fn, what: str, source: str | None = None):
    """Chạy fn(source) lần lượt qua các nguồn, trả kết quả đầu tiên thành công."""
    last_exc: Exception | None = None
    for src in _source_chain(source):
        try:
            return fn(src)
        except Exception as exc:
            logger.warning("%s: nguồn %s lỗi (%s) — thử nguồn kế tiếp", what, src, exc)
            last_exc = exc
    raise last_exc if last_exc else RuntimeError(f"{what}: không có nguồn dữ liệu nào")
OHLCV_COLUMNS = ["time", "open", "high", "low", "close", "volume"]

# Các chỉ số chính của thị trường Việt Nam: mã (vnstock) -> tên hiển thị
MARKET_INDICES = {
    "VNINDEX": "VN-Index (HOSE)",
    "VN30": "VN30",
    "HNXINDEX": "HNX-Index",
    "UPCOMINDEX": "UPCOM-Index",
}

# item_id (vnstock) -> nhãn tiếng Việt ngắn gọn để hiển thị
KEY_RATIO_IDS = {
    "pe_ratio": "P/E",
    "pb_ratio": "P/B",
    "ps_ratio": "P/S",
    "roe": "ROE (%)",
    "roa": "ROA (%)",
    "dividend_yield": "Tỷ suất cổ tức (%)",
    "debt_to_equity": "Nợ / Vốn chủ sở hữu",
    "current_ratio": "Khả năng thanh toán hiện hành",
    "gross_margin": "Biên lợi nhuận gộp (%)",
    "net_margin": "Biên lợi nhuận ròng (%)",
}


def get_price_history(
    symbol: str,
    start: str,
    end: str,
    interval: str = "1D",
    source: str | None = None,
) -> pd.DataFrame:
    """Lấy dữ liệu giá lịch sử OHLCV. Trả về DataFrame với cột: time, open, high, low, close, volume.

    vnstock trả về giá theo đơn vị nghìn đồng (vd: 55.3), trong khi phần tổng quan công ty
    (get_company_overview) trả về giá theo đơn vị đồng (vd: 54900.0). Ở đây quy đổi giá về
    đơn vị đồng để nhất quán trong toàn bộ dashboard.

    source=None -> tự thử lần lượt các nguồn (VCI rồi KBS).
    """

    def _fetch(src: str) -> pd.DataFrame:
        quote = Quote(symbol=symbol.upper(), source=src)
        df = quote.history(start=start, end=end, interval=interval)
        if df is None or df.empty:
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        df = df.rename(columns=str.lower)
        df = df.sort_values("time").reset_index(drop=True)
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]] * 1000
        return df[OHLCV_COLUMNS]

    return _try_sources(_fetch, f"Giá lịch sử {symbol}", source)


def get_company_overview(symbol: str, source: str | None = None) -> dict:
    """Lấy thông tin tổng quan công ty: giá hiện tại, vốn hóa, vùng giá 52 tuần,
    tỷ lệ sở hữu nước ngoài, khuyến nghị của công ty chứng khoán, giá mục tiêu...

    Lưu ý: nguồn KBS (dự phòng) trả về ít trường hơn VCI nhiều — phần hiển thị
    phải dùng .get() và chấp nhận N/A.
    """

    def _fetch(src: str) -> dict:
        company = Company(symbol=symbol.upper(), source=src)
        df = company.overview()
        if df is None or df.empty:
            return {}
        return df.iloc[0].to_dict()

    try:
        return _try_sources(_fetch, f"Tổng quan {symbol}", source)
    except Exception:
        return {}


def get_key_ratios(symbol: str, source: str | None = None) -> dict:
    """Lấy một số chỉ số tài chính cơ bản (P/E, P/B, ROE, ROA, biên lợi nhuận...).

    Lưu ý: bảng ratio() của vnstock hiện gắn nhãn kỳ báo cáo không ổn định, nên hàm
    này bỏ qua nhãn cột và chỉ lấy giá trị ở cột cuối cùng (kỳ có dữ liệu gần nhất
    mà nguồn trả về). Vì vậy đây chỉ là số tham khảo, không chắc là quý mới nhất.
    """
    try:
        df = _try_sources(
            lambda src: Finance(
                symbol=symbol.upper(), source=src, period="quarter", get_all=False
            ).ratio(),
            f"Chỉ số tài chính {symbol}",
            source,
        )
    except Exception:
        return {}

    if df is None or df.empty or "item_id" not in df.columns:
        return {}

    data_cols = [c for c in df.columns if c not in ("item", "item_en", "item_id")]
    if not data_cols:
        return {}
    latest_col = data_cols[-1]

    result = {}
    for item_id, label in KEY_RATIO_IDS.items():
        row = df[df["item_id"] == item_id]
        if row.empty:
            continue
        value = row.iloc[0][latest_col]
        if pd.notna(value):
            result[label] = value
    return result


def search_symbols(source: str | None = None) -> pd.DataFrame:
    """Trả về danh sách toàn bộ mã cổ phiếu niêm yết kèm tên công ty."""

    def _fetch(src: str) -> pd.DataFrame:
        df = Listing(source=src).all_symbols()
        return df.sort_values("symbol").reset_index(drop=True)

    return _try_sources(_fetch, "Danh sách mã", source)


def get_index_history(
    index_symbol: str,
    start: str,
    end: str,
    interval: str = "1D",
    source: str | None = None,
) -> pd.DataFrame:
    """Lấy lịch sử một chỉ số thị trường (VNINDEX, VN30, HNXINDEX, UPCOMINDEX...).

    Khác với giá cổ phiếu, chỉ số tính bằng điểm nên giữ nguyên giá trị gốc,
    không quy đổi đơn vị.
    """

    def _fetch(src: str) -> pd.DataFrame:
        quote = Quote(symbol=index_symbol.upper(), source=src)
        df = quote.history(start=start, end=end, interval=interval)
        if df is None or df.empty:
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        df = df.rename(columns=str.lower)
        df = df.sort_values("time").reset_index(drop=True)
        return df[OHLCV_COLUMNS]

    return _try_sources(_fetch, f"Chỉ số {index_symbol}", source)


def get_group_symbols(group: str = "VN30", source: str | None = None) -> list[str]:
    """Danh sách mã cổ phiếu thuộc một rổ chỉ số (vd: VN30, HNX30)."""
    return _try_sources(
        lambda src: list(Listing(source=src).symbols_by_group(group)),
        f"Rổ {group}",
        source,
    )


def _parse_board_vci(raw: pd.DataFrame) -> pd.DataFrame:
    """Bảng giá VCI: cột MultiIndex; accumulated_value theo TRIỆU đồng,
    foreign_*_value theo đồng — quy tất cả về đồng."""
    return pd.DataFrame(
        {
            "symbol": raw[("listing", "symbol")],
            "exchange": raw[("listing", "exchange")],
            "ref_price": raw[("match", "reference_price")],
            "price": raw[("match", "match_price")],
            "high": raw[("match", "highest")],
            "low": raw[("match", "lowest")],
            "volume": raw[("match", "accumulated_volume")],
            "value": raw[("match", "accumulated_value")] * 1e6,
            "foreign_buy_value": raw[("match", "foreign_buy_value")],
            "foreign_sell_value": raw[("match", "foreign_sell_value")],
        }
    )


def _parse_board_kbs(raw: pd.DataFrame) -> pd.DataFrame:
    """Bảng giá KBS: cột phẳng; khối ngoại chỉ có KHỐI LƯỢNG nên giá trị
    mua/bán ròng được ước lượng = khối lượng x giá khớp (ghi chú trong CLAUDE.md)."""
    price = raw["close_price"].where(raw["close_price"] > 0, raw["reference_price"])
    return pd.DataFrame(
        {
            "symbol": raw["symbol"],
            "exchange": raw["exchange"],
            "ref_price": raw["reference_price"],
            "price": raw["close_price"],
            "high": raw["high_price"],
            "low": raw["low_price"],
            "volume": raw["volume_accumulated"],
            "value": raw["total_value"],
            "foreign_buy_value": raw["foreign_buy_volume"] * price,
            "foreign_sell_value": raw["foreign_sell_volume"] * price,
        }
    )


def get_price_board(symbols: list[str], source: str | None = None) -> pd.DataFrame:
    """Bảng giá trực tiếp của một nhóm mã: giá khớp, thay đổi so với tham chiếu,
    khối lượng, giá trị giao dịch, và mua/bán ròng của khối ngoại.

    Đơn vị sau khi chuẩn hóa: giá theo đồng, khối lượng theo cổ phiếu,
    các cột giá trị (value, foreign_*) theo đồng.
    """

    def _fetch(src: str) -> pd.DataFrame:
        raw = Trading(source=src).price_board(symbols_list=symbols)
        if raw is None or raw.empty:
            return pd.DataFrame()
        df = _parse_board_kbs(raw) if src == "KBS" else _parse_board_vci(raw)
        # Mã chưa có giao dịch trong phiên (price = 0) thì lấy giá tham chiếu để tránh -100%
        df["price"] = df["price"].where(df["price"] > 0, df["ref_price"])
        df["change"] = df["price"] - df["ref_price"]
        df["pct_change"] = (df["change"] / df["ref_price"].replace(0, pd.NA)) * 100
        df["foreign_net_value"] = df["foreign_buy_value"] - df["foreign_sell_value"]
        return df.sort_values("pct_change", ascending=False).reset_index(drop=True)

    return _try_sources(_fetch, "Bảng giá", source)


def run_screen(
    watchlist: list[str],
    rsi_min: float = 50.0,
    rsi_max: float = 70.0,
    vol_breakout: float = 1.5,
    max_dist_52w: float = 0.15,
    lookback_days: int = 400,
    sleep_between_calls: float = 0.8,
    progress_callback=None,
    source: str | None = None,
) -> pd.DataFrame:
    """Sàng lọc cả watchlist theo 6 tiêu chí kỹ thuật (xem analysis.score_symbol).

    Trả về DataFrame đã sort theo score giảm dần. Gọi API tuần tự và nghỉ
    sleep_between_calls giữa các mã để tránh bị nguồn dữ liệu chặn — với ~30 mã
    mất khoảng 30-60 giây, nên chạy theo nút bấm + cache, không chạy mỗi lần
    render lại giao diện.

    progress_callback(symbol, i, total): hook để dashboard hiển thị tiến độ.
    """
    end = dt.date.today()
    # +30 ngày bù cho ngày nghỉ/lễ để chắc chắn đủ 200 phiên tính MA200
    start = end - dt.timedelta(days=lookback_days + 30)

    idx = get_index_history("VNINDEX", str(start), str(end), source=source)
    if idx.empty:
        logger.error("Không lấy được VN-Index làm mốc so sánh, dừng screen.")
        return pd.DataFrame()
    index_ret_3m = analysis.return_over(idx["close"])

    rows = []
    total = len(watchlist)
    for i, sym in enumerate(watchlist, 1):
        if progress_callback:
            progress_callback(sym, i, total)
        try:
            df = get_price_history(sym, str(start), str(end), source=source)
            if len(df) < 200:
                logger.warning("%s: thiếu dữ liệu (<200 phiên), bỏ qua", sym)
                continue
            rows.append(
                analysis.score_symbol(
                    sym,
                    df,
                    index_ret_3m,
                    rsi_min=rsi_min,
                    rsi_max=rsi_max,
                    vol_breakout=vol_breakout,
                    max_dist_52w=max_dist_52w,
                )
            )
        except Exception as exc:
            logger.error("%s: lỗi %s", sym, exc)
        time.sleep(sleep_between_calls)

    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    )


def run_screen_subprocess(
    watchlist: list[str],
    rsi_min: float = 50.0,
    rsi_max: float = 70.0,
    vol_breakout: float = 1.5,
    max_dist_52w: float = 0.15,
    progress_callback=None,
    timeout: int = 900,
) -> pd.DataFrame:
    """Chạy run_screen trong tiến trình con (xem lý do trong screen_worker.py).

    Đọc tiến độ trực tiếp từ stdout của worker nên vẫn hiển thị được progress bar.
    """
    params = {
        "watchlist": list(watchlist),
        "rsi_min": rsi_min,
        "rsi_max": rsi_max,
        "vol_breakout": vol_breakout,
        "max_dist_52w": max_dist_52w,
    }
    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screen_worker.py")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen(
        [sys.executable, worker, json.dumps(params)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=os.path.dirname(worker),
        env=env,
    )
    result_json = None
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line.startswith("PROGRESS\t"):
                _, sym, i, total = line.split("\t")
                if progress_callback:
                    progress_callback(sym, int(i), int(total))
            elif line.startswith("RESULT\t"):
                result_json = line.split("\t", 1)[1]
        proc.wait(timeout=timeout)
    finally:
        if proc.poll() is None:
            proc.kill()

    if not result_json:
        logger.error("Worker sàng lọc không trả về kết quả.")
        return pd.DataFrame()
    from io import StringIO

    df = pd.read_json(StringIO(result_json), orient="records")
    return df if not df.empty else pd.DataFrame()
