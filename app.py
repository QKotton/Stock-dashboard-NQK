"""Dashboard học chứng khoán Việt Nam - dữ liệu trực tiếp qua vnstock.

Chạy: streamlit run app.py

Bố cục một màn hình lớn:
- Cột trái : số liệu trực tiếp của thị trường (độ rộng, thanh khoản, khối ngoại),
             các mã đang có tín hiệu đi lên, bảng giá VN30.
- Cột phải : 4 chỉ số thị trường (góc trên) + biểu đồ so sánh ngay bên dưới.
- Bên dưới : chi tiết VN-Index / khối ngoại (thu gọn) và khu phân tích từng cổ phiếu.
"""
from __future__ import annotations

import datetime as dt
import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import analysis
import data_loader

st.set_page_config(page_title="VN Stock Dashboard", layout="wide", page_icon="📈")

# Nguồn dữ liệu: data_loader tự thử VCI trước rồi KBS (VCI chặn IP nước ngoài,
# cần khi deploy cloud). Ưu tiên nguồn khác: đặt biến môi trường VNSTOCK_SOURCE.

# ---------------------------------------------------------------------------
# Giao diện: theme tối kiểu bảng điện tài chính (kết hợp .streamlit/config.toml)
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
/* Thẻ số liệu: dạng card có viền, nền gradient nhẹ */
[data-testid="stMetric"] {
    background: linear-gradient(160deg, #151d31 0%, #10172a 100%);
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 14px;
    padding: 14px 16px 10px 16px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25);
}
[data-testid="stMetricLabel"] p {
    font-size: 0.78rem;
    color: #94a3b8;
    letter-spacing: 0.02em;
}
[data-testid="stMetricValue"] {
    font-size: 1.4rem;
    font-weight: 700;
}
[data-testid="stMetricDelta"] { font-size: 0.85rem; }

/* Tiêu đề */
h1 { font-weight: 800; letter-spacing: -0.5px; }
h2, h3 { font-weight: 700; letter-spacing: -0.3px; }
h4, h5 { font-weight: 650; }

/* Expander: bo góc, viền mảnh */
[data-testid="stExpander"] {
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 12px;
    background: rgba(20, 27, 45, 0.45);
}
[data-testid="stExpander"] summary { font-weight: 600; }

/* Nút */
.stButton > button, .stDownloadButton > button {
    border-radius: 10px;
    font-weight: 600;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1322;
    border-right: 1px solid rgba(148, 163, 184, 0.12);
}

/* Bảng dữ liệu: bo góc khung */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-radius: 12px;
    overflow: hidden;
}

/* Chú thích */
[data-testid="stCaptionContainer"] { color: #7c8698; }

/* Divider mảnh hơn */
hr { border-color: rgba(148, 163, 184, 0.14); }

/* Thu hẹp khoảng trống trên cùng của trang */
.block-container { padding-top: 2.2rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Tải dữ liệu (có cache)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_symbol_list() -> pd.DataFrame:
    return data_loader.search_symbols()


@st.cache_data(ttl=15 * 60, show_spinner=False)
def load_price_history(symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
    return data_loader.get_price_history(symbol, start, end, interval=interval)


@st.cache_data(ttl=15 * 60, show_spinner=False)
def load_index_history(index_symbol: str, start: str, end: str) -> pd.DataFrame:
    return data_loader.get_index_history(index_symbol, start, end)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_overview(symbol: str) -> dict:
    return data_loader.get_company_overview(symbol)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_ratios(symbol: str) -> dict:
    return data_loader.get_key_ratios(symbol)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_group_symbols(group: str) -> list[str]:
    return data_loader.get_group_symbols(group)


@st.cache_data(ttl=5 * 60, show_spinner=False)
def load_price_board(symbols: tuple[str, ...]) -> pd.DataFrame:
    return data_loader.get_price_board(list(symbols))


@st.cache_data(ttl=15 * 60, show_spinner=False)
def run_screen_cached(
    watchlist: tuple[str, ...],
    rsi_min: float,
    rsi_max: float,
    vol_breakout: float,
    max_dist_52w: float,
    _progress_callback=None,
) -> pd.DataFrame:
    """Cache kết quả screen 15 phút. _progress_callback có gạch dưới đầu
    để Streamlit không băm nó vào khóa cache.

    Chạy qua tiến trình con để không đụng hạn mức API (20 lệnh/phút) mà
    dashboard đang dùng cho bảng giá/chỉ số — xem screen_worker.py.
    """
    return data_loader.run_screen_subprocess(
        list(watchlist),
        rsi_min=rsi_min,
        rsi_max=rsi_max,
        vol_breakout=vol_breakout,
        max_dist_52w=max_dist_52w,
        progress_callback=_progress_callback,
    )


def fmt_money(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:,.0f} đ"


def fmt_number(value, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:,.2f}{suffix}"


def fmt_billion(value) -> str:
    """Đổi đồng -> tỷ đồng cho dễ đọc."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value / 1e9:,.1f} tỷ"


def color_pos_neg(v):
    if isinstance(v, (int, float)) and not pd.isna(v):
        if v > 0:
            return "color: #26a69a"
        if v < 0:
            return "color: #ef5350"
    return ""


# ---------------------------------------------------------------------------
# Khung trang
# ---------------------------------------------------------------------------
st.markdown(
    """
<div style="margin-bottom: 0.6rem;">
  <h1 style="margin-bottom: 0.1rem;">📈
    <span style="background: linear-gradient(90deg, #2dd4bf 0%, #60a5fa 100%);
                 -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
      VN Stock Dashboard
    </span>
  </h1>
  <p style="color: #94a3b8; margin: 0; font-size: 0.95rem;">
    Toàn cảnh thị trường chứng khoán Việt Nam · dữ liệu trực tiếp từ VCI · dành cho người đang học nghề
  </p>
</div>
""",
    unsafe_allow_html=True,
)

today = dt.date.today()

# --- Tải dữ liệu thị trường dùng chung cho cả màn hình ---
try:
    with st.spinner("Đang tải bảng giá VN30..."):
        vn30_symbols = load_group_symbols("VN30")
        board = load_price_board(tuple(vn30_symbols))
except Exception as exc:
    board = pd.DataFrame()
    st.warning(f"Không tải được bảng giá VN30: {exc}")

# ===========================================================================
# MÀN HÌNH CHÍNH: TRÁI = số liệu thị trường | PHẢI = chỉ số + so sánh
# ===========================================================================
left_col, right_col = st.columns([0.46, 0.54], gap="large")

# ---------------------------------------------------------------------------
# CỘT PHẢI: chỉ số thị trường (góc trên) + biểu đồ so sánh ngay dưới
# ---------------------------------------------------------------------------
with right_col:
    period_options = {
        "1T": 30,
        "3T": 91,
        "6T": 182,
        "1N": 365,
        "3N": 3 * 365,
    }
    head_l, head_r = st.columns([0.55, 0.45])
    head_l.markdown("#### Chỉ số thị trường")
    period_label = head_r.radio(
        "Khoảng thời gian",
        list(period_options.keys()),
        index=2,
        horizontal=True,
        label_visibility="collapsed",
    )
    market_start = today - dt.timedelta(days=period_options[period_label])

    index_data: dict[str, pd.DataFrame] = {}
    failed_indices: list[str] = []
    with st.spinner("Đang tải dữ liệu các chỉ số..."):
        for idx_symbol, idx_name in data_loader.MARKET_INDICES.items():
            try:
                df = load_index_history(idx_symbol, str(market_start), str(today))
                if not df.empty:
                    index_data[idx_symbol] = df
                else:
                    failed_indices.append(idx_name)
            except Exception:
                failed_indices.append(idx_name)

    if failed_indices:
        st.caption(f"Không tải được: {', '.join(failed_indices)}")

    if not index_data:
        st.error("Không tải được dữ liệu chỉ số nào. Kiểm tra kết nối mạng rồi thử lại.")
    else:
        # 4 thẻ chỉ số xếp thành hàng ngang trên cùng
        metric_cols = st.columns(len(index_data))
        for col, (idx_symbol, df) in zip(metric_cols, index_data.items()):
            last = df["close"].iloc[-1]
            prev = df["close"].iloc[-2] if len(df) > 1 else last
            change = last - prev
            pct = change / prev * 100 if prev else 0
            col.metric(
                data_loader.MARKET_INDICES[idx_symbol],
                f"{last:,.2f}",
                f"{change:+,.2f} ({pct:+.2f}%)",
            )

        # Biểu đồ so sánh ngay bên dưới các thẻ chỉ số
        index_colors = {
            "VNINDEX": "#2dd4bf",
            "VN30": "#60a5fa",
            "HNXINDEX": "#f59e0b",
            "UPCOMINDEX": "#a78bfa",
        }
        cmp_fig = go.Figure()
        for idx_symbol, df in index_data.items():
            base = df["close"].iloc[0]
            cmp_fig.add_trace(
                go.Scatter(
                    x=df["time"],
                    y=(df["close"] / base - 1) * 100,
                    name=data_loader.MARKET_INDICES[idx_symbol],
                    line=dict(width=2.2, color=index_colors.get(idx_symbol)),
                )
            )
        cmp_fig.add_hline(y=0, line_dash="dot", line_color="gray")
        cmp_fig.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title="% thay đổi so với đầu giai đoạn",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(cmp_fig, use_container_width=True)
        st.caption(
            "Mỗi đường là mức tăng/giảm (%) so với đầu giai đoạn — giúp so sánh các chỉ số "
            "có thang điểm khác nhau trên cùng một biểu đồ."
        )

# ---------------------------------------------------------------------------
# CỘT TRÁI: số liệu nhảy của thị trường + các mã có tín hiệu đi lên
# ---------------------------------------------------------------------------
with left_col:
    tl, tr = st.columns([0.7, 0.3])
    tl.markdown("#### Nhịp thị trường (rổ VN30)")
    if tr.button("🔄 Cập nhật", use_container_width=True):
        load_price_board.clear()
        st.rerun()

    # Sau 0h, sàn reset bảng giá chờ phiên mới: giá khớp/khối lượng đều bằng 0
    # cho tới khi mở cửa (9h15). Khi đó các số "tăng/giảm" không có ý nghĩa.
    no_trades_yet = not board.empty and board["volume"].fillna(0).sum() == 0

    if board.empty:
        st.info("Chưa có dữ liệu bảng giá.")
    elif no_trades_yet:
        st.info(
            "Phiên hôm nay chưa mở cửa — bảng giá đã reset chờ phiên mới. "
            "Số liệu sẽ bắt đầu 'nhảy' trong giờ giao dịch (9h15–14h45 các ngày làm việc). "
            "Cột Giá bên dưới đang là giá tham chiếu (giá đóng cửa phiên gần nhất)."
        )
    else:
        gainers = int((board["pct_change"] > 0).sum())
        losers = int((board["pct_change"] < 0).sum())
        unchanged = len(board) - gainers - losers
        total_value = board["value"].sum()
        foreign_net = board["foreign_net_value"].sum()

        b1, b2, b3 = st.columns(3)
        b1.metric("Mã tăng", f"🟢 {gainers}")
        b2.metric("Mã giảm", f"🔴 {losers}")
        b3.metric("Đứng giá", f"⚪ {unchanged}")

        b4, b5 = st.columns(2)
        b4.metric("GT giao dịch VN30", fmt_billion(total_value))
        b5.metric(
            "Khối ngoại (ròng)",
            fmt_billion(foreign_net),
            "mua ròng" if foreign_net >= 0 else "bán ròng",
            delta_color="normal" if foreign_net >= 0 else "inverse",
        )

        # --- Các mã đang có tín hiệu đi lên ---
        st.markdown("##### 🚀 Mã đang có tín hiệu đi lên")
        st.caption(
            "Tiêu chí (tham khảo để học): giá đang tăng so với tham chiếu, "
            "khớp gần mức cao nhất phiên (lực mua còn mạnh); ✓ KN = khối ngoại mua ròng."
        )
        up = board[board["pct_change"] > 0].copy()
        # Đóng/khớp càng gần đỉnh phiên thì lực mua càng chiếm ưu thế
        up["near_high"] = up["price"] >= up["high"] * 0.995
        up["foreign_buying"] = up["foreign_net_value"] > 0
        signals = up[up["near_high"]].sort_values("pct_change", ascending=False)
        if signals.empty:
            # Không mã nào khớp sát đỉnh phiên -> nới tiêu chí, chỉ còn "đang tăng giá"
            st.caption("Hiện không có mã nào khớp gần đỉnh phiên — hiển thị các mã đang tăng giá.")
            signals = up.sort_values("pct_change", ascending=False)

        if signals.empty:
            st.info("Cả rổ VN30 hiện không có mã nào tăng giá.")
        else:
            sig_display = pd.DataFrame(
                {
                    "Mã": signals["symbol"],
                    "Giá (đ)": signals["price"],
                    "%": signals["pct_change"],
                    "Sát đỉnh phiên": signals["near_high"].map(lambda x: "✓" if x else ""),
                    "KN mua ròng": signals["foreign_buying"].map(lambda x: "✓" if x else ""),
                    "GT (tỷ đ)": signals["value"] / 1e9,
                }
            )
            st.dataframe(
                sig_display.style.map(color_pos_neg, subset=["%"]),
                column_config={
                    "Giá (đ)": st.column_config.NumberColumn(format="localized"),
                    "%": st.column_config.NumberColumn(format="%+.2f"),
                    "GT (tỷ đ)": st.column_config.NumberColumn(format="%.1f"),
                },
                hide_index=True,
                use_container_width=True,
                height=min(38 * (len(sig_display) + 1), 260),
            )

    # --- Bảng giá đầy đủ (số liệu nhảy của cả rổ) ---
    if not board.empty:
        st.markdown("##### Bảng giá VN30")
        display = board[
            ["symbol", "price", "change", "pct_change", "volume", "value", "foreign_net_value"]
        ].copy()
        display["value"] = display["value"] / 1e9
        display["foreign_net_value"] = display["foreign_net_value"] / 1e9
        display = display.rename(
            columns={
                "symbol": "Mã",
                "price": "Giá (đ)",
                "change": "+/- (đ)",
                "pct_change": "%",
                "volume": "Khối lượng",
                "value": "GT (tỷ đ)",
                "foreign_net_value": "KN ròng (tỷ đ)",
            }
        )
        st.dataframe(
            display.style.map(color_pos_neg, subset=["+/- (đ)", "%", "KN ròng (tỷ đ)"]),
            column_config={
                "Giá (đ)": st.column_config.NumberColumn(format="localized"),
                "+/- (đ)": st.column_config.NumberColumn(format="%+d"),
                "%": st.column_config.NumberColumn(format="%+.2f"),
                "Khối lượng": st.column_config.NumberColumn(format="localized"),
                "GT (tỷ đ)": st.column_config.NumberColumn(format="%.1f"),
                "KN ròng (tỷ đ)": st.column_config.NumberColumn(format="%+.1f"),
            },
            hide_index=True,
            use_container_width=True,
            height=330,
        )
        st.caption(
            "Ngoài giờ giao dịch (9h00-15h00), bảng thể hiện số chốt của phiên gần nhất. "
            "Bấm 🔄 Cập nhật để lấy số mới."
        )

# ===========================================================================
# CHI TIẾT THÊM (thu gọn để màn hình chính không rối)
# ===========================================================================
with st.expander("📊 VN-Index chi tiết & giao dịch khối ngoại"):
    detail_l, detail_r = st.columns(2)

    if "VNINDEX" in index_data:
        vni = index_data["VNINDEX"]
        vni_fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03
        )
        vni_fig.add_trace(
            go.Candlestick(
                x=vni["time"],
                open=vni["open"],
                high=vni["high"],
                low=vni["low"],
                close=vni["close"],
                name="VN-Index",
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
            ),
            row=1,
            col=1,
        )
        vni_colors = [
            "#26a69a" if c >= o else "#ef5350" for c, o in zip(vni["close"], vni["open"])
        ]
        vni_fig.add_trace(
            go.Bar(x=vni["time"], y=vni["volume"], name="Khối lượng", marker_color=vni_colors),
            row=2,
            col=1,
        )
        vni_fig.update_layout(
            title="VN-Index (nến + khối lượng)",
            height=400,
            xaxis_rangeslider_visible=False,
            showlegend=False,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        detail_l.plotly_chart(vni_fig, use_container_width=True)

    if not board.empty:
        fboard = board.dropna(subset=["foreign_net_value"]).sort_values("foreign_net_value")
        foreign_fig = go.Figure(
            go.Bar(
                x=fboard["symbol"],
                y=fboard["foreign_net_value"] / 1e9,
                marker_color=[
                    "#26a69a" if v >= 0 else "#ef5350" for v in fboard["foreign_net_value"]
                ],
            )
        )
        foreign_fig.update_layout(
            title="Khối ngoại mua/bán ròng từng mã (tỷ đồng)",
            height=400,
            margin=dict(l=10, r=10, t=40, b=10),
            yaxis_title="Tỷ đồng",
        )
        detail_r.plotly_chart(foreign_fig, use_container_width=True)

st.divider()

# ===========================================================================
# SÀNG LỌC CỔ PHIẾU (tích hợp từ vn-stock-screener_1)
# ===========================================================================
st.header("🎯 Sàng lọc cổ phiếu theo tín hiệu kỹ thuật")
st.caption(
    "Chấm điểm mỗi mã theo 6 tiêu chí (mỗi tiêu chí 1 điểm). Điểm càng cao, "
    "càng nhiều tín hiệu kỹ thuật tích cực cùng lúc. "
    "⚠️ Đây là công cụ sàng lọc phục vụ phân tích và học tập, không phải khuyến nghị đầu tư."
)

default_watchlist = ", ".join(vn30_symbols) if not board.empty else (
    "FPT, MWG, HPG, VCB, TCB, MBB, ACB, STB, SSI, VND, VCI, HCM, "
    "VNM, MSN, PNJ, DGC, REE, GMD, PVS, DPM"
)

with st.expander("⚙️ Cấu hình bộ lọc", expanded=False):
    raw_watchlist = st.text_area(
        "Danh sách mã cần quét (cách nhau bởi dấu phẩy)",
        value=default_watchlist,
        height=80,
        help="Mặc định là rổ VN30. Bạn có thể thêm/bớt mã tùy ý.",
    )
    sc1, sc2, sc3 = st.columns(3)
    scr_rsi_min, scr_rsi_max = sc1.slider(
        "Vùng RSI chấp nhận", 0, 100, (50, 70),
        help="RSI trong vùng này = đủ khỏe nhưng chưa quá mua.",
    )
    scr_vol_breakout = sc2.slider(
        "Ngưỡng volume đột biến (x TB 20 phiên)", 1.0, 3.0, 1.5, 0.1,
        help="Khối lượng phiên gần nhất phải gấp bao nhiêu lần trung bình 20 phiên.",
    )
    scr_max_dist = sc3.slider(
        "Cách đỉnh 52 tuần tối đa (%)", 5, 40, 15,
        help="Cổ phiếu khỏe thường không rơi quá xa đỉnh 52 tuần.",
    ) / 100

screen_watchlist = tuple(
    s.strip().upper() for s in raw_watchlist.split(",") if s.strip()
)

run_col, note_col = st.columns([0.25, 0.75])
run_clicked = run_col.button("▶️ Chạy sàng lọc", type="primary", use_container_width=True)
note_col.caption(
    f"Sẽ quét {len(screen_watchlist)} mã, mất khoảng "
    f"{round(len(screen_watchlist) * 3.5 / 60)}–{round(len(screen_watchlist) * 5 / 60)} phút "
    "(nguồn miễn phí giới hạn 20 lệnh gọi/phút nên phải quét chậm rãi). "
    "Kết quả được nhớ 15 phút."
)

if run_clicked:
    progress_bar = st.progress(0.0, text="Chuẩn bị...")

    def _update_progress(sym: str, i: int, total: int) -> None:
        progress_bar.progress(i / total, text=f"Đang quét {sym} ({i}/{total})...")

    screen_df = run_screen_cached(
        screen_watchlist,
        float(scr_rsi_min),
        float(scr_rsi_max),
        scr_vol_breakout,
        scr_max_dist,
        _progress_callback=_update_progress,
    )
    progress_bar.empty()
    st.session_state["screen_result"] = screen_df
    st.session_state["screen_time"] = dt.datetime.now().strftime("%H:%M:%S")

if "screen_result" in st.session_state:
    screen_df = st.session_state["screen_result"]
    if screen_df.empty:
        st.error("Không lấy được dữ liệu sàng lọc. Kiểm tra kết nối hoặc thử lại sau.")
    else:
        candidates = screen_df[screen_df["score"] >= 5]["symbol"].tolist()
        sm1, sm2, sm3 = st.columns(3)
        sm1.metric("Số mã quét được", len(screen_df))
        sm2.metric("Đạt ≥ 5/6 tiêu chí", len(candidates))
        sm3.metric("Ứng viên nổi bật", ", ".join(candidates) if candidates else "Không có")
        st.caption(f"Kết quả lúc {st.session_state.get('screen_time', 'N/A')}.")

        screen_display = screen_df.rename(
            columns={
                "symbol": "Mã",
                "price": "Giá (đ)",
                "rsi": "RSI",
                "vol_ratio": "Vol/TB20",
                "dist_52w_high": "Cách đỉnh 52w",
                "rs_vs_index_3m": "RS vs VN-Index",
                "trend_ok": "Xu hướng",
                "rsi_ok": "RSI ok",
                "macd_ok": "MACD",
                "volume_ok": "Volume",
                "near_high_ok": "Gần đỉnh",
                "rs_ok": "RS",
                "score": "Điểm",
            }
        ).drop(columns=["max_score"], errors="ignore")

        st.dataframe(
            screen_display.style.background_gradient(
                subset=["Điểm"], cmap="RdYlGn", vmin=0, vmax=6
            ),
            column_config={
                "Giá (đ)": st.column_config.NumberColumn(format="localized"),
                "RSI": st.column_config.NumberColumn(format="%.1f"),
                "Vol/TB20": st.column_config.NumberColumn(format="%.2f"),
                "Cách đỉnh 52w": st.column_config.NumberColumn(format="percent"),
                "RS vs VN-Index": st.column_config.NumberColumn(format="percent"),
            },
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "⬇️ Tải CSV",
            screen_display.to_csv(index=False).encode("utf-8-sig"),
            "ket_qua_sang_loc.csv",
            "text/csv",
        )

        with st.expander("📘 6 tiêu chí sàng lọc nghĩa là gì?"):
            st.markdown(
                """
| Tiêu chí | Điều kiện | Ý nghĩa khi đạt |
|---|---|---|
| **Xu hướng** | Giá > MA50 > MA200 | Xu hướng tăng cả trung và dài hạn |
| **RSI ok** | RSI trong vùng bạn chọn (mặc định 50–70) | Đà tăng đủ khỏe nhưng chưa vào vùng quá mua |
| **MACD** | Đường MACD nằm trên đường signal | Động lượng ngắn hạn đang dương |
| **Volume** | KL phiên gần nhất ≥ ngưỡng × TB 20 phiên | Dòng tiền đang chú ý đột biến |
| **Gần đỉnh** | Giá cách đỉnh 52 tuần dưới ngưỡng chọn | Cổ phiếu mạnh thường bám sát đỉnh cũ |
| **RS** | 3 tháng qua tăng mạnh hơn VN-Index | Khỏe hơn mặt bằng chung thị trường |

Điểm 5–6/6 nghĩa là nhiều tín hiệu tích cực **cùng lúc** — đáng để đưa vào danh sách
tìm hiểu tiếp (đọc báo cáo tài chính, tin tức...), chứ không phải lệnh mua.
                """
            )

st.divider()

# ===========================================================================
# PHÂN TÍCH TỪNG CỔ PHIẾU
# ===========================================================================
st.header("🔍 Phân tích cổ phiếu")

# --- Sidebar: bộ lọc cho khu phân tích cổ phiếu ---
st.sidebar.title("Bộ lọc (Phân tích cổ phiếu)")

with st.sidebar.expander("Tìm mã theo tên công ty"):
    try:
        symbol_df = load_symbol_list()
        keyword = st.text_input("Nhập từ khóa (vd: sữa, thép, ngân hàng...)")
        if keyword:
            matches = symbol_df[
                symbol_df["organ_name"].str.contains(keyword, case=False, na=False)
                | symbol_df["symbol"].str.contains(keyword, case=False, na=False)
            ].head(30)
            st.dataframe(matches, hide_index=True, use_container_width=True)
    except Exception as exc:
        st.caption(f"Không tải được danh sách mã: {exc}")

symbol = st.sidebar.text_input("Mã cổ phiếu", value="VNM").strip().upper()

default_start = today - dt.timedelta(days=365)
date_range = st.sidebar.date_input(
    "Khoảng thời gian",
    value=(default_start, today),
    max_value=today,
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = default_start, today

interval = st.sidebar.selectbox("Khung thời gian nến", ["1D", "1W", "1M"], index=0)
ma_windows = st.sidebar.multiselect(
    "Đường trung bình động (MA)", [10, 20, 50, 100, 200], default=[20, 50]
)
rsi_period = st.sidebar.slider("Chu kỳ RSI", min_value=5, max_value=30, value=14)

st.sidebar.caption(
    "Dữ liệu lấy trực tiếp từ nguồn VCI qua thư viện `vnstock`. "
    "Ứng dụng này chỉ phục vụ mục đích học tập, không phải khuyến nghị đầu tư."
)

if not symbol:
    st.info("Nhập mã cổ phiếu ở thanh bên trái để bắt đầu (vd: VNM, FPT, HPG, VCB...).")
    st.stop()

with st.spinner(f"Đang tải dữ liệu {symbol}..."):
    try:
        price_df = load_price_history(symbol, str(start_date), str(end_date), interval)
    except Exception as exc:
        st.error(f"Không lấy được dữ liệu giá cho mã '{symbol}'. Chi tiết lỗi: {exc}")
        st.stop()

if price_df.empty:
    st.warning(
        f"Không có dữ liệu cho mã '{symbol}' trong khoảng thời gian đã chọn. "
        "Kiểm tra lại mã cổ phiếu hoặc chọn khoảng thời gian khác."
    )
    st.stop()

overview = load_overview(symbol)
ratios = load_ratios(symbol)
enriched = analysis.enrich(price_df, ma_windows=tuple(ma_windows) if ma_windows else (20, 50))
enriched = analysis.compute_rsi(enriched, period=rsi_period)
stats = analysis.summary_stats(price_df)

# --- Header: tên công ty + giá hiện tại ---
company_name = overview.get("organ_short_name") or overview.get("organ_name") or symbol
st.subheader(f"{symbol} — {company_name}")

header_cols = st.columns(5)
header_cols[0].metric(
    "Giá đóng cửa gần nhất",
    fmt_money(stats.get("last_close")),
    f"{stats.get('change', 0):+,.0f} đ ({stats.get('pct_change', 0):+.2f}%)",
)
header_cols[1].metric("Khối lượng phiên gần nhất", f"{stats.get('last_volume', 0):,.0f}")
header_cols[2].metric("Cao nhất giai đoạn", fmt_money(stats.get("period_high")))
header_cols[3].metric("Thấp nhất giai đoạn", fmt_money(stats.get("period_low")))
header_cols[4].metric(
    "Biến động (năm hóa)", fmt_number(stats.get("annualized_volatility_pct"), "%")
)

# --- Tổng quan công ty ---
if overview:
    with st.expander("Tổng quan công ty", expanded=True):
        cols = st.columns(4)
        cols[0].metric("Vốn hóa", fmt_money(overview.get("market_cap")))
        cols[1].metric("Cao nhất 52 tuần", fmt_money(overview.get("highest_price1_year")))
        cols[2].metric("Thấp nhất 52 tuần", fmt_money(overview.get("lowest_price1_year")))
        foreign_pct = overview.get("foreigner_percentage")
        cols[3].metric(
            "Sở hữu nước ngoài",
            fmt_number(foreign_pct * 100 if foreign_pct is not None else None, "%"),
        )

        cols2 = st.columns(4)
        cols2[0].metric("Ngành", overview.get("sector") or "N/A")
        cols2[1].metric("Khuyến nghị (CTCK)", overview.get("rating") or "N/A")
        cols2[2].metric("Giá mục tiêu", fmt_money(overview.get("target_price")))
        upside = overview.get("upside_to_target_percent")
        cols2[3].metric(
            "Tiềm năng tăng giá",
            fmt_number(upside * 100 if upside is not None else None, "%"),
        )

        profile = overview.get("company_profile")
        if profile:
            st.caption(profile[:600] + ("..." if len(profile) > 600 else ""))

# --- Chỉ số tài chính cơ bản ---
if ratios:
    with st.expander("Chỉ số tài chính cơ bản"):
        st.caption(
            "Số liệu lấy từ kỳ báo cáo gần nhất mà nguồn dữ liệu miễn phí trả về, "
            "có thể chưa phải quý mới nhất — chỉ nên dùng để tham khảo, làm quen."
        )
        ratio_cols = st.columns(len(ratios))
        for col, (label, value) in zip(ratio_cols, ratios.items()):
            col.metric(label, fmt_number(value))

# ===========================================================================
# PHÂN TÍCH KỸ THUẬT CHUYÊN SÂU
# ===========================================================================
st.markdown("### 🔬 Phân tích kỹ thuật")

# --- A. Trạng thái đa khung thời gian (ngày / tuần / tháng) ---
st.markdown("##### Trạng thái đa khung thời gian")
st.caption(
    "Cùng một cổ phiếu có thể tăng ở khung ngày nhưng vẫn giảm ở khung tuần/tháng. "
    "Nhìn cả 3 khung giúp tránh 'thấy cây mà không thấy rừng'. "
    "Xu hướng xét theo giá so với MA10 và MA20 của từng khung."
)

TF_FRAMES = {
    "1D": ("Khung NGÀY", 300),
    "1W": ("Khung TUẦN", 750),
    "1M": ("Khung THÁNG", 1900),
}

tf_status: dict[str, dict] = {}
with st.spinner("Đang phân tích 3 khung thời gian..."):
    for tf_interval, (tf_label, lookback) in TF_FRAMES.items():
        try:
            tf_df = load_price_history(
                symbol, str(today - dt.timedelta(days=lookback)), str(today), tf_interval
            )
            tf_status[tf_interval] = analysis.frame_status(tf_df)
        except Exception:
            tf_status[tf_interval] = {}


def tf_card_html(title: str, s: dict) -> str:
    base_style = (
        "flex:1; background:linear-gradient(160deg,#151d31,#10172a);"
        "border:1px solid rgba(148,163,184,.16); border-radius:14px; padding:14px 16px;"
    )
    if not s:
        return (
            f'<div style="{base_style}">'
            f'<div style="color:#94a3b8;font-size:.78rem;margin-bottom:6px;">{title}</div>'
            f'<div style="color:#cbd5e1;font-size:.9rem;">Không đủ dữ liệu</div></div>'
        )
    trend_map = {
        "up": ("📈", "Xu hướng tăng"),
        "down": ("📉", "Xu hướng giảm"),
        "side": ("➡️", "Đi ngang"),
    }
    rsi_map = {"overbought": "quá mua", "oversold": "quá bán", "neutral": "trung tính"}
    t_icon, t_label = trend_map[s["trend"]]
    macd_label = "trên signal (tích cực)" if s["macd_ok"] else "dưới signal (tiêu cực)"
    vol_label = "sôi động" if s["vol_ratio"] >= 1.2 else ("trầm lắng" if s["vol_ratio"] <= 0.8 else "bình thường")
    return (
        f'<div style="{base_style}">'
        f'<div style="color:#94a3b8;font-size:.78rem;letter-spacing:.02em;margin-bottom:6px;">{title}</div>'
        f'<div style="font-size:1.05rem;font-weight:700;margin-bottom:8px;">{t_icon} {t_label}</div>'
        f'<div style="font-size:.85rem;color:#cbd5e1;line-height:1.75;">'
        f"RSI {s['rsi']:.0f} · {rsi_map[s['rsi_state']]}<br>"
        f"MACD {macd_label}<br>"
        f"Khối lượng ×{s['vol_ratio']:.2f} TB20 · {vol_label}"
        f"</div></div>"
    )


st.markdown(
    '<div style="display:flex; gap:12px; margin-bottom:10px;">'
    + "".join(
        tf_card_html(label, tf_status.get(tf, {}))
        for tf, (label, _) in TF_FRAMES.items()
    )
    + "</div>",
    unsafe_allow_html=True,
)

trends = [s.get("trend") for s in tf_status.values() if s]
if len(trends) == 3 and all(t == "up" for t in trends):
    st.success("✅ Cả 3 khung thời gian cùng xu hướng TĂNG — xu hướng đang đồng thuận mạnh.")
elif len(trends) == 3 and all(t == "down" for t in trends):
    st.error("🔻 Cả 3 khung thời gian cùng xu hướng GIẢM — áp lực bán trên mọi khung.")
elif trends:
    st.info(
        "ℹ️ Các khung thời gian chưa đồng thuận — thị trường đang trong giai đoạn chuyển tiếp "
        "hoặc điều chỉnh. Người mới nên thận trọng khi tín hiệu các khung mâu thuẫn nhau."
    )

# --- B. Biểu đồ kỹ thuật hợp nhất: giá + volume + RSI + MACD ---
st.markdown("##### Biểu đồ kỹ thuật (theo khung đã chọn ở thanh bên)")

# Bảng màu series (đã kiểm định cho nền tối): MA lần lượt xanh dương/vàng/tím/ngọc/hồng
MA_COLORS = ["#3987e5", "#c98500", "#9085e9", "#199e70", "#d55181"]

tech_fig = make_subplots(
    rows=4,
    cols=1,
    shared_xaxes=True,
    row_heights=[0.48, 0.14, 0.19, 0.19],
    vertical_spacing=0.03,
)

# Hàng 1: nến + dải Bollinger + các đường MA
tech_fig.add_trace(
    go.Scatter(
        x=enriched["time"], y=enriched["bb_upper"], name="Bollinger",
        line=dict(width=1, color="rgba(57,135,229,0.35)"), showlegend=False,
    ),
    row=1, col=1,
)
tech_fig.add_trace(
    go.Scatter(
        x=enriched["time"], y=enriched["bb_lower"], name="Bollinger (20, ±2σ)",
        line=dict(width=1, color="rgba(57,135,229,0.35)"),
        fill="tonexty", fillcolor="rgba(57,135,229,0.07)",
    ),
    row=1, col=1,
)
tech_fig.add_trace(
    go.Candlestick(
        x=enriched["time"],
        open=enriched["open"], high=enriched["high"],
        low=enriched["low"], close=enriched["close"],
        name=symbol,
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ),
    row=1, col=1,
)
for i, w in enumerate(ma_windows):
    col_name = f"sma_{w}"
    if col_name in enriched.columns:
        tech_fig.add_trace(
            go.Scatter(
                x=enriched["time"], y=enriched[col_name], name=f"MA{w}",
                line=dict(width=1.6, color=MA_COLORS[i % len(MA_COLORS)]),
            ),
            row=1, col=1,
        )

# Hàng 2: khối lượng (màu theo phiên tăng/giảm) + trung bình 20 phiên
volume_colors = [
    "#26a69a" if c >= o else "#ef5350" for c, o in zip(enriched["close"], enriched["open"])
]
tech_fig.add_trace(
    go.Bar(x=enriched["time"], y=enriched["volume"], name="Khối lượng",
           marker_color=volume_colors, showlegend=False),
    row=2, col=1,
)
vol_ma20 = enriched["volume"].rolling(20).mean()
tech_fig.add_trace(
    go.Scatter(x=enriched["time"], y=vol_ma20, name="KL TB20",
               line=dict(width=1.6, color="#3987e5")),
    row=2, col=1,
)

# Hàng 3: RSI + vùng 30-70
tech_fig.add_trace(
    go.Scatter(x=enriched["time"], y=enriched["rsi"], name=f"RSI({rsi_period})",
               line=dict(width=1.8, color="#9085e9")),
    row=3, col=1,
)
tech_fig.add_hline(y=70, line_dash="dot", line_color="rgba(239,83,80,0.55)", row=3, col=1)
tech_fig.add_hline(y=30, line_dash="dot", line_color="rgba(38,166,154,0.55)", row=3, col=1)
tech_fig.add_hrect(y0=30, y1=70, fillcolor="rgba(148,163,184,0.05)", line_width=0, row=3, col=1)

# Hàng 4: MACD (đường + signal + histogram)
macd_hist_colors = [
    "rgba(38,166,154,0.65)" if v >= 0 else "rgba(239,83,80,0.65)"
    for v in enriched["macd_hist"].fillna(0)
]
tech_fig.add_trace(
    go.Bar(x=enriched["time"], y=enriched["macd_hist"], name="Histogram",
           marker_color=macd_hist_colors, showlegend=False),
    row=4, col=1,
)
tech_fig.add_trace(
    go.Scatter(x=enriched["time"], y=enriched["macd"], name="MACD",
               line=dict(width=1.8, color="#3987e5")),
    row=4, col=1,
)
tech_fig.add_trace(
    go.Scatter(x=enriched["time"], y=enriched["macd_signal"], name="Signal",
               line=dict(width=1.6, color="#c98500")),
    row=4, col=1,
)

tech_fig.update_layout(
    height=860,
    xaxis_rangeslider_visible=False,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.01),
    margin=dict(l=10, r=10, t=30, b=10),
)
tech_fig.update_yaxes(title_text="Giá (đ)", row=1, col=1)
tech_fig.update_yaxes(title_text="KL", row=2, col=1)
tech_fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
tech_fig.update_yaxes(title_text="MACD", row=4, col=1)
st.plotly_chart(tech_fig, use_container_width=True)

# --- C. Phân tích dòng tiền (volume) ---
st.markdown("##### Phân tích dòng tiền (khối lượng)")
vstats = analysis.volume_stats(price_df)
if vstats:
    vcol1, vcol2, vcol3, vcol4 = st.columns(4)
    ratio = vstats["vol_ratio"]
    vcol1.metric(
        "KL phiên gần nhất so với TB20",
        f"×{ratio:.2f}",
        "sôi động" if ratio >= 1.2 else ("trầm lắng" if ratio <= 0.8 else "bình thường"),
        delta_color="normal" if ratio >= 1 else "inverse",
    )
    vcol2.metric(
        "Phiên tích lũy (20 phiên)",
        f"{vstats['accum_days']} phiên",
        help="Phiên TĂNG giá kèm khối lượng cao hơn trung bình — dấu hiệu tiền vào.",
    )
    vcol3.metric(
        "Phiên phân phối (20 phiên)",
        f"{vstats['distrib_days']} phiên",
        help="Phiên GIẢM giá kèm khối lượng cao hơn trung bình — dấu hiệu tiền ra.",
    )
    vcol4.metric(
        "OBV (dòng tiền lũy kế)",
        "Đang tăng 📈" if vstats["obv_rising"] else "Đang giảm 📉",
        help="OBV cộng dồn khối lượng theo hướng giá. OBV tăng cùng giá = xu hướng được khối lượng xác nhận.",
    )

    obv_series = analysis.compute_obv(price_df)
    obv_fig = go.Figure()
    obv_fig.add_trace(
        go.Scatter(x=price_df["time"], y=obv_series, name="OBV",
                   line=dict(width=1.8, color="#199e70"))
    )
    obv_fig.add_trace(
        go.Scatter(x=price_df["time"], y=obv_series.rolling(20).mean(), name="OBV TB20",
                   line=dict(width=1.4, color="rgba(148,163,184,0.6)", dash="dot"))
    )
    obv_fig.update_layout(
        height=240,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
        yaxis_title="OBV",
    )
    st.plotly_chart(obv_fig, use_container_width=True)
    st.caption(
        "Đọc nhanh: giá tăng + OBV tăng = xu hướng khỏe; giá tăng nhưng OBV giảm = "
        "tăng thiếu khối lượng đỡ, dễ đảo chiều (phân kỳ). Nhiều phiên tích lũy hơn "
        "phân phối cho thấy dòng tiền vẫn đang ở lại cổ phiếu."
    )
else:
    st.caption("Không đủ dữ liệu để phân tích khối lượng (cần tối thiểu ~21 phiên).")

# --- Lợi nhuận & biến động ---
st.markdown("### Lợi nhuận & biến động")
ret_cols = st.columns(2)

returns_fig = go.Figure()
returns_fig.add_trace(
    go.Scatter(x=enriched["time"], y=enriched["cumulative_return"], name="Lợi nhuận lũy kế (%)")
)
returns_fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="%")
ret_cols[0].plotly_chart(returns_fig, use_container_width=True)

hist_fig = go.Figure()
hist_fig.add_trace(
    go.Histogram(x=enriched["daily_return"].dropna(), nbinsx=40, name="Tỷ suất sinh lời ngày (%)")
)
hist_fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="%")
ret_cols[1].plotly_chart(hist_fig, use_container_width=True)

stat_cols = st.columns(4)
stat_cols[0].metric("Lợi nhuận TB/ngày", fmt_number(stats.get("mean_daily_return_pct"), "%"))
stat_cols[1].metric("Biến động (năm hóa)", fmt_number(stats.get("annualized_volatility_pct"), "%"))
stat_cols[2].metric("Sụt giảm tối đa (drawdown)", fmt_number(stats.get("max_drawdown_pct"), "%"))
stat_cols[3].metric("Khối lượng TB", f"{stats.get('avg_volume', 0):,.0f}")

# ===========================================================================
# XUẤT BÁO CÁO EXCEL (theo HUONG_DAN_XUAT_EXCEL.md)
# ===========================================================================
st.markdown("### 📊 Xuất báo cáo Excel")
st.caption(
    "Tạo file Excel đánh giá kỹ thuật chuyên nghiệp: trạng thái đa khung, chỉ báo "
    "kỹ thuật + volume, so sánh sức mạnh RS/RRG với VN-Index. Nhập NHIỀU mã (cách "
    "nhau dấu phẩy) để nhận thêm sheet so sánh xếp hạng giữa các mã. "
    "⚠️ Báo cáo mang tính tham khảo, không phải khuyến nghị đầu tư."
)

exp_col1, exp_col2 = st.columns([0.5, 0.5])
report_symbols_raw = exp_col1.text_input(
    "Mã cần phân tích (1 hoặc nhiều, cách nhau dấu phẩy)",
    value=symbol,
    help="Ví dụ: ACB hoặc ACB, FPT, HPG — mã bất kỳ trên HOSE/HNX/UPCoM.",
)
uploaded_file = exp_col2.file_uploader(
    "Hoặc upload file OHLCV (CSV/XLSX) cho MỘT mã",
    type=["csv", "xlsx"],
    help="Cột tối thiểu: time/date, open, high, low, close, volume. "
    "Ngày dạng dd/mm/yyyy hoặc yyyy-mm-dd.",
)

report_symbols = [s.strip().upper() for s in report_symbols_raw.split(",") if s.strip()]

if st.button("📊 Xuất báo cáo Excel", type="primary"):
    import excel_report

    if "report_ohlcv" not in st.session_state:
        st.session_state["report_ohlcv"] = {}
    ohlcv_cache: dict = st.session_state["report_ohlcv"]
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")

    try:
        # Benchmark VN-Index dùng chung, cache trong session để không gọi API lần hai
        if "report_benchmark" not in st.session_state:
            with st.spinner("Đang tải VN-Index làm chuẩn so sánh..."):
                st.session_state["report_benchmark"] = excel_report.fetch_benchmark("VNINDEX")
        benchmark_df = st.session_state["report_benchmark"]

        # Sàn niêm yết (nếu có sẵn từ bảng giá VN30, không gọi thêm API)
        exchange_map = (
            dict(zip(board["symbol"], board["exchange"])) if not board.empty else {}
        )

        if uploaded_file is not None:
            # Đường nhập (b): file người dùng upload — một mã
            raw = (
                pd.read_csv(uploaded_file)
                if uploaded_file.name.lower().endswith(".csv")
                else pd.read_excel(uploaded_file)
            )
            norm_df, upload_warns = excel_report.normalize_ohlcv(raw)
            label = report_symbols[0] if report_symbols else "UPLOAD"
            with st.spinner("Đang dựng báo cáo từ file upload..."):
                buf = excel_report.build_symbol_report(
                    label,
                    daily=norm_df,
                    benchmark=benchmark_df,
                    output=io.BytesIO(),
                    source_note=f"file người dùng upload: {uploaded_file.name}",
                    upload_warnings=upload_warns,
                )
            st.session_state["report_bytes"] = buf.getvalue()
            st.session_state["report_name"] = f"bao_cao_{label}_{stamp}.xlsx"
            for w in upload_warns:
                st.warning(w)
        elif not report_symbols:
            st.warning("Nhập ít nhất một mã hoặc upload file dữ liệu.")
            st.stop()
        else:
            # Đường nhập (a): fetch qua vnstock, cache OHLCV theo mã trong session_state
            fetch_bar = st.progress(0.0, text="Chuẩn bị...")
            dailies: dict[str, pd.DataFrame] = {}
            failed: list[str] = []
            for i, sym in enumerate(report_symbols, 1):
                fetch_bar.progress(i / len(report_symbols), text=f"Dữ liệu {sym} ({i}/{len(report_symbols)})...")
                try:
                    if sym not in ohlcv_cache:
                        ohlcv_cache[sym] = excel_report.fetch_daily_dong(sym)
                    dailies[sym] = ohlcv_cache[sym]
                except Exception:
                    failed.append(sym)
            fetch_bar.empty()
            if failed:
                st.warning(
                    f"Không lấy được dữ liệu cho: {', '.join(failed)} — kiểm tra lại mã "
                    "(có thể mã không tồn tại hoặc API tạm lỗi)."
                )
            if not dailies:
                st.error("Không có mã nào lấy được dữ liệu — chưa tạo được báo cáo.")
                st.stop()

            ok_symbols = list(dailies.keys())
            with st.spinner("Đang dựng file Excel..."):
                if len(ok_symbols) == 1:
                    sym = ok_symbols[0]
                    buf = excel_report.build_symbol_report(
                        sym,
                        daily=dailies[sym],
                        benchmark=benchmark_df,
                        output=io.BytesIO(),
                        exchange=exchange_map.get(sym, "—"),
                    )
                    st.session_state["report_name"] = f"bao_cao_{sym}_{stamp}.xlsx"
                else:
                    buf = excel_report.build_multi_report(
                        ok_symbols,
                        dailies,
                        benchmark=benchmark_df,
                        output=io.BytesIO(),
                        exchanges=exchange_map,
                    )
                    st.session_state["report_name"] = (
                        f"so_sanh_{len(ok_symbols)}ma_{stamp}.xlsx"
                    )
            st.session_state["report_bytes"] = buf.getvalue()
        st.success("Đã tạo xong báo cáo — bấm nút bên dưới để tải về.")
    except ValueError as exc:
        st.error(f"Dữ liệu không hợp lệ: {exc}")
    except Exception as exc:
        st.error(f"Không tạo được báo cáo: {exc}")

if st.session_state.get("report_bytes"):
    st.download_button(
        f"⬇️ Tải {st.session_state['report_name']}",
        data=st.session_state["report_bytes"],
        file_name=st.session_state["report_name"],
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# --- Giải thích chỉ số cho người mới ---
with st.expander("📘 Giải thích các chỉ số (dành cho người mới học)"):
    st.markdown(
        """
- **MA (Moving Average)**: giá đóng cửa trung bình trong N phiên gần nhất, giúp làm mượt biến động
  và thấy xu hướng. Giá cắt lên trên MA thường được xem là tín hiệu tích cực, cắt xuống là tiêu cực.
- **RSI (Relative Strength Index)**: đo tốc độ và mức độ thay đổi giá, dao động 0-100.
  Trên 70 thường gọi là "quá mua", dưới 30 là "quá bán" — đây là kinh nghiệm tham khảo, không phải quy luật.
- **MACD**: hiệu giữa 2 đường EMA (12 và 26 phiên). MACD cắt lên trên đường Signal thường được
  xem là tín hiệu động lượng tích cực; histogram (cột) thể hiện khoảng cách giữa 2 đường —
  cột cao dần nghĩa là động lượng đang mạnh lên.
- **Dải Bollinger**: MA20 ± 2 độ lệch chuẩn. Dải co hẹp = giá đang tích lũy, thường trước một
  biến động mạnh; giá bám mép trên dải = xu hướng tăng mạnh (không tự động là "quá mua").
- **OBV (On-Balance Volume)**: cộng dồn khối lượng theo hướng giá — kiểm tra xem xu hướng giá
  có được "tiền thật" ủng hộ không. Giá tăng mà OBV giảm là phân kỳ đáng cảnh giác.
- **Phân tích đa khung**: xu hướng khung NGÀY cho tín hiệu vào/ra ngắn hạn, khung TUẦN/THÁNG
  cho bối cảnh lớn. Nguyên tắc phổ biến: giao dịch thuận theo khung lớn.
- **Lợi nhuận lũy kế**: nếu mua ở đầu giai đoạn và giữ tới nay thì lời/lỗ bao nhiêu phần trăm.
- **Biến động (volatility, năm hóa)**: độ dao động của giá, quy đổi ra thang một năm. Số càng lớn,
  giá càng "nhảy" mạnh, rủi ro ngắn hạn càng cao.
- **Sụt giảm tối đa (Max Drawdown)**: mức giảm sâu nhất tính từ đỉnh gần nhất trong giai đoạn xem —
  thước đo rủi ro "tệ nhất từng xảy ra".
- **P/E (Price/Earnings)**: giá cổ phiếu đang cao gấp bao nhiêu lần lợi nhuận một cổ phần. P/E cao có
  thể do thị trường kỳ vọng tăng trưởng tốt, hoặc đơn giản là cổ phiếu đang đắt.
- **P/B (Price/Book)**: giá so với giá trị sổ sách (tài sản ròng) mỗi cổ phần.
- **ROE / ROA**: hiệu quả sinh lời trên vốn chủ sở hữu / trên tổng tài sản — số càng cao, công ty
  dùng vốn càng hiệu quả.
        """
    )

st.caption(
    "Nguồn dữ liệu: VCI (qua thư viện vnstock). Ứng dụng chỉ mang tính học tập, "
    "không phải khuyến nghị mua/bán."
)
