"""Worker chạy sàng lọc cổ phiếu trong tiến trình riêng.

Vì sao cần tiến trình riêng: nguồn dữ liệu miễn phí giới hạn 20 lệnh gọi/phút
cho MỖI tiến trình (vnai đếm trong bộ nhớ). Dashboard đang chạy đã tiêu tốn
hạn mức cho bảng giá/chỉ số, nên nếu quét 30 mã trong cùng tiến trình sẽ chạm
trần ngay và bị thư viện tự ngủ chờ rất lâu, treo luôn giao diện. Tiến trình
worker có bộ đếm riêng, tự giãn nhịp ~3s/mã để không vượt 20 lệnh/phút.

Giao tiếp với tiến trình cha qua stdout, mỗi dòng một sự kiện:
    PROGRESS<TAB>ma<TAB>i<TAB>tong   : tiến độ
    RESULT<TAB><json records>        : kết quả cuối (DataFrame dạng records)
Các dòng khác (banner của vnstock in ra...) sẽ bị tiến trình cha bỏ qua.

Chạy tay để thử:  python screen_worker.py '{"watchlist": ["FPT", "VNM"]}'
"""
from __future__ import annotations

import json
import sys

import data_loader


def main() -> None:
    params = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}

    def progress(sym: str, i: int, total: int) -> None:
        print(f"PROGRESS\t{sym}\t{i}\t{total}", flush=True)

    df = data_loader.run_screen(
        params.get("watchlist", []),
        rsi_min=params.get("rsi_min", 50.0),
        rsi_max=params.get("rsi_max", 70.0),
        vol_breakout=params.get("vol_breakout", 1.5),
        max_dist_52w=params.get("max_dist_52w", 0.15),
        sleep_between_calls=params.get("sleep_between_calls", 3.0),
        progress_callback=progress,
    )
    print("RESULT\t" + df.to_json(orient="records"), flush=True)


if __name__ == "__main__":
    main()
