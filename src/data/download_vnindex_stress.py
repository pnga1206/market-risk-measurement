import os
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from dnse import DNSEClient


START_DATE = "2007-01-01"
END_DATE = "2009-12-31"

SYMBOL = "VNINDEX"
RESOLUTION = "1D"

OUTPUT_FILE = Path("data/stress_2008/vnindex_raw.csv")


def date_to_timestamp(date_string):
    dt = datetime.strptime(date_string, "%Y-%m-%d")
    dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def timestamp_to_date(timestamp):
    return datetime.fromtimestamp(
        timestamp,
        tz=timezone.utc
    ).strftime("%Y-%m-%d")


def download_vnindex():

    load_dotenv()

    api_key = os.getenv("DNSE_API_KEY")
    api_secret = os.getenv("DNSE_API_SECRET")

    if not api_key or not api_secret:
        raise ValueError(
            "Không tìm thấy DNSE_API_KEY hoặc DNSE_API_SECRET trong .env"
        )

    client = DNSEClient(
        api_key=api_key,
        api_secret=api_secret
    )

    start_timestamp = date_to_timestamp(START_DATE)
    end_timestamp = date_to_timestamp(END_DATE)

    print("Downloading VNINDEX...")
    print(f"From: {START_DATE}")
    print(f"To:   {END_DATE}")

    status, body = client.get_ohlc(
        "INDEX",
        {
            "symbol": SYMBOL,
            "resolution": RESOLUTION,
            "from": start_timestamp,
            "to": end_timestamp,
        }
    )

    print("STATUS:", status)

    if status != 200:
        raise RuntimeError(f"DNSE API error: {body}")

    data = json.loads(body)

    timestamps = data["t"]
    opens = data["o"]
    highs = data["h"]
    lows = data["l"]
    closes = data["c"]
    volumes = data["v"]

    lengths = {
        len(timestamps),
        len(opens),
        len(highs),
        len(lows),
        len(closes),
        len(volumes),
    }

    if len(lengths) != 1:
        raise ValueError(
            "Các mảng dữ liệu OHLC không cùng độ dài."
        )

    rows = []

    for i in range(len(timestamps)):
        rows.append({
            "date": timestamp_to_date(timestamps[i]),
            "open": opens[i],
            "high": highs[i],
            "low": lows[i],
            "close": closes[i],
            "volume": volumes[i],
        })

    rows.sort(key=lambda x: x["date"])

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", OUTPUT_FILE)
    print("Rows:", len(rows))
    print("First date:", rows[0]["date"])
    print("Last date:", rows[-1]["date"])


if __name__ == "__main__":
    download_vnindex()