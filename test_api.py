import requests

url = "http://localhost:8080/tradingview"

# نمونه متن استراتژی برای تست
strategy_text = """
This is a Smart Money Concept strategy. Entry after BOS and FVG confirmation. 
SL below last swing low. TP at next liquidity zone.
"""

# داده تستی (OHLCV) برای بک‌تست
ohlc_data = [
    {"timestamp": "2025-06-07T00:00:00Z", "open": 100, "high": 105, "low": 99, "close": 104, "volume": 1200},
    {"timestamp": "2025-06-07T01:00:00Z", "open": 104, "high": 107, "low": 102, "close": 106, "volume": 1100},
    {"timestamp": "2025-06-07T02:00:00Z", "open": 106, "high": 110, "low": 105, "close": 109, "volume": 1500},
    {"timestamp": "2025-06-07T03:00:00Z", "open": 109, "high": 112, "low": 107, "close": 108, "volume": 1300},
    {"timestamp": "2025-06-07T04:00:00Z", "open": 108, "high": 111, "low": 106, "close": 110, "volume": 1700}
]

# بسته‌بندی دیتا
payload = {
    "strategy_text": strategy_text,
    "ohlc_data": ohlc_data
}

# ارسال POST به سرور محلی
response = requests.post(url, json=payload)

# نمایش پاسخ
print("Server response:")
print(response.json())