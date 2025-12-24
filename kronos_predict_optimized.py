# kronos_predict_optimized.py
import sys
sys.path.append(r"C:\Users\Haichuan Yu\Desktop\BinanceAPI\kronos_source")

import pandas as pd
import matplotlib.pyplot as plt
import torch
import ccxt
import requests
from datetime import datetime, timedelta, timezone
from transformers import pipeline

from model.kronos import Kronos, KronosTokenizer, KronosPredictor

# ===================== 中文字体 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 配置 =====================
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
LOOKBACK_DAYS = 7
PREDICT_DAYS = 3
API_KEY = "e72048f8f8f31eda27a1094887cf9961e24a5245"

# 情绪缩放系数（⭐核心调参位）
SENTIMENT_SCALE = 0.015   # 建议 0.01 ~ 0.03

SOURCE_WEIGHTS = {
    "CoinDesk": 0.30,
    "CoinTelegraph": 0.25,
    "CryptoPanic": 0.20,
    "Twitter": 0.15,
    "Reddit": 0.10
}

# ===================== 情绪模型 =====================
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)

# ===================== 新闻采集 =====================
def fetch_cryptopanic_news(hours_back=72):
    url = "https://cryptopanic.com/api/developer/v2/posts/"
    params = {
        "auth_token": API_KEY,
        "currencies": "BTC",
        "public": "true",
        "limit": 50,
        "kind": "news"
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        return []

    data = response.json()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)
    news = []

    for post in data.get("results", []):
        pub_time = datetime.fromisoformat(
            post["published_at"].replace("Z", "+00:00")
        )
        if pub_time < cutoff:
            continue

        news.append({
            "title": post.get("title", ""),
            "source": post.get("source", {}).get("title", "Unknown"),
            "published_at": pub_time
        })

    print(f"采集到 {len(news)} 条新闻")
    return news

# ===================== 情绪分析 =====================
def analyze_sentiment(title):
    r = sentiment_pipeline(title)[0]
    label, score = r["label"], r["score"]

    if label == "positive":
        return 3 if score >= 0.95 else 2 if score >= 0.8 else 1
    if label == "negative":
        return -3 if score >= 0.95 else -2 if score >= 0.8 else -1
    return 0

def split_sentiment(news):
    """分 short / mid 两段情绪"""
    if not news:
        return 0.0, 0.0

    df = pd.DataFrame(news)
    df["sentiment"] = df["title"].apply(analyze_sentiment)

    now = datetime.now(timezone.utc)
    df["hours_ago"] = df["published_at"].apply(
        lambda t: (now - t).total_seconds() / 3600
    )

    df["source_weight"] = df["source"].map(SOURCE_WEIGHTS).fillna(0.1)

    def weighted_avg(sub):
        if len(sub) == 0:
            return 0.0
        w = sub["source_weight"]
        return (sub["sentiment"] * w).sum() / w.sum()

    short_df = df[df["hours_ago"] <= 24]
    mid_df = df[(df["hours_ago"] > 24) & (df["hours_ago"] <= 72)]

    short_sent = weighted_avg(short_df)
    mid_sent = weighted_avg(mid_df)

    # 限幅
    short_sent = max(min(short_sent, 0.5), -0.5)
    mid_sent = max(min(mid_sent, 0.5), -0.5)

    return short_sent, mid_sent

# ===================== 主程序 =====================
if __name__ == "__main__":

    # ===== 1️⃣ 历史数据 =====
    exchange = ccxt.binance()
    now = datetime.utcnow()
    start = now - timedelta(days=LOOKBACK_DAYS)

    ohlcv = exchange.fetch_ohlcv(
        SYMBOL,
        TIMEFRAME,
        since=exchange.parse8601(start.isoformat()),
        limit=1000
    )

    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamps"] = pd.to_datetime(df["timestamp"], unit="ms")

    x_df = df[["open", "high", "low", "close", "volume"]]
    x_timestamp = df["timestamps"]

    pred_len = PREDICT_DAYS * 24
    last_time = df["timestamps"].iloc[-1]
    y_timestamp = pd.Series(
        [last_time + timedelta(hours=i) for i in range(1, pred_len + 1)]
    )

    # ===== 2️⃣ 新闻情绪 =====
    news = fetch_cryptopanic_news(72)
    short_sent, mid_sent = split_sentiment(news)

    print(f"短期情绪 (0-24h): {short_sent:.2f}")
    print(f"中期情绪 (24-72h): {mid_sent:.2f}")

    # ===== 3️⃣ Kronos 预测 =====
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

    pred_df_pure = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=0.7,
        top_p=0.8,
        sample_count=3,
        verbose=True
    )

    # ===== 4️⃣ 情绪 × 衰减 × 乘数 =====
    pred_df = pred_df_pure.copy()

    for i in range(pred_len):
        hour = i + 1
        decay = max(0, 1 - hour / 72)

        sentiment = short_sent if hour <= 24 else mid_sent
        factor = 1 + sentiment * decay * SENTIMENT_SCALE

        pred_df.iloc[i, pred_df.columns.get_loc("close")] *= factor

    # ===== 5️⃣ 可视化 =====
    plt.figure(figsize=(14, 7))
    plt.plot(df["timestamps"], df["close"], label="历史收盘价", color="blue")
    plt.plot(y_timestamp, pred_df_pure["close"],
             "--", color="gray", label="纯 Kronos")
    plt.plot(y_timestamp, pred_df["close"],
             color="red", label="Kronos + 新闻情绪")

    plt.title("BTC/USDT 预测：时间衰减 × 分段新闻情绪")
    plt.xlabel("时间")
    plt.ylabel("价格 (USDT)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    pred_df_pure.to_csv("btc_1h_prediction_pure.csv", index=False)
    pred_df.to_csv("btc_1h_prediction_with_news.csv", index=False)

    print("✅ 预测完成并保存")
