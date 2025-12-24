# news_sentiment.py
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from transformers import pipeline
import matplotlib.pyplot as plt

# 设置中文字体（Windows常用字体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

API_KEY = "e72048f8f8f31eda27a1094887cf9961e24a5245"  # 你的 Key

sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

def fetch_cryptopanic_news(hours_back=24):
    url = "https://cryptopanic.com/api/developer/v2/posts/"
    params = {
        "auth_token": API_KEY,
        "currencies": "BTC",
        "public": "true",
        "limit": 50,
        "kind": "news"
    }

    print("请求 URL:", requests.Request('GET', url, params=params).prepare().url)
    response = requests.get(url, params=params)

    print(f"状态码: {response.status_code}")
    print(f"响应预览: {response.text[:500]}...")

    if response.status_code != 200:
        print(f"API 失败: {response.status_code} - {response.text}")
        return []

    data = response.json()
    news_list = []
    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)

    for post in data.get("results", []):
        pub_time_str = post.get("published_at")
        if not pub_time_str:
            continue
        pub_time = datetime.fromisoformat(pub_time_str.replace("Z", "+00:00"))

        if pub_time < cutoff_time:
            continue

        title = post.get("title", "")
        url = post.get("url", "")
        source = post.get("source", {}).get("title", "Unknown")

        news_list.append({
            "title": title,
            "url": url,
            "source": source,
            "published_at": pub_time
        })

    print(f"采集到 {len(news_list)} 条新闻")
    return news_list

def analyze_sentiment(title):
    result = sentiment_pipeline(title)[0]
    label = result['label']
    score = result['score']

    if label == "positive":
        if score >= 0.95: return 3
        elif score >= 0.80: return 2
        else: return 1
    elif label == "negative":
        if score >= 0.95: return -3
        elif score >= 0.80: return -2
        else: return -1
    else:
        return 0

if __name__ == "__main__":
    print("开始采集 CryptoPanic 的 BTC 新闻...")
    news = fetch_cryptopanic_news(hours_back=72)  # 改成72小时，采集更多

    if news:
        df_news = pd.DataFrame(news)
        df_news['sentiment'] = df_news['title'].apply(analyze_sentiment)

        print("\n最近新闻情绪分析：")
        print(df_news[['title', 'source', 'published_at', 'sentiment']])

        avg_sentiment = df_news['sentiment'].mean()
        print(f"\n平均情绪分数: {avg_sentiment:.2f} (范围 -3 ~ +3)")

        df_news['sentiment'].value_counts().sort_index().plot(kind='bar', title='BTC 新闻情绪分布')
        plt.xlabel('情绪分数 (-3 ~ +3)')
        plt.ylabel('新闻数量')
        plt.show()
    else:
        print("未采集到新闻")