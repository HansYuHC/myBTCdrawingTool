# news_utils.py
import requests
from datetime import datetime, timedelta, timezone

API_KEY = "e72048f8f8f31eda27a1094887cf9961e24a5245"

def fetch_cryptopanic_news(hours_back=72):  # 宽松到72小时
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

    print(f"过滤时间: {cutoff_time}")

    for post in data.get("results", []):
        pub_time_str = post.get("published_at")
        if not pub_time_str:
            continue
        pub_time = datetime.fromisoformat(pub_time_str.replace("Z", "+00:00"))

        print(f"新闻时间: {pub_time} | 标题: {post.get('title')[:50]}...")  # 调试打印

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

# 测试运行
if __name__ == "__main__":
    print("测试新闻采集...")
    news = fetch_cryptopanic_news(hours_back=72)
    if news:
        print(f"成功采集 {len(news)} 条新闻")
        for item in news[:3]:
            print(f"- {item['title'][:50]}... ({item['source']}, {item['published_at']})")
    else:
        print("未采集到新闻")