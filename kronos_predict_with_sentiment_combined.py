# kronos_predict_optimized_interactive.py
import sys

sys.path.append(r"C:\Users\Haichuan Yu\Desktop\BinanceAPI\kronos_source")

import pandas as pd
import numpy as np
import urllib.parse
import torch
import ccxt
import requests
import json
from datetime import datetime, timedelta, timezone
from transformers import pipeline
from model.kronos import Kronos, KronosTokenizer, KronosPredictor
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ===================== 配置 =====================
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
LOOKBACK_DAYS = 7
PREDICT_DAYS = 3
API_KEY = "e72048f8f8f31eda27a1094887cf9961e24a5245"
SENTIMENT_SCALE = 0.015

SOURCE_WEIGHTS = {
    "CoinDesk": 0.30,
    "CoinTelegraph": 0.25,
    "CryptoPanic": 0.20,
    "Twitter": 0.15,
    "Reddit": 0.10
}

# ===================== 情绪模型 =====================
sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")


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
        print(f"API 失败: {response.status_code}")
        return []

    data = response.json()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)

    news = []
    for post in data.get("results", []):
        pub_time_str = post.get("published_at")
        if not pub_time_str:
            continue

        pub_time = datetime.fromisoformat(pub_time_str.replace("Z", "+00:00"))
        if pub_time < cutoff:
            continue

        # Extract actual source and original URL if available
        source_title = post.get("source", {}).get("title", "CryptoPanic")
        original_url = post.get("original_url", post.get("url", f"https://cryptopanic.com/search?q={urllib.parse.quote(post.get('title',''))}"))

        news.append({
            "title": post.get("title", ""),
            "url": original_url,
            "source": source_title,
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
    if not news:
        return 0.0, 0.0, pd.DataFrame()

    df = pd.DataFrame(news)
    df["sentiment"] = df["title"].apply(analyze_sentiment)
    now = datetime.now(timezone.utc)
    df["hours_ago"] = df["published_at"].apply(lambda t: (now - t).total_seconds() / 3600)
    df["source_weight"] = df["source"].map(SOURCE_WEIGHTS).fillna(0.1)

    def weighted_avg(sub):
        if len(sub) == 0:
            return 0.0
        w = sub["source_weight"]
        return (sub["sentiment"] * w).sum() / w.sum()

    short_df = df[df["hours_ago"] <= 24]
    mid_df = df[(df["hours_ago"] > 24) & (df["hours_ago"] <= 72)]
    short_sent = max(min(weighted_avg(short_df), 0.5), -0.5)
    mid_sent = max(min(weighted_avg(mid_df), 0.5), -0.5)

    return short_sent, mid_sent, df


# ===================== 生成交互HTML =====================
def generate_interactive_html(fig, df_news, output_path):
    """完全重写版本 - 确保JavaScript能够执行"""

    import webbrowser
    import os

    # 准备新闻数据
    news_data_by_sentiment = {}
    if not df_news.empty:
        for sentiment_score in df_news['sentiment'].unique():
            matching = df_news[df_news['sentiment'] == sentiment_score]
            news_list = []
            for _, row in matching.iterrows():
                news_list.append({
                    "title": row['title'],
                    "url": row['url'],
                    "source": row['source'],
                    "time": row['published_at'].strftime('%m-%d %H:%M')
                })
            news_data_by_sentiment[str(int(sentiment_score))] = news_list

    news_json = json.dumps(news_data_by_sentiment, ensure_ascii=False)

    # 获取图表的JSON数据
    fig_json = fig.to_json()

    # 手动构建完整HTML
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BTC 预测与新闻情绪分析</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
            min-height: 100vh;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        .header {{ text-align: center; padding: 20px; margin-bottom: 20px; }}
        .header h1 {{ color: #00d4ff; font-size: 28px; margin-bottom: 10px; }}
        .header p {{ color: #888; font-size: 14px; }}
        .chart-container {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }}
        .tip-box {{
            background: rgba(0, 212, 255, 0.1);
            border: 1px solid #00d4ff;
            border-radius: 8px;
            padding: 10px 15px;
            margin-bottom: 15px;
            font-size: 13px;
            color: #00d4ff;
        }}
        .debug-box {{
            background: rgba(255, 255, 0, 0.15);
            border: 2px solid yellow;
            border-radius: 8px;
            padding: 12px 15px;
            margin-bottom: 15px;
            font-size: 13px;
            color: yellow;
            font-family: 'Courier New', monospace;
            font-weight: bold;
        }}
        #plotlyChart {{ width: 100%; height: 750px; }}
        .news-popup {{
            display: none;
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: linear-gradient(135deg, #1e3a5f 0%, #0d1b2a 100%);
            border: 2px solid #00d4ff;
            border-radius: 15px;
            padding: 0;
            min-width: 500px;
            max-width: 700px;
            max-height: 80vh;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
            z-index: 10000;
            overflow: hidden;
            pointer-events: auto;
        }}
        .popup-header {{
            background: linear-gradient(90deg, #00d4ff 0%, #0099cc 100%);
            color: #0d1b2a;
            padding: 15px 20px;
            font-size: 16px;
            font-weight: bold;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .popup-close {{
            background: rgba(0,0,0,0.2);
            border: none;
            color: #0d1b2a;
            font-size: 24px;
            cursor: pointer;
            width: 35px;
            height: 35px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s;
        }}
        .popup-close:hover {{ background: rgba(0,0,0,0.4); transform: rotate(90deg); }}
        .popup-content {{ padding: 15px 20px; max-height: 60vh; overflow-y: auto; }}
        .news-item {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 12px;
            border-left: 4px solid #00d4ff;
            transition: all 0.3s;
        }}
        .news-item:hover {{
            background: rgba(0, 212, 255, 0.15);
            transform: translateX(5px);
            box-shadow: 0 5px 20px rgba(0, 212, 255, 0.2);
        }}
        .news-meta {{ display: flex; gap: 15px; margin-bottom: 8px; font-size: 12px; }}
        .news-source {{ color: #ffd700; font-weight: bold; }}
        .news-time {{ color: #888; }}
        .news-link {{ 
            color: #e0e0e0; 
            text-decoration: none; 
            font-size: 14px; 
            line-height: 1.5; 
            display: block;
            cursor: pointer;
        }}
        .news-link:hover {{ color: #00d4ff; text-decoration: underline; }}
        .news-arrow {{ color: #00d4ff; margin-left: 8px; }}
        .overlay {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            z-index: 9999;
            backdrop-filter: blur(5px);
            pointer-events: auto;
        }}
        .sentiment-badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 12px;
            margin-left: 10px;
        }}
        .sentiment-positive {{ background: #27ae60; color: white; }}
        .sentiment-negative {{ background: #e74c3c; color: white; }}
        .sentiment-neutral {{ background: #95a5a6; color: white; }}
        .no-news {{ text-align: center; padding: 30px; color: #888; }}
        .popup-content::-webkit-scrollbar {{ width: 8px; }}
        .popup-content::-webkit-scrollbar-track {{ background: rgba(255, 255, 255, 0.1); border-radius: 4px; }}
        .popup-content::-webkit-scrollbar-thumb {{ background: #00d4ff; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📈 BTC/USDT 预测与新闻情绪分析</h1>
            <p>基于 Kronos 模型 + 新闻情绪修正 | 数据更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        <div class="chart-container">
            <div class="tip-box">
                💡 <strong>提示：</strong>点击下方情绪柱状图的柱子，可弹出该情绪评分对应的所有新闻链接
            </div>
            <div class="debug-box" id="debugBox">🔍 正在加载...</div>
            <div id="plotlyChart"></div>
        </div>
    </div>

    <div class="overlay" id="overlay"></div>
    <div class="news-popup" id="newsPopup">
        <div class="popup-header">
            <span id="popupTitle">新闻列表</span>
            <button class="popup-close" onclick="window.closePopup()">×</button>
        </div>
        <div class="popup-content" id="popupContent"></div>
    </div>

    <script type="text/javascript">
        // 全局变量
        window.newsData = {news_json};
        window.plotlyChart = null;

        const sentimentLabels = {{
            '-3': '🔴 极度负面', '-2': '🟠 负面', '-1': '🟡 轻微负面',
            '0': '⚪ 中性', '1': '🟢 轻微正面', '2': '🟢 正面', '3': '🟢 极度正面'
        }};

        // 调试：打印新闻数据
        console.log('加载的新闻数据:', window.newsData);

        function log(msg) {{
            console.log('[BTC-APP]', msg);
            const box = document.getElementById('debugBox');
            if (box) {{
                box.innerHTML = '🔍 ' + msg;
            }}
        }}

        function getSentimentClass(score) {{
            score = parseInt(score);
            return score > 0 ? 'sentiment-positive' : score < 0 ? 'sentiment-negative' : 'sentiment-neutral';
        }}

        // 修改点击行为：复制标题并使用 Google "I'm Feeling Lucky" 作为 fallback，如果需要
        function handleNewsClick(title, url) {{
            // 首先尝试复制标题到剪贴板
            navigator.clipboard.writeText(title).then(() => {{
                console.log('标题已复制到剪贴板:', title);
            }}).catch(err => {{
                console.error('复制失败:', err);
            }});

            // 如果 URL 是 CryptoPanic 的，改用 Google Lucky 搜索标题以获取全文
            if (url.includes('cryptopanic.com')) {{
                const searchQuery = encodeURIComponent(title);
                const googleLuckyUrl = `https://www.google.com/search?q=${{searchQuery}}&btnI`;
                window.open(googleLuckyUrl, '_blank');
            }} else {{
                // 否则直接打开原 URL
                window.open(url, '_blank');
            }}
        }}

        window.showPopup = function(sentiment) {{
            log('显示新闻弹窗 - 情绪分数: ' + sentiment);

            const sentimentStr = String(sentiment);
            const label = sentimentLabels[sentimentStr] || ('情绪 ' + sentiment);
            const badgeClass = getSentimentClass(sentiment);

            document.getElementById('popupTitle').innerHTML = 
                label + ' <span class="sentiment-badge ' + badgeClass + '">评分: ' + sentiment + '</span>';

            const newsList = window.newsData[sentimentStr] || [];
            const content = document.getElementById('popupContent');

            if (newsList.length === 0) {{
                content.innerHTML = '<div class="no-news">该情绪评分暂无新闻</div>';
            }} else {{
                let html = '';
                newsList.forEach(function(news) {{
                    // 验证URL
                    const newsUrl = news.url || '#';
                    console.log('新闻URL:', newsUrl, '来源:', news.source);

                    html += '<div class="news-item">';
                    html += '<div class="news-meta">';
                    html += '<span class="news-source">📰 ' + (news.source || 'Unknown') + '</span>';
                    html += '<span class="news-time">🕐 ' + news.time + '</span>';
                    html += '</div>';
                    html += '<a onclick="handleNewsClick(\\'' + news.title.replace(/'/g, "\\\\'") + '\\', \\'' + newsUrl + '\\'); return false;" class="news-link">';
                    html += news.title + '<span class="news-arrow">→</span>';
                    html += '</a></div>';
                }});
                content.innerHTML = html;
            }}

            document.getElementById('overlay').style.display = 'block';
            const popup = document.getElementById('newsPopup');
            popup.style.display = 'block';
            popup.style.opacity = '0';
            popup.style.transform = 'translate(-50%, -50%) scale(0.8)';

            setTimeout(function() {{
                popup.style.transition = 'all 0.3s ease';
                popup.style.opacity = '1';
                popup.style.transform = 'translate(-50%, -50%) scale(1)';
            }}, 10);
        }};

        window.closePopup = function() {{
            const popup = document.getElementById('newsPopup');
            popup.style.opacity = '0';
            popup.style.transform = 'translate(-50%, -50%) scale(0.8)';
            setTimeout(function() {{
                popup.style.display = 'none';
                document.getElementById('overlay').style.display = 'none';
            }}, 300);
        }};

        // 点击遮罩层关闭弹窗
        document.getElementById('overlay').addEventListener('click', function(e) {{
            if (e.target.id === 'overlay') {{
                window.closePopup();
            }}
        }});

        document.addEventListener('keydown', function(e) {{
            if (e.key === 'Escape') window.closePopup();
        }});

        // 初始化Plotly图表
        function initChart() {{
            log('开始初始化图表...');

            try {{
                const figData = {fig_json};
                log('图表数据已加载，trace数量: ' + figData.data.length);

                Plotly.newPlot('plotlyChart', figData.data, figData.layout, {{responsive: true}})
                    .then(function(gd) {{
                        log('图表渲染完成！');
                        window.plotlyChart = gd;

                        // 绑定点击事件
                        gd.on('plotly_click', function(data) {{
                            log('🎯 检测到点击！');
                            console.log('点击数据:', data);

                            if (data && data.points && data.points.length > 0) {{
                                const point = data.points[0];
                                log('点击位置: curveNumber=' + point.curveNumber + ', x=' + point.x + ', type=' + point.data.type);

                                if (point.data.type === 'bar') {{
                                    log('✅ 这是柱状图！打开新闻弹窗');
                                    window.showPopup(point.x);
                                }} else {{
                                    log('这是' + point.data.type + '，不是柱状图');
                                }}
                            }}
                        }});

                        log('✅ 事件绑定成功！请点击下方柱状图');
                    }})
                    .catch(function(err) {{
                        log('❌ 图表渲染失败: ' + err.message);
                        console.error(err);
                    }});

            }} catch (err) {{
                log('❌ 初始化失败: ' + err.message);
                console.error(err);
            }}
        }}

        // 页面加载完成后初始化
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', initChart);
        }} else {{
            initChart();
        }}

        log('脚本已加载，等待初始化...');
    </script>
</body>
</html>'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\n✅ 已保存交互图表: {output_path}")
    print("📌 打开HTML文件后，页面顶部会显示黄色调试框")
    print("📌 按F12打开浏览器控制台可查看详细日志")

    # 自动在浏览器中打开
    try:
        abs_path = os.path.abspath(output_path)
        webbrowser.open('file://' + abs_path)
        print(f"🌐 已在浏览器中打开: {output_path}")
    except Exception as e:
        print(f"⚠️ 无法自动打开浏览器: {e}")
        print(f"   请手动打开: {os.path.abspath(output_path)}")


# ===================== 主程序 =====================
if __name__ == "__main__":
    print("=" * 60)
    print("BTC 价格预测 + 新闻情绪分析")
    print("=" * 60)

    # 1. 历史数据
    print("\n[1/6] 获取历史数据...")
    exchange = ccxt.binance()
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=LOOKBACK_DAYS)

    ohlcv = exchange.fetch_ohlcv(
        SYMBOL, TIMEFRAME,
        since=exchange.parse8601(start.isoformat()),
        limit=1000
    )

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamps"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    print(f"   历史数据: {df['timestamps'].iloc[0]} 到 {df['timestamps'].iloc[-1]}")
    print(f"   数据行数: {len(df)}")

    x_df = df[["open", "high", "low", "close", "volume"]]
    x_timestamp = df["timestamps"]

    pred_len = PREDICT_DAYS * 24
    last_time = df["timestamps"].iloc[-1]
    y_timestamp = pd.Series([last_time + timedelta(hours=i) for i in range(1, pred_len + 1)])

    print(f"   预测范围: {y_timestamp.iloc[0]} 到 {y_timestamp.iloc[-1]}")

    # 2. 新闻情绪
    print("\n[2/6] 采集新闻并分析情绪...")
    news = fetch_cryptopanic_news(72)
    short_sent, mid_sent, df_news = split_sentiment(news)
    print(f"   短期情绪 (0-24h): {short_sent:.2f}")
    print(f"   中期情绪 (24-72h): {mid_sent:.2f}")

    # 3. Kronos预测
    print("\n[3/6] 加载 Kronos 模型...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

    print("\n[4/6] 执行预测...")
    pred_df_pure = predictor.predict(
        df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
        pred_len=pred_len, T=0.7, top_p=0.8, sample_count=3, verbose=True
    )

    # 4. 情绪修正
    print("\n[5/6] 应用情绪修正...")
    pred_df = pred_df_pure.copy()
    pred_df["close"] = pred_df["close"].astype(np.float64)

    for i in range(pred_len):
        hour = i + 1
        decay = max(0, 1 - hour / 72)
        sentiment = short_sent if hour <= 24 else mid_sent
        factor = 1 + sentiment * decay * SENTIMENT_SCALE
        pred_df.iloc[i, pred_df.columns.get_loc("close")] *= factor

    # 5. 创建图表
    print("\n[6/6] 生成可视化图表...")
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.12,
        subplot_titles=("BTC/USDT 价格预测（过去7天 + 未来3天）", "📊 点击柱子查看新闻详情")
    )

    hist_timestamps = df["timestamps"].tolist()
    pred_timestamps = y_timestamp.tolist()

    # 价格曲线
    fig.add_trace(go.Scatter(
        x=hist_timestamps, y=df["close"].tolist(),
        mode='lines', name='历史收盘价',
        line=dict(color='#3498db', width=2)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=pred_timestamps, y=pred_df_pure["close"].tolist(),
        mode='lines', name='纯 Kronos 预测',
        line=dict(color='#95a5a6', width=2, dash='dash')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=pred_timestamps, y=pred_df["close"].tolist(),
        mode='lines', name='Kronos + 新闻情绪',
        line=dict(color='#e74c3c', width=2)
    ), row=1, col=1)

    # 分界线
    boundary_time = hist_timestamps[-1]
    y_min = min(df["close"].min(), pred_df["close"].min()) * 0.995
    y_max = max(df["close"].max(), pred_df["close"].max()) * 1.005

    fig.add_shape(
        type="line", x0=boundary_time, x1=boundary_time,
        y0=y_min, y1=y_max,
        line=dict(color="yellow", width=2, dash="dot"),
        row=1, col=1
    )

    fig.add_annotation(
        x=boundary_time, y=y_max,
        text="← 历史 | 预测 →",
        showarrow=False,
        font=dict(size=10, color="yellow"),
        bgcolor="rgba(0,0,0,0.5)",
        row=1, col=1
    )

    all_timestamps = hist_timestamps + pred_timestamps
    fig.update_xaxes(range=[all_timestamps[0], all_timestamps[-1]], tickformat='%m-%d %H:%M', tickangle=45, row=1,
                     col=1)
    fig.update_yaxes(range=[y_min, y_max], row=1, col=1)

    # 情绪注解
    fig.add_annotation(
        text=f"<b>短期情绪 (0-24h):</b> {short_sent:.2f}<br><b>中期情绪 (24-72h):</b> {mid_sent:.2f}",
        xref="paper", yref="paper", x=0.02, y=0.95,
        showarrow=False, font=dict(size=12, color='white'),
        bgcolor="rgba(0,0,0,0.7)", bordercolor="#00d4ff",
        borderwidth=2, borderpad=8
    )

    # 情绪柱状图
    if not df_news.empty:
        sentiment_counts = df_news['sentiment'].value_counts().sort_index()
        colors = []
        for score in sentiment_counts.index:
            if score <= -2:
                colors.append('#e74c3c')
            elif score == -1:
                colors.append('#f39c12')
            elif score == 0:
                colors.append('#95a5a6')
            elif score == 1:
                colors.append('#27ae60')
            else:
                colors.append('#2ecc71')

        fig.add_trace(go.Bar(
            x=sentiment_counts.index.tolist(),
            y=sentiment_counts.values.tolist(),
            marker_color=colors,
            name='新闻数量',
            text=sentiment_counts.values.tolist(),
            textposition='outside',
            hovertemplate="<b>情绪分数: %{x}</b><br>新闻数量: %{y}<br><i>👆 点击查看新闻详情</i><extra></extra>"
        ), row=2, col=1)

        fig.update_xaxes(title_text="情绪分数 (-3极负 ~ +3极正)", tickmode='linear', tick0=-3, dtick=1, row=2, col=1)
        fig.update_yaxes(title_text="新闻数量", row=2, col=1)

    # 整体布局
    fig.update_layout(
        height=750,
        template='plotly_dark',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(0,0,0,0.5)'),
        hovermode="closest",
        paper_bgcolor='rgba(26,26,46,1)',
        plot_bgcolor='rgba(22,33,62,1)'
    )

    # 保存文件
    generate_interactive_html(fig, df_news, "btc_prediction_with_sentiment.html")
    pred_df_pure.to_csv("btc_1h_prediction_pure.csv", index=False)
    pred_df.to_csv("btc_1h_prediction_with_news.csv", index=False)

    print("\n" + "=" * 60)
    print("✅ 全部完成！")
    print("=" * 60)
    print(f"📊 交互图表: btc_prediction_with_sentiment.html")
    print(f"📁 纯预测: btc_1h_prediction_pure.csv")
    print(f"📁 情绪修正预测: btc_1h_prediction_with_news.csv")
    print("=" * 60)