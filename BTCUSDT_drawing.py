# btc_true_full_history_web.py
# 终极完整版：所有周期拉满2018年至今 + 多周期按钮 + 手机电脑都能看

from binance.client import Client
from flask import Flask, render_template_string, redirect, url_for
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta
import threading
import time
from datetime import datetime, timedelta
import json
import os

# ================== 永久保存你的画图 ==================
SAVE_FILE = "btc_drawings.json"


def save_drawings(shapes):
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump({"shapes": [dict(s) for s in shapes]}, f)


def load_drawings():
    if os.path.exists(SAVE_FILE):
        try:
            with open(SAVE_FILE, "r", encoding="utf-8") as f:
                return json.load(f).get("shapes", [])
        except:
            return []
    return []


# ================== 配置 ==================
SYMBOL = "BTCUSDT"
client = Client()
app = Flask(__name__)
data_cache = {}
current_interval = "4h"
INTERVALS = {"1m": "1m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w", "1M": "1M"}

fig = None


# 创建图表（只保留K线和MA）
def create_figure():
    global fig
    fig = make_subplots(
        rows=1, cols=1,  # 只用1行1列
        subplot_titles=("BTC/USDT ─ 真·全历史多周期神器（2018年至今）",),
    )

    fig.update_layout(
        template="plotly_dark",
        height=700,  # 高度调低一点，更简洁
        title={"text": "<b>BTC/USDT 全历史神器</b><br>点击下方按钮切换周期", "x": 0.5},
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        dragmode="zoom",
        hovermode="x unified",
        xaxis=dict(showgrid=True, gridcolor="#333", rangeslider_visible=False),
        modebar_add=['drawline', 'drawopenpath', 'drawcircle', 'drawrect', 'eraseshape'],
        updatemenus=[
            dict(
                buttons=[
                    dict(label="框选放大", method="relayout", args=[{"dragmode": "zoom"}]),
                    dict(label="画线模式", method="relayout", args=[{"dragmode": "drawline"}]),
                    dict(label="清除所有", method="relayout", args=[{"shapes": []}]),
                ],
                direction="right", showactive=True, x=0, y=1.05,
                bgcolor="#ff4444", font=dict(color="white", size=12)
            )
        ],
        shapes=load_drawings()
    )

    # 只添加K线和MA轨迹
    fig.add_trace(go.Candlestick(name="BTCUSDT",
                                 increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350"), row=1, col=1)
    fig.add_trace(go.Scatter(mode="lines", name="MA20", line=dict(color="#00ff88", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(mode="lines", name="MA50", line=dict(color="#ff6b6b", width=2)), row=1, col=1)


# 拉取完整历史数据（所有周期都支持）
def load_full_history(interval):
    cache_key = f"{interval}_2018_full"
    if cache_key in data_cache:
        return data_cache[cache_key]

    print(f"正在拉取 {interval} 从2018年至今完整数据...")
    all_klines = []
    start_str = "1 Jan, 2018"

    while True:
        try:
            klines = client.get_historical_klines(
                SYMBOL, interval, start_str=start_str, limit=1000
            )
            if not klines or len(klines) == 0:
                break
            all_klines.extend(klines)
            start_str = str(klines[-1][0] + 1)
            if len(klines) < 1000:
                break
            time.sleep(0.15)
        except:
            break

    if len(all_klines) == 0:
        print("拉取失败，降级使用最近数据")
        klines = client.get_klines(symbol=SYMBOL, interval=interval, limit=1000)
        all_klines = klines

    df = pd.DataFrame(all_klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'ct', 'qav', 'trades', 'tbb', 'tbq', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

    df['ma20'] = df['close'].rolling(20, min_periods=1).mean()
    df['ma50'] = df['close'].rolling(50, min_periods=1).mean()
    df['rsi'] = ta.rsi(df['close'], length=14)
    macd = ta.macd(df['close'])
    df = pd.concat([df, macd.add_prefix('macd_')], axis=1)

    data_cache[cache_key] = df
    print(f"{interval} 完整历史加载成功！共 {len(df)} 根K线")
    return df


def update_chart():
    global fig
    if fig is None: return

    df = load_full_history(current_interval)
    d = df.copy().ffill().fillna(0)

    last_time = d['timestamp'].iloc[-1]
    future_days = {"1m": 30, "15m": 90, "1h": 180, "4h": 365, "1d": 730, "1w": 1825, "1M": 3650}
    future_time = last_time + timedelta(days=future_days.get(current_interval, 730))
    fig.update_xaxes(range=[d['timestamp'].iloc[0], future_time])

    # 只更新K线和MA
    fig.data[0].x = d['timestamp']
    fig.data[0].open = d['open']
    fig.data[0].high = d['high']
    fig.data[0].low = d['low']
    fig.data[0].close = d['close']
    fig.data[1].x = d['timestamp']
    fig.data[1].y = d['ma20']
    fig.data[2].x = d['timestamp']
    fig.data[2].y = d['ma50']


# ================== HTML模板 + 周期按钮 ==================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>BTC/USDT 全历史神器</title>
<style>
    body{margin:0;background:#000;font-family:Arial;}
    .header{padding:15px;text-align:center;background:#111;color:#0f0;}
    .buttons{text-align:center;padding:10px;background:#222;}
    button{margin:5px;padding:10px 20px;font-size:16px;background:#333;color:#0f0;border:1px solid #0f0;border-radius:8px;cursor:pointer;}
    button:hover{background:#0f0;color:#000;}
    button.active{background:#0f0;color:#000;font-weight:bold;}
</style></head>
<body>
<div class="header"><h1>BTC/USDT 全历史多周期神器</h1>
<p>当前周期: <b>{{ interval }}</b> | 更新: <span id="time">{{ now }}</span></p></div>
<div class="buttons">
{% for key in intervals %}
    <button onclick="location.href='/switch/{{ key }}'" {% if key == interval %}class="active"{% endif %}>{{ key }}</button>
{% endfor %}
</div>
{{ plot_div|safe }}
<script>
    setInterval(() => location.reload(), 120000);
    setInterval(() => document.getElementById('time').innerText = new Date().toLocaleString(), 1000);
</script>
</body></html>
"""


@app.route('/')
def index():
    global fig
    if fig is None:
        create_figure()
        update_chart()
    plot_div = fig.to_html(include_plotlyjs="cdn", div_id="chart")
    return render_template_string(HTML_TEMPLATE, plot_div=plot_div, interval=current_interval,
                                  now=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), intervals=INTERVALS.keys())


@app.route('/switch/<interval>')
def switch(interval):
    global current_interval
    if interval in INTERVALS:
        current_interval = interval
        update_chart()
    return redirect(url_for('index'))


def auto_task():
    while True:
        time.sleep(120)
        try:
            if fig and fig.layout.shapes:
                save_drawings(fig.layout.shapes)
        except:
            pass


# ================== 主程序 ==================
if __name__ == '__main__':
    create_figure()
    update_chart()
    threading.Thread(target=auto_task, daemon=True).start()

    print("\n" + "=" * 100)
    print("BTC/USDT 真·全历史多周期神器启动成功！")
    print("所有周期都拉满2018年至今（1分钟线也能看到2018年大底！）")
    print("地址：http://127.0.0.1:8888")
    print("手机访问：http://你的IP:8888")
    print("=" * 100)

    app.run(host='0.0.0.0', port=8888, debug=False, threaded=True)