# kronos_predict_optimized.py
import sys
sys.path.append(r"C:\Users\Haichuan Yu\Desktop\BinanceAPI\kronos_source")

import pandas as pd
import matplotlib.pyplot as plt
import torch
import ccxt
from datetime import datetime, timedelta

from model.kronos import Kronos, KronosTokenizer, KronosPredictor

# 设置中文字体（解决方块问题）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 用 SimHei
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# ===================== 配置 =====================
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'          # 1小时线
LOOKBACK_DAYS = 7         # 只用最近1周历史数据（168根）
PREDICT_DAYS = 3          # 预测未来3天（72根）

# 计算时间范围
now = datetime.utcnow()
start_date = now - timedelta(days=LOOKBACK_DAYS)

# 1. 自动下载最近1周的1h数据
print("正在从 Binance 下载最近1周的1h数据...")
exchange = ccxt.binance()
ohlcv = exchange.fetch_ohlcv(
    SYMBOL,
    TIMEFRAME,
    since=exchange.parse8601(start_date.isoformat()),
    limit=1000
)

df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamps'] = pd.to_datetime(df['timestamp'], unit='ms')
df.to_csv('btc_1h_recent.csv', index=False)
print(f"下载完成！共 {len(df)} 根K线，从 {df['timestamps'].iloc[0]} 到 {df['timestamps'].iloc[-1]}")

# 2. 准备输入
lookback = len(df)  # 全部历史（约168根）
pred_len = PREDICT_DAYS * 24  # 3天 * 24小时 = 72根

x_df = df[['open', 'high', 'low', 'close', 'volume']]
x_timestamp = df['timestamps']

# 生成未来时间戳
last_time = df['timestamps'].iloc[-1]
future_timestamps = [last_time + timedelta(hours=i) for i in range(1, pred_len + 1)]
y_timestamp = pd.Series(future_timestamps)

# 3. 加载模型（用 Kronos-base 提高精度）
print("加载 Kronos 模型（base版，精度更高）...")
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-base")  # 换成 base，102M 参数
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")
predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

# 4. 预测（增加 sample_count 平均多条路径）
print("开始预测未来3天...")
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=0.7,             # 降低温度，更确定性
    top_p=0.8,
    sample_count=3,    # 生成3条预测路径，取平均
    verbose=True
)

print("预测结果前几行：")
print(pred_df.head())

# 5. 可视化（中文字体已设置）
plt.figure(figsize=(14, 7))
plt.plot(df['timestamps'], df['close'], label='历史收盘价 (1h)', color='blue', linewidth=1)
plt.plot(y_timestamp, pred_df['close'], label='预测收盘价 (未来3天)', color='red', linestyle='--', linewidth=1.5)
plt.title(f'BTC/USDT 预测 - Kronos (最近1周历史 + 未来3天)')
plt.xlabel('时间')
plt.ylabel('价格 (USDT)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 保存预测
pred_df.to_csv('btc_1h_prediction_future_3days.csv')
print("预测结果已保存到 btc_1h_prediction_future_3days.csv")