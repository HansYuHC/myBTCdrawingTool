# BinanceAPI - BTC 价格预测项目集合

这个文件夹目前主要存放几套与 **BTC/USDT** 价格预测相关的实验性脚本，主要基于 **Kronos** 时序模型，并结合不同程度的新闻情绪分析。

目前项目还处于开发/实验阶段，结构比较混乱，未来计划整理成更清晰的模块化结构。

## 当前主要脚本说明

| 文件名                                   | 主要功能描述                                                                 | 数据来源              | 输出形式                          | 新闻情绪 | 交互界面 | 推荐优先级 |
|------------------------------------------|-----------------------------------------------------------------------------|-----------------------|-----------------------------------|----------|----------|------------|
| `kronos_predict.py`                      | 最基础的 Kronos 模型纯价格预测                                              | 仅 Binance K 线       | 控制台 + csv                      | ✗        | ✗        | ★☆☆☆☆      |
| `kronos_predict_optimized.py`            | Kronos 预测 + CryptoPanic 新闻情绪修正（较早版本）                         | Binance + CryptoPanic | 控制台 + csv + 简单 html 图表     | ✓        | 基本     | ★★☆☆☆      |
| `kronos_predict_optimized_interactive.py`| Kronos + 新闻情绪修正 + **交互式 HTML 报告**（目前推荐版本）                | Binance + CryptoPanic | 交互式 HTML（Plotly）+ csv        | ✓        | ★★★★     | ★★★★★      |
| `kronos_predict_with_sentiment_combined.py` | Kronos 预测 + 情绪分析 + 新闻列表整合到一个 HTML 页面，可点击查看原文     | Binance + CryptoPanic | 更完整的交互 HTML                 | ✓        | ★★★★     | ★★★★☆      |
| `BTC_news_price_gui.py`                  | 尝试将新闻进行向量化/学习后，与价格做联合建模（实验性，尚未成熟）        | Binance + 多来源新闻  | GUI 界面（可能使用 tkinter/pyqt） | ✓        | GUI      | ★★☆☆☆      |

### 目前推荐使用顺序（2025年12月）

1. **最推荐**：`kronos_predict_optimized_interactive.py`  
   → 目前功能最完整、交互体验最好、情绪修正比较合理  
   → 会生成一个很好看的 Plotly 交互 HTML 报告，包含价格预测曲线 + 可点击的情绪新闻弹窗

2. 如果想要更完整的新闻列表展示：  
   → `kronos_predict_with_sentiment_combined.py`

3. 只想看最纯粹的 Kronos 预测效果（不带情绪）：  
   → `kronos_predict.py`

4. 目前**不推荐**优先运行：`BTC_news_price_gui.py`  
   （还在早期实验阶段，模型思路尚未验证稳定，代码也可能比较乱）

## 使用快速指引

```bash
# 1. 推荐从这个脚本开始（目前最完整版本）
python kronos_predict_optimized_interactive.py

# 运行完成后会在当前目录生成：
# - btc_prediction_with_sentiment.html     ← 打开这个看最漂亮的交互报告
# - btc_1h_prediction_pure.csv
# - btc_1h_prediction_with_news.csv